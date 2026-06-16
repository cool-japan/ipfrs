//! Peer Announcement Manager
//!
//! Manages CID announcements to the DHT and GossipSub, tracking which content
//! has been announced, when, and with what success rate.

use std::collections::HashMap;

/// Channel through which a CID is announced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AnnouncementChannel {
    /// Announce via the Kademlia DHT.
    Dht,
    /// Announce via GossipSub pub/sub.
    Gossipsub,
    /// Announce via both DHT and GossipSub.
    Both,
}

/// A single record tracking the announcement of a CID.
#[derive(Clone, Debug)]
pub struct AnnouncementRecord {
    /// The content identifier being announced.
    pub cid: String,
    /// The channel(s) used for announcement.
    pub channel: AnnouncementChannel,
    /// The tick at which the announcement was created or last re-announced.
    pub announced_at_tick: u64,
    /// The tick at which this record expires.
    pub expiry_tick: u64,
    /// Whether the announcement succeeded.
    pub successful: bool,
    /// How many times this announcement has been retried after failure.
    pub retry_count: u32,
}

impl AnnouncementRecord {
    /// Returns `true` when `current_tick` has reached or passed `expiry_tick`.
    pub fn is_expired(&self, current_tick: u64) -> bool {
        current_tick >= self.expiry_tick
    }

    /// Returns `true` when the record has not succeeded and has retries remaining.
    pub fn should_retry(&self, max_retries: u32) -> bool {
        !self.successful && self.retry_count < max_retries
    }
}

/// Configuration for the `PeerAnnouncementManager`.
#[derive(Clone, Debug)]
pub struct AnnouncementConfig {
    /// How many ticks a record lives before it is considered expired.
    pub ttl_ticks: u64,
    /// Maximum number of retry attempts for a failed announcement.
    pub max_retries: u32,
    /// Ticks between successive re-announcements of already-successful records.
    pub reannounce_interval_ticks: u64,
}

impl Default for AnnouncementConfig {
    fn default() -> Self {
        Self {
            ttl_ticks: 1000,
            max_retries: 3,
            reannounce_interval_ticks: 500,
        }
    }
}

/// Aggregate statistics derived from the current state of the manager.
#[derive(Clone, Debug, Default)]
pub struct AnnouncementStats {
    /// Total number of active (non-evicted) records.
    pub total_announced: usize,
    /// Records that have `successful == true`.
    pub successful_count: usize,
    /// Records that have failed (not successful, exhausted retries, or still pending failure).
    pub failed_count: usize,
    /// Records moved to `expired_records` (evicted).
    pub expired_count: usize,
    /// Active records that still have retries remaining.
    pub pending_retry_count: usize,
}

impl AnnouncementStats {
    /// Returns `successful / (successful + failed)`, or `0.0` when both are zero.
    pub fn success_rate(&self) -> f64 {
        let total = self.successful_count + self.failed_count;
        if total == 0 {
            0.0
        } else {
            self.successful_count as f64 / total as f64
        }
    }
}

/// Manages CID announcements to DHT and GossipSub.
pub struct PeerAnnouncementManager {
    /// Active announcement records, keyed by CID string.
    pub records: HashMap<String, AnnouncementRecord>,
    /// Configuration controlling TTL, retries, and re-announce intervals.
    pub config: AnnouncementConfig,
    /// Records that have been evicted due to expiry.
    pub expired_records: Vec<AnnouncementRecord>,
}

impl PeerAnnouncementManager {
    /// Creates a new manager with the given configuration and no records.
    pub fn new(config: AnnouncementConfig) -> Self {
        Self {
            records: HashMap::new(),
            config,
            expired_records: Vec::new(),
        }
    }

    /// Attempts to announce `cid` on `channel` at logical time `tick`.
    ///
    /// Returns `false` if the CID already has a non-expired, successful record
    /// (indicating re-announcement is not needed yet).  Returns `true` when a
    /// new record is created or an existing failed/expired record is overwritten.
    pub fn announce(&mut self, cid: &str, channel: AnnouncementChannel, tick: u64) -> bool {
        if let Some(existing) = self.records.get(cid) {
            if existing.successful && !existing.is_expired(tick) {
                return false;
            }
        }

        let record = AnnouncementRecord {
            cid: cid.to_owned(),
            channel,
            announced_at_tick: tick,
            expiry_tick: tick + self.config.ttl_ticks,
            successful: false,
            retry_count: 0,
        };
        self.records.insert(cid.to_owned(), record);
        true
    }

    /// Marks the record for `cid` as successful.
    ///
    /// Returns `true` on success, `false` if `cid` is not tracked.
    pub fn mark_success(&mut self, cid: &str) -> bool {
        match self.records.get_mut(cid) {
            Some(record) => {
                record.successful = true;
                true
            }
            None => false,
        }
    }

    /// Increments the `retry_count` for the record belonging to `cid`.
    ///
    /// Returns `true` on success, `false` if `cid` is not tracked.
    pub fn mark_failure(&mut self, cid: &str) -> bool {
        match self.records.get_mut(cid) {
            Some(record) => {
                record.retry_count += 1;
                true
            }
            None => false,
        }
    }

    /// Returns `true` when `cid` has a successful, non-expired record whose age
    /// (in ticks since `announced_at_tick`) has met or exceeded `reannounce_interval_ticks`.
    pub fn needs_reannounce(&self, cid: &str, current_tick: u64) -> bool {
        match self.records.get(cid) {
            Some(record) => {
                record.successful
                    && !record.is_expired(current_tick)
                    && current_tick.saturating_sub(record.announced_at_tick)
                        >= self.config.reannounce_interval_ticks
            }
            None => false,
        }
    }

    /// Moves all expired records out of `records` and into `expired_records`.
    pub fn evict_expired(&mut self, current_tick: u64) {
        let expired_keys: Vec<String> = self
            .records
            .iter()
            .filter(|(_, r)| r.is_expired(current_tick))
            .map(|(k, _)| k.clone())
            .collect();

        for key in expired_keys {
            if let Some(record) = self.records.remove(&key) {
                self.expired_records.push(record);
            }
        }
    }

    /// Returns references to all active records that should be retried,
    /// sorted by `retry_count` ascending (fewest retries first).
    pub fn pending_retries(&self, current_tick: u64) -> Vec<&AnnouncementRecord> {
        let max_retries = self.config.max_retries;
        let mut pending: Vec<&AnnouncementRecord> = self
            .records
            .values()
            .filter(|r| r.should_retry(max_retries) && !r.is_expired(current_tick))
            .collect();
        pending.sort_by_key(|r| r.retry_count);
        pending
    }

    /// Returns a reference to the record for `cid`, if it exists.
    pub fn get(&self, cid: &str) -> Option<&AnnouncementRecord> {
        self.records.get(cid)
    }

    /// Computes aggregate statistics from the current state.
    pub fn stats(&self, current_tick: u64) -> AnnouncementStats {
        let max_retries = self.config.max_retries;
        let mut successful_count = 0usize;
        let mut failed_count = 0usize;
        let mut pending_retry_count = 0usize;

        for record in self.records.values() {
            if record.successful {
                successful_count += 1;
            } else {
                failed_count += 1;
            }
            if record.should_retry(max_retries) && !record.is_expired(current_tick) {
                pending_retry_count += 1;
            }
        }

        AnnouncementStats {
            total_announced: self.records.len(),
            successful_count,
            failed_count,
            expired_count: self.expired_records.len(),
            pending_retry_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_manager() -> PeerAnnouncementManager {
        PeerAnnouncementManager::new(AnnouncementConfig::default())
    }

    // ── construction ─────────────────────────────────────────────────────────

    #[test]
    fn test_new_starts_empty() {
        let mgr = default_manager();
        assert!(mgr.records.is_empty());
        assert!(mgr.expired_records.is_empty());
    }

    // ── announce ─────────────────────────────────────────────────────────────

    #[test]
    fn test_announce_creates_record() {
        let mut mgr = default_manager();
        let result = mgr.announce("cid1", AnnouncementChannel::Dht, 0);
        assert!(result);
        assert!(mgr.get("cid1").is_some());
    }

    #[test]
    fn test_announce_sets_fields_correctly() {
        let mut mgr = default_manager();
        mgr.announce("cid1", AnnouncementChannel::Gossipsub, 10);
        let record = mgr
            .get("cid1")
            .expect("test: record should exist after announce");
        assert_eq!(record.channel, AnnouncementChannel::Gossipsub);
        assert_eq!(record.announced_at_tick, 10);
        assert_eq!(record.expiry_tick, 10 + mgr.config.ttl_ticks);
        assert!(!record.successful);
        assert_eq!(record.retry_count, 0);
    }

    #[test]
    fn test_announce_returns_false_for_successful_non_expired() {
        let mut mgr = default_manager();
        mgr.announce("cid1", AnnouncementChannel::Dht, 0);
        mgr.mark_success("cid1");
        // Second announce while still successful and not expired => false
        let result = mgr.announce("cid1", AnnouncementChannel::Dht, 1);
        assert!(!result);
    }

    #[test]
    fn test_announce_returns_true_for_failed_record() {
        let mut mgr = default_manager();
        mgr.announce("cid1", AnnouncementChannel::Dht, 0);
        // record is failed (successful == false) => re-announce allowed
        let result = mgr.announce("cid1", AnnouncementChannel::Both, 1);
        assert!(result);
    }

    #[test]
    fn test_announce_returns_true_for_expired_successful_record() {
        let config = AnnouncementConfig {
            ttl_ticks: 10,
            ..Default::default()
        };
        let mut mgr = PeerAnnouncementManager::new(config);
        mgr.announce("cid1", AnnouncementChannel::Dht, 0);
        mgr.mark_success("cid1");
        // Tick 10 => expiry_tick == 10 => is_expired
        let result = mgr.announce("cid1", AnnouncementChannel::Dht, 10);
        assert!(result);
    }

    // ── mark_success ─────────────────────────────────────────────────────────

    #[test]
    fn test_mark_success_sets_flag() {
        let mut mgr = default_manager();
        mgr.announce("cid1", AnnouncementChannel::Dht, 0);
        assert!(mgr.mark_success("cid1"));
        assert!(
            mgr.get("cid1")
                .expect("test: record should exist after mark_success")
                .successful
        );
    }

    #[test]
    fn test_mark_success_false_for_unknown() {
        let mut mgr = default_manager();
        assert!(!mgr.mark_success("unknown"));
    }

    // ── mark_failure ─────────────────────────────────────────────────────────

    #[test]
    fn test_mark_failure_increments_retry_count() {
        let mut mgr = default_manager();
        mgr.announce("cid1", AnnouncementChannel::Dht, 0);
        mgr.mark_failure("cid1");
        assert_eq!(
            mgr.get("cid1")
                .expect("test: record should exist after first mark_failure")
                .retry_count,
            1
        );
        mgr.mark_failure("cid1");
        assert_eq!(
            mgr.get("cid1")
                .expect("test: record should exist after second mark_failure")
                .retry_count,
            2
        );
    }

    #[test]
    fn test_mark_failure_false_for_unknown() {
        let mut mgr = default_manager();
        assert!(!mgr.mark_failure("unknown"));
    }

    // ── needs_reannounce ─────────────────────────────────────────────────────

    #[test]
    fn test_needs_reannounce_true_when_interval_elapsed() {
        let config = AnnouncementConfig {
            reannounce_interval_ticks: 100,
            ..Default::default()
        };
        let mut mgr = PeerAnnouncementManager::new(config);
        mgr.announce("cid1", AnnouncementChannel::Dht, 0);
        mgr.mark_success("cid1");
        assert!(mgr.needs_reannounce("cid1", 100));
    }

    #[test]
    fn test_needs_reannounce_false_before_interval() {
        let config = AnnouncementConfig {
            reannounce_interval_ticks: 100,
            ..Default::default()
        };
        let mut mgr = PeerAnnouncementManager::new(config);
        mgr.announce("cid1", AnnouncementChannel::Dht, 0);
        mgr.mark_success("cid1");
        assert!(!mgr.needs_reannounce("cid1", 99));
    }

    #[test]
    fn test_needs_reannounce_false_for_failed_record() {
        let mut mgr = default_manager();
        mgr.announce("cid1", AnnouncementChannel::Dht, 0);
        // Record is NOT marked successful
        assert!(!mgr.needs_reannounce("cid1", 600));
    }

    #[test]
    fn test_needs_reannounce_false_for_unknown() {
        let mgr = default_manager();
        assert!(!mgr.needs_reannounce("nonexistent", 999));
    }

    // ── is_expired / should_retry ─────────────────────────────────────────────

    #[test]
    fn test_is_expired_true_at_expiry() {
        let record = AnnouncementRecord {
            cid: "cid1".to_owned(),
            channel: AnnouncementChannel::Dht,
            announced_at_tick: 0,
            expiry_tick: 100,
            successful: false,
            retry_count: 0,
        };
        assert!(record.is_expired(100));
        assert!(record.is_expired(101));
    }

    #[test]
    fn test_is_expired_false_before_expiry() {
        let record = AnnouncementRecord {
            cid: "cid1".to_owned(),
            channel: AnnouncementChannel::Dht,
            announced_at_tick: 0,
            expiry_tick: 100,
            successful: false,
            retry_count: 0,
        };
        assert!(!record.is_expired(99));
    }

    #[test]
    fn test_should_retry_true_when_not_successful_and_retries_remaining() {
        let record = AnnouncementRecord {
            cid: "cid1".to_owned(),
            channel: AnnouncementChannel::Dht,
            announced_at_tick: 0,
            expiry_tick: 1000,
            successful: false,
            retry_count: 2,
        };
        assert!(record.should_retry(3));
    }

    #[test]
    fn test_should_retry_false_at_max_retries() {
        let record = AnnouncementRecord {
            cid: "cid1".to_owned(),
            channel: AnnouncementChannel::Dht,
            announced_at_tick: 0,
            expiry_tick: 1000,
            successful: false,
            retry_count: 3,
        };
        assert!(!record.should_retry(3));
    }

    #[test]
    fn test_should_retry_false_when_successful() {
        let record = AnnouncementRecord {
            cid: "cid1".to_owned(),
            channel: AnnouncementChannel::Dht,
            announced_at_tick: 0,
            expiry_tick: 1000,
            successful: true,
            retry_count: 0,
        };
        assert!(!record.should_retry(3));
    }

    // ── evict_expired ─────────────────────────────────────────────────────────

    #[test]
    fn test_evict_expired_moves_records() {
        let config = AnnouncementConfig {
            ttl_ticks: 10,
            ..Default::default()
        };
        let mut mgr = PeerAnnouncementManager::new(config);
        mgr.announce("cid1", AnnouncementChannel::Dht, 0);
        mgr.evict_expired(10); // expiry_tick == 10 => expired
        assert!(!mgr.records.contains_key("cid1"));
        assert_eq!(mgr.expired_records.len(), 1);
        assert_eq!(mgr.expired_records[0].cid, "cid1");
    }

    #[test]
    fn test_evict_expired_keeps_fresh_records() {
        let config = AnnouncementConfig {
            ttl_ticks: 100,
            ..Default::default()
        };
        let mut mgr = PeerAnnouncementManager::new(config);
        mgr.announce("cid1", AnnouncementChannel::Dht, 0);
        mgr.evict_expired(50); // expiry_tick == 100 > 50 => fresh
        assert!(mgr.records.contains_key("cid1"));
        assert!(mgr.expired_records.is_empty());
    }

    #[test]
    fn test_evict_expired_mixed() {
        let config = AnnouncementConfig {
            ttl_ticks: 10,
            ..Default::default()
        };
        let mut mgr = PeerAnnouncementManager::new(config);
        mgr.announce("old", AnnouncementChannel::Dht, 0); // expires at 10
        mgr.announce("new", AnnouncementChannel::Dht, 5); // expires at 15
        mgr.evict_expired(12);
        assert!(!mgr.records.contains_key("old"));
        assert!(mgr.records.contains_key("new"));
        assert_eq!(mgr.expired_records.len(), 1);
    }

    // ── pending_retries ───────────────────────────────────────────────────────

    #[test]
    fn test_pending_retries_filtered_correctly() {
        let config = AnnouncementConfig {
            max_retries: 3,
            ttl_ticks: 1000,
            ..Default::default()
        };
        let mut mgr = PeerAnnouncementManager::new(config);
        mgr.announce("a", AnnouncementChannel::Dht, 0);
        mgr.mark_failure("a"); // retry_count=1, max_retries=3 => pending
        mgr.announce("b", AnnouncementChannel::Dht, 0);
        mgr.mark_success("b"); // successful => not pending

        let pending = mgr.pending_retries(0);
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].cid, "a");
    }

    #[test]
    fn test_pending_retries_sorted_by_retry_count_asc() {
        let config = AnnouncementConfig {
            max_retries: 5,
            ttl_ticks: 1000,
            ..Default::default()
        };
        let mut mgr = PeerAnnouncementManager::new(config);
        mgr.announce("a", AnnouncementChannel::Dht, 0);
        mgr.mark_failure("a");
        mgr.mark_failure("a"); // retry_count=2

        mgr.announce("b", AnnouncementChannel::Dht, 0);
        mgr.mark_failure("b"); // retry_count=1

        mgr.announce("c", AnnouncementChannel::Dht, 0);
        mgr.mark_failure("c");
        mgr.mark_failure("c");
        mgr.mark_failure("c"); // retry_count=3

        let pending = mgr.pending_retries(0);
        assert_eq!(pending.len(), 3);
        // Should be sorted ascending: 1, 2, 3
        assert_eq!(pending[0].retry_count, 1);
        assert_eq!(pending[1].retry_count, 2);
        assert_eq!(pending[2].retry_count, 3);
    }

    #[test]
    fn test_pending_retries_excludes_expired() {
        let config = AnnouncementConfig {
            max_retries: 3,
            ttl_ticks: 10,
            ..Default::default()
        };
        let mut mgr = PeerAnnouncementManager::new(config);
        mgr.announce("cid1", AnnouncementChannel::Dht, 0);
        mgr.mark_failure("cid1");
        // At tick 10 the record is expired => not in pending_retries
        let pending = mgr.pending_retries(10);
        assert!(pending.is_empty());
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_success_rate_computed() {
        let mut mgr = default_manager();
        mgr.announce("a", AnnouncementChannel::Dht, 0);
        mgr.mark_success("a");
        mgr.announce("b", AnnouncementChannel::Dht, 0);
        // b remains failed

        let stats = mgr.stats(0);
        assert_eq!(stats.successful_count, 1);
        assert_eq!(stats.failed_count, 1);
        assert!((stats.success_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_success_rate_zero_when_both_zero() {
        let mgr = default_manager();
        let stats = mgr.stats(0);
        assert_eq!(stats.success_rate(), 0.0);
    }

    #[test]
    fn test_stats_expired_count_reflects_evictions() {
        let config = AnnouncementConfig {
            ttl_ticks: 5,
            ..Default::default()
        };
        let mut mgr = PeerAnnouncementManager::new(config);
        mgr.announce("cid1", AnnouncementChannel::Dht, 0);
        mgr.evict_expired(5);

        let stats = mgr.stats(5);
        assert_eq!(stats.expired_count, 1);
        assert_eq!(stats.total_announced, 0);
    }

    #[test]
    fn test_stats_pending_retry_count() {
        let config = AnnouncementConfig {
            max_retries: 3,
            ttl_ticks: 1000,
            ..Default::default()
        };
        let mut mgr = PeerAnnouncementManager::new(config);
        mgr.announce("a", AnnouncementChannel::Dht, 0);
        mgr.mark_failure("a");

        let stats = mgr.stats(0);
        assert_eq!(stats.pending_retry_count, 1);
    }
}
