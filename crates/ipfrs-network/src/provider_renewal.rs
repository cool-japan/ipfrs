//! DHT provider record auto-renewal scheduler.
//!
//! This module implements [`ProviderRenewalScheduler`], which tracks CIDs that
//! have been announced to the Kademlia DHT as provider records and schedules
//! automatic re-announcement before the standard 24-hour TTL expires.
//!
//! ## Background
//!
//! The Kademlia specification (and the libp2p/IPFS implementation) assigns a
//! 24-hour TTL to provider records.  If a node does not re-publish a record
//! before the TTL elapses, remote peers will stop discovering that node as a
//! provider for the corresponding CID.  The scheduler triggers renewal at 80%
//! of the TTL (configurable), giving a comfortable safety margin.
//!
//! ## Usage
//!
//! ```rust
//! use ipfrs_network::provider_renewal::{ProviderRenewalScheduler, RenewalConfig};
//! use std::time::{SystemTime, UNIX_EPOCH};
//!
//! let scheduler = ProviderRenewalScheduler::new(RenewalConfig::default());
//!
//! let now = SystemTime::now()
//!     .duration_since(UNIX_EPOCH)
//!     .unwrap_or_default()
//!     .as_secs();
//!
//! scheduler.track("bafybeihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku", now);
//!
//! // Later, check what needs renewal:
//! let due = scheduler.due_for_renewal(now + 70_000);
//! scheduler.mark_renewed(&due, now + 70_000);
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Default DHT provider record TTL (24 hours in seconds, per Kademlia spec).
pub const DEFAULT_PROVIDER_TTL_SECS: u64 = 86_400;

/// Renew when this fraction of TTL has elapsed (80 %).
pub const DEFAULT_RENEWAL_THRESHOLD: f64 = 0.80;

/// Default maximum number of CIDs to track simultaneously.
const DEFAULT_MAX_TRACKED_CIDS: usize = 100_000;

// ─────────────────────────────────────────────────────────────────────────────
// RenewalConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the [`ProviderRenewalScheduler`].
#[derive(Debug, Clone)]
pub struct RenewalConfig {
    /// Provider record TTL in seconds.  Default: 86 400 (24 h).
    pub ttl_secs: u64,

    /// Fraction of TTL at which to trigger renewal.  Default: 0.80.
    pub renewal_threshold: f64,

    /// Maximum CIDs to track simultaneously.  Default: 100 000.
    pub max_tracked_cids: usize,
}

impl Default for RenewalConfig {
    fn default() -> Self {
        Self {
            ttl_secs: DEFAULT_PROVIDER_TTL_SECS,
            renewal_threshold: DEFAULT_RENEWAL_THRESHOLD,
            max_tracked_cids: DEFAULT_MAX_TRACKED_CIDS,
        }
    }
}

impl RenewalConfig {
    /// Number of seconds after which a record should be renewed.
    ///
    /// Computed as `ttl_secs * renewal_threshold`, truncated to an integer.
    pub fn renewal_interval_secs(&self) -> u64 {
        (self.ttl_secs as f64 * self.renewal_threshold) as u64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ProviderRecord
// ─────────────────────────────────────────────────────────────────────────────

/// State for a single provided CID tracked by the scheduler.
#[derive(Debug, Clone)]
pub struct ProviderRecord {
    /// String representation of the CID.
    pub cid_str: String,

    /// Unix timestamp (seconds) at which `track()` was first called.
    pub provided_at_secs: u64,

    /// Unix timestamp (seconds) of the most recent successful renewal.
    /// `0` means the record has never been explicitly renewed since tracking
    /// began.
    pub last_renewal_secs: u64,

    /// How many times this record has been successfully renewed.
    pub renewal_count: u64,
}

impl ProviderRecord {
    /// Create a new record anchored at `now_secs`.
    pub fn new(cid_str: impl Into<String>, now_secs: u64) -> Self {
        Self {
            cid_str: cid_str.into(),
            provided_at_secs: now_secs,
            last_renewal_secs: 0,
            renewal_count: 0,
        }
    }

    /// Returns `true` when the record is old enough to require renewal.
    ///
    /// Age is measured from the *later* of `provided_at_secs` and
    /// `last_renewal_secs` (treating `0` as "never renewed").
    pub fn needs_renewal(&self, now_secs: u64, config: &RenewalConfig) -> bool {
        let baseline = if self.last_renewal_secs == 0 {
            self.provided_at_secs
        } else {
            self.last_renewal_secs.max(self.provided_at_secs)
        };
        let age = now_secs.saturating_sub(baseline);
        age >= config.renewal_interval_secs()
    }

    /// Update the record after a successful renewal.
    pub fn mark_renewed(&mut self, now_secs: u64) {
        self.last_renewal_secs = now_secs;
        self.renewal_count += 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ProviderRenewalScheduler
// ─────────────────────────────────────────────────────────────────────────────

/// Manages scheduled renewal of DHT provider records.
///
/// The scheduler is cheap to clone because it is `Arc`-wrapped internally.
/// All methods take `&self` to allow concurrent access from multiple tasks.
pub struct ProviderRenewalScheduler {
    config: RenewalConfig,
    records: RwLock<HashMap<String, ProviderRecord>>,
    total_renewals: AtomicU64,
}

impl ProviderRenewalScheduler {
    /// Create a new scheduler wrapped in an [`Arc`].
    pub fn new(config: RenewalConfig) -> Arc<Self> {
        Arc::new(Self {
            config,
            records: RwLock::new(HashMap::new()),
            total_renewals: AtomicU64::new(0),
        })
    }

    /// Register a CID for renewal tracking.
    ///
    /// Returns `true` if the CID was newly inserted or was already tracked.
    /// Returns `false` when the capacity limit is reached and the CID is new.
    pub fn track(&self, cid_str: impl Into<String>, now_secs: u64) -> bool {
        let key: String = cid_str.into();

        // Fast-path: already tracked — just update the timestamp.
        {
            let mut records = self.records.write();
            if records.contains_key(&key) {
                // Refresh the anchor timestamp so existing records remain valid.
                if let Some(r) = records.get_mut(&key) {
                    r.provided_at_secs = now_secs;
                }
                return true;
            }

            // Capacity check.
            if records.len() >= self.config.max_tracked_cids {
                return false;
            }

            records.insert(key.clone(), ProviderRecord::new(key, now_secs));
        }

        true
    }

    /// Stop tracking a CID (e.g. when the corresponding block is deleted).
    ///
    /// Returns `true` if the record existed and was removed.
    pub fn untrack(&self, cid_str: &str) -> bool {
        self.records.write().remove(cid_str).is_some()
    }

    /// Returns all CIDs that are due for renewal at `now_secs`.
    pub fn due_for_renewal(&self, now_secs: u64) -> Vec<String> {
        self.records
            .read()
            .values()
            .filter(|r| r.needs_renewal(now_secs, &self.config))
            .map(|r| r.cid_str.clone())
            .collect()
    }

    /// Mark a slice of CIDs as successfully renewed at `now_secs`.
    ///
    /// Returns the number of records that were actually updated (i.e. were
    /// present in the tracker).
    pub fn mark_renewed(&self, cid_strs: &[String], now_secs: u64) -> usize {
        let mut count = 0usize;
        let mut records = self.records.write();

        for key in cid_strs {
            if let Some(r) = records.get_mut(key.as_str()) {
                r.mark_renewed(now_secs);
                count += 1;
            }
        }

        if count > 0 {
            self.total_renewals
                .fetch_add(count as u64, Ordering::Relaxed);
        }

        count
    }

    /// Total number of CIDs currently tracked.
    pub fn tracked_count(&self) -> usize {
        self.records.read().len()
    }

    /// Total successful renewals across all CIDs since the scheduler was
    /// created.
    pub fn total_renewals(&self) -> u64 {
        self.total_renewals.load(Ordering::Relaxed)
    }

    /// Return a point-in-time snapshot of all tracked records.
    pub fn snapshot(&self) -> Vec<ProviderRecord> {
        self.records.read().values().cloned().collect()
    }

    /// Remove records that are definitively stale: the record has *never* been
    /// renewed (`last_renewal_secs == 0`) and its age exceeds `2 × TTL`.
    ///
    /// Returns the number of records pruned.
    pub fn prune_stale(&self, now_secs: u64) -> usize {
        let cutoff = self.config.ttl_secs.saturating_mul(2);
        let mut records = self.records.write();
        let before = records.len();

        records.retain(|_, r| {
            if r.last_renewal_secs == 0 {
                let age = now_secs.saturating_sub(r.provided_at_secs);
                age <= cutoff
            } else {
                true
            }
        });

        before - records.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Convenience: epoch-like base timestamp that is large enough to avoid
    // saturating-sub oddities.
    const T0: u64 = 1_700_000_000;

    fn default_scheduler() -> Arc<ProviderRenewalScheduler> {
        ProviderRenewalScheduler::new(RenewalConfig::default())
    }

    // ── Config ────────────────────────────────────────────────────────────────

    #[test]
    fn test_default_config_renewal_interval() {
        let cfg = RenewalConfig::default();
        // 86_400 * 0.80 = 69_120
        assert_eq!(cfg.renewal_interval_secs(), 69_120);
    }

    #[test]
    fn test_renewal_config_custom() {
        let cfg = RenewalConfig {
            ttl_secs: 3_600,
            renewal_threshold: 0.75,
            max_tracked_cids: 50,
        };
        // 3600 * 0.75 = 2700
        assert_eq!(cfg.renewal_interval_secs(), 2_700);
        assert_eq!(cfg.max_tracked_cids, 50);
    }

    // ── Track / Untrack ───────────────────────────────────────────────────────

    #[test]
    fn test_track_and_count() {
        let sched = default_scheduler();
        assert_eq!(sched.tracked_count(), 0);

        assert!(sched.track("cid1", T0));
        assert!(sched.track("cid2", T0));
        assert_eq!(sched.tracked_count(), 2);
    }

    #[test]
    fn test_untrack() {
        let sched = default_scheduler();
        sched.track("cid1", T0);
        assert_eq!(sched.tracked_count(), 1);

        assert!(sched.untrack("cid1"));
        assert_eq!(sched.tracked_count(), 0);

        // Removing a non-existent key returns false.
        assert!(!sched.untrack("cid1"));
    }

    // ── Renewal detection ─────────────────────────────────────────────────────

    #[test]
    fn test_due_for_renewal_empty_initially() {
        let sched = default_scheduler();
        sched.track("cid1", T0);

        // Immediately after tracking, nothing should be due.
        let due = sched.due_for_renewal(T0);
        assert!(
            due.is_empty(),
            "expected no renewals immediately: got {due:?}"
        );
    }

    #[test]
    fn test_due_for_renewal_after_threshold() {
        let sched = default_scheduler();
        sched.track("cid1", T0);

        // Advance time by exactly the renewal interval.
        let interval = sched.config.renewal_interval_secs();
        let due = sched.due_for_renewal(T0 + interval);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0], "cid1");
    }

    #[test]
    fn test_mark_renewed_resets_timer() {
        let sched = default_scheduler();
        sched.track("cid1", T0);

        let interval = sched.config.renewal_interval_secs();
        let t1 = T0 + interval; // due now

        // Confirm it is due.
        assert_eq!(sched.due_for_renewal(t1).len(), 1);

        // Renew it.
        sched.mark_renewed(&["cid1".to_string()], t1);

        // Should no longer be due right after renewal.
        assert!(sched.due_for_renewal(t1).is_empty());

        // Should become due again after another full interval.
        let t2 = t1 + interval;
        assert_eq!(sched.due_for_renewal(t2).len(), 1);
    }

    #[test]
    fn test_mark_renewed_increments_count() {
        let sched = default_scheduler();
        sched.track("cid1", T0);

        let interval = sched.config.renewal_interval_secs();

        // Renew twice.
        sched.mark_renewed(&["cid1".to_string()], T0 + interval);
        sched.mark_renewed(&["cid1".to_string()], T0 + interval * 2);

        let snap = sched.snapshot();
        let rec = snap
            .iter()
            .find(|r| r.cid_str == "cid1")
            .expect("record missing");
        assert_eq!(rec.renewal_count, 2);
    }

    // ── Total-renewals counter ─────────────────────────────────────────────────

    #[test]
    fn test_total_renewals_counter() {
        let sched = default_scheduler();
        sched.track("cid1", T0);
        sched.track("cid2", T0);
        sched.track("cid3", T0);

        assert_eq!(sched.total_renewals(), 0);

        let interval = sched.config.renewal_interval_secs();
        let renewed = sched.mark_renewed(
            &["cid1".to_string(), "cid2".to_string(), "cid3".to_string()],
            T0 + interval,
        );
        assert_eq!(renewed, 3);
        assert_eq!(sched.total_renewals(), 3);

        // Second batch.
        let renewed2 = sched.mark_renewed(&["cid1".to_string()], T0 + interval * 2);
        assert_eq!(renewed2, 1);
        assert_eq!(sched.total_renewals(), 4);
    }

    // ── Capacity limit ────────────────────────────────────────────────────────

    #[test]
    fn test_capacity_limit() {
        let cfg = RenewalConfig {
            max_tracked_cids: 3,
            ..Default::default()
        };
        let sched = ProviderRenewalScheduler::new(cfg);

        assert!(sched.track("cid1", T0));
        assert!(sched.track("cid2", T0));
        assert!(sched.track("cid3", T0));

        // 4th new CID must be rejected.
        assert!(!sched.track("cid4", T0));
        assert_eq!(sched.tracked_count(), 3);

        // Existing CIDs must still be re-trackable (idempotent).
        assert!(sched.track("cid1", T0 + 1));
        assert_eq!(sched.tracked_count(), 3);
    }

    // ── Prune stale ───────────────────────────────────────────────────────────

    #[test]
    fn test_prune_stale_removes_old_records() {
        let sched = default_scheduler();
        let ttl = sched.config.ttl_secs;

        sched.track("old_cid", T0);
        sched.track("fresh_cid", T0);

        // Advance time so "old_cid" is older than 2×TTL but "fresh_cid" is not.
        let old_enough = T0 + ttl * 2 + 1;

        // Manually update fresh_cid so it was tracked recently.
        sched.track("fresh_cid", old_enough - 100); // re-track with recent time

        let pruned = sched.prune_stale(old_enough);
        assert_eq!(pruned, 1, "expected exactly 1 stale record pruned");
        assert_eq!(sched.tracked_count(), 1);

        let snap = sched.snapshot();
        assert_eq!(snap[0].cid_str, "fresh_cid");
    }

    // ── Snapshot ─────────────────────────────────────────────────────────────

    #[test]
    fn test_snapshot() {
        let sched = default_scheduler();
        sched.track("a", T0);
        sched.track("b", T0);
        sched.track("c", T0);

        let snap = sched.snapshot();
        assert_eq!(snap.len(), 3);

        // All CIDs must be present.
        let mut cids: Vec<&str> = snap.iter().map(|r| r.cid_str.as_str()).collect();
        cids.sort_unstable();
        assert_eq!(cids, vec!["a", "b", "c"]);
    }
}
