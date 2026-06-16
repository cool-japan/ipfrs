//! Routing Table Auditor
//!
//! Audits the DHT routing table for health issues including stale entries,
//! bucket imbalances, and unreachable peers.

/// Severity level of an audit finding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AuditSeverity {
    /// Informational finding; no action required.
    Info = 0,
    /// Warning finding; should be investigated.
    Warning = 1,
    /// Error finding; requires immediate attention.
    Error = 2,
}

/// A single finding produced by a routing-table audit run.
#[derive(Debug, Clone)]
pub struct AuditFinding {
    /// How serious this finding is.
    pub severity: AuditSeverity,
    /// Machine-readable category such as `"stale_entry"` or `"empty_buckets"`.
    pub category: String,
    /// Human-readable description of the finding.
    pub description: String,
    /// Peer IDs (or bucket identifiers) involved in this finding.
    pub affected_peers: Vec<String>,
    /// Unix timestamp (seconds) at which the finding was detected.
    pub detected_at_secs: u64,
}

/// Default k-bucket capacity as defined by the Kademlia paper.
pub const DEFAULT_MAX_CAPACITY: usize = 20;

/// Information about a single Kademlia k-bucket.
#[derive(Debug, Clone)]
pub struct BucketInfo {
    /// Kademlia bucket index (0–255).
    pub bucket_index: usize,
    /// Number of peers currently tracked in this bucket.
    pub peer_count: usize,
    /// Maximum number of peers this bucket can hold.
    pub max_capacity: usize,
    /// Unix timestamp (seconds) of the last time this bucket was refreshed.
    pub last_refreshed_secs: u64,
}

impl BucketInfo {
    /// Returns `true` when the bucket has reached its maximum capacity.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.peer_count >= self.max_capacity
    }

    /// Returns `true` when the bucket contains no peers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.peer_count == 0
    }

    /// Returns the ratio of current peers to maximum capacity (0.0 – 1.0+).
    #[must_use]
    pub fn fill_ratio(&self) -> f64 {
        if self.max_capacity == 0 {
            return 0.0;
        }
        self.peer_count as f64 / self.max_capacity as f64
    }
}

/// Configuration for the [`RoutingTableAuditor`].
#[derive(Debug, Clone)]
pub struct AuditorConfig {
    /// An entry is considered stale if it has not been refreshed within this
    /// many seconds (default: 3600 = 1 hour).
    pub stale_threshold_secs: u64,
    /// Emit a warning when the number of empty buckets exceeds this value
    /// (default: 10).
    pub max_empty_buckets: usize,
    /// Emit an error when the total number of tracked peers falls below this
    /// value (default: 3).
    pub min_total_peers: usize,
}

impl Default for AuditorConfig {
    fn default() -> Self {
        Self {
            stale_threshold_secs: 3600,
            max_empty_buckets: 10,
            min_total_peers: 3,
        }
    }
}

/// Audits a DHT routing table represented as a collection of [`BucketInfo`]
/// entries and produces [`AuditFinding`] items describing any health issues.
#[derive(Debug, Clone)]
pub struct RoutingTableAuditor {
    /// Up to 256 Kademlia k-buckets tracked by this auditor.
    pub buckets: Vec<BucketInfo>,
    /// Configuration driving audit thresholds.
    pub config: AuditorConfig,
}

impl RoutingTableAuditor {
    /// Creates a new auditor with the given configuration and an empty bucket
    /// list.
    #[must_use]
    pub fn new(config: AuditorConfig) -> Self {
        Self {
            buckets: Vec::new(),
            config,
        }
    }

    /// Appends a bucket to the auditor's bucket list.
    pub fn add_bucket(&mut self, bucket: BucketInfo) {
        self.buckets.push(bucket);
    }

    /// Updates an existing bucket identified by `bucket_index`.  If no bucket
    /// with that index exists a new one is inserted with
    /// [`DEFAULT_MAX_CAPACITY`].
    pub fn update_bucket(&mut self, bucket_index: usize, peer_count: usize, now_secs: u64) {
        if let Some(b) = self
            .buckets
            .iter_mut()
            .find(|b| b.bucket_index == bucket_index)
        {
            b.peer_count = peer_count;
            b.last_refreshed_secs = now_secs;
        } else {
            self.buckets.push(BucketInfo {
                bucket_index,
                peer_count,
                max_capacity: DEFAULT_MAX_CAPACITY,
                last_refreshed_secs: now_secs,
            });
        }
    }

    /// Runs the full audit at time `now_secs` and returns all findings sorted
    /// by severity descending (most severe first).
    ///
    /// Checks performed:
    /// 1. **insufficient_peers** (Error) – total peer count < `min_total_peers`.
    /// 2. **empty_buckets** (Warning) – count of empty buckets > `max_empty_buckets`.
    /// 3. **stale_bucket** (Warning) – each bucket whose `last_refreshed_secs`
    ///    is older than `stale_threshold_secs`.
    /// 4. **full_bucket** (Info) – each bucket at maximum capacity.
    #[must_use]
    pub fn audit(&self, now_secs: u64) -> Vec<AuditFinding> {
        let mut findings: Vec<AuditFinding> = Vec::new();

        // 1. Insufficient peers
        let total = self.total_peers();
        if total < self.config.min_total_peers {
            findings.push(AuditFinding {
                severity: AuditSeverity::Error,
                category: "insufficient_peers".to_string(),
                description: format!(
                    "Routing table has only {} peer(s); minimum required is {}.",
                    total, self.config.min_total_peers
                ),
                affected_peers: Vec::new(),
                detected_at_secs: now_secs,
            });
        }

        // 2. Too many empty buckets
        let empty_count = self.empty_bucket_count();
        if empty_count > self.config.max_empty_buckets {
            findings.push(AuditFinding {
                severity: AuditSeverity::Warning,
                category: "empty_buckets".to_string(),
                description: format!(
                    "{} bucket(s) are empty; threshold is {}.",
                    empty_count, self.config.max_empty_buckets
                ),
                affected_peers: Vec::new(),
                detected_at_secs: now_secs,
            });
        }

        // 3. Stale buckets
        for bucket in &self.buckets {
            let age = now_secs.saturating_sub(bucket.last_refreshed_secs);
            if age > self.config.stale_threshold_secs {
                findings.push(AuditFinding {
                    severity: AuditSeverity::Warning,
                    category: "stale_bucket".to_string(),
                    description: format!(
                        "Bucket {} has not been refreshed for {} second(s) (threshold: {}).",
                        bucket.bucket_index, age, self.config.stale_threshold_secs
                    ),
                    affected_peers: vec![format!("bucket-{}", bucket.bucket_index)],
                    detected_at_secs: now_secs,
                });
            }
        }

        // 4. Full buckets
        for bucket in &self.buckets {
            if bucket.is_full() {
                findings.push(AuditFinding {
                    severity: AuditSeverity::Info,
                    category: "full_bucket".to_string(),
                    description: format!(
                        "Bucket {} is at full capacity ({}/{}).",
                        bucket.bucket_index, bucket.peer_count, bucket.max_capacity
                    ),
                    affected_peers: vec![format!("bucket-{}", bucket.bucket_index)],
                    detected_at_secs: now_secs,
                });
            }
        }

        // Sort by severity descending (Error > Warning > Info)
        findings.sort_by_key(|f| std::cmp::Reverse(f.severity));

        findings
    }

    /// Returns the total number of peers across all tracked buckets.
    #[must_use]
    pub fn total_peers(&self) -> usize {
        self.buckets.iter().map(|b| b.peer_count).sum()
    }

    /// Returns the number of buckets that currently contain no peers.
    #[must_use]
    pub fn empty_bucket_count(&self) -> usize {
        self.buckets.iter().filter(|b| b.is_empty()).count()
    }

    /// Returns the number of buckets that have not been refreshed within the
    /// configured `stale_threshold_secs`.
    #[must_use]
    pub fn stale_bucket_count(&self, now_secs: u64) -> usize {
        self.buckets
            .iter()
            .filter(|b| {
                now_secs.saturating_sub(b.last_refreshed_secs) > self.config.stale_threshold_secs
            })
            .count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bucket(index: usize, peer_count: usize, refreshed: u64) -> BucketInfo {
        BucketInfo {
            bucket_index: index,
            peer_count,
            max_capacity: DEFAULT_MAX_CAPACITY,
            last_refreshed_secs: refreshed,
        }
    }

    fn default_auditor() -> RoutingTableAuditor {
        RoutingTableAuditor::new(AuditorConfig::default())
    }

    // ── add_bucket ───────────────────────────────────────────────────────────

    #[test]
    fn test_add_bucket_increases_count() {
        let mut auditor = default_auditor();
        assert_eq!(auditor.buckets.len(), 0);
        auditor.add_bucket(make_bucket(0, 5, 1000));
        assert_eq!(auditor.buckets.len(), 1);
        auditor.add_bucket(make_bucket(1, 3, 1000));
        assert_eq!(auditor.buckets.len(), 2);
    }

    #[test]
    fn test_add_bucket_stores_correct_data() {
        let mut auditor = default_auditor();
        auditor.add_bucket(make_bucket(42, 7, 999));
        let b = &auditor.buckets[0];
        assert_eq!(b.bucket_index, 42);
        assert_eq!(b.peer_count, 7);
        assert_eq!(b.last_refreshed_secs, 999);
    }

    // ── update_bucket ────────────────────────────────────────────────────────

    #[test]
    fn test_update_bucket_updates_existing() {
        let mut auditor = default_auditor();
        auditor.add_bucket(make_bucket(5, 2, 500));
        auditor.update_bucket(5, 10, 1000);
        assert_eq!(auditor.buckets.len(), 1, "no new bucket should be inserted");
        assert_eq!(auditor.buckets[0].peer_count, 10);
        assert_eq!(auditor.buckets[0].last_refreshed_secs, 1000);
    }

    #[test]
    fn test_update_bucket_inserts_when_missing() {
        let mut auditor = default_auditor();
        auditor.update_bucket(7, 4, 2000);
        assert_eq!(auditor.buckets.len(), 1);
        assert_eq!(auditor.buckets[0].bucket_index, 7);
        assert_eq!(auditor.buckets[0].peer_count, 4);
        assert_eq!(auditor.buckets[0].max_capacity, DEFAULT_MAX_CAPACITY);
    }

    #[test]
    fn test_update_bucket_upsert_multiple() {
        let mut auditor = default_auditor();
        auditor.update_bucket(1, 3, 100);
        auditor.update_bucket(2, 5, 200);
        auditor.update_bucket(1, 8, 300); // update existing
        assert_eq!(auditor.buckets.len(), 2);
        let b1 = auditor
            .buckets
            .iter()
            .find(|b| b.bucket_index == 1)
            .expect("test: bucket with index 1 should exist after upsert");
        assert_eq!(b1.peer_count, 8);
        assert_eq!(b1.last_refreshed_secs, 300);
    }

    // ── total_peers ──────────────────────────────────────────────────────────

    #[test]
    fn test_total_peers_empty_table() {
        let auditor = default_auditor();
        assert_eq!(auditor.total_peers(), 0);
    }

    #[test]
    fn test_total_peers_sum() {
        let mut auditor = default_auditor();
        auditor.add_bucket(make_bucket(0, 5, 1000));
        auditor.add_bucket(make_bucket(1, 3, 1000));
        auditor.add_bucket(make_bucket(2, 0, 1000));
        assert_eq!(auditor.total_peers(), 8);
    }

    // ── empty_bucket_count ───────────────────────────────────────────────────

    #[test]
    fn test_empty_bucket_count() {
        let mut auditor = default_auditor();
        auditor.add_bucket(make_bucket(0, 0, 1000));
        auditor.add_bucket(make_bucket(1, 5, 1000));
        auditor.add_bucket(make_bucket(2, 0, 1000));
        assert_eq!(auditor.empty_bucket_count(), 2);
    }

    // ── stale_bucket_count ───────────────────────────────────────────────────

    #[test]
    fn test_stale_bucket_count() {
        let mut auditor = RoutingTableAuditor::new(AuditorConfig {
            stale_threshold_secs: 600,
            ..AuditorConfig::default()
        });
        let now = 10_000u64;
        // refreshed 500 s ago → not stale
        auditor.add_bucket(make_bucket(0, 2, now - 500));
        // refreshed 601 s ago → stale
        auditor.add_bucket(make_bucket(1, 2, now - 601));
        // refreshed 700 s ago → stale
        auditor.add_bucket(make_bucket(2, 2, now - 700));
        assert_eq!(auditor.stale_bucket_count(now), 2);
    }

    // ── fill_ratio, is_full, is_empty ────────────────────────────────────────

    #[test]
    fn test_fill_ratio_zero_peers() {
        let b = make_bucket(0, 0, 0);
        assert!((b.fill_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fill_ratio_half_full() {
        let b = make_bucket(0, 10, 0); // max_capacity = 20
        assert!((b.fill_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fill_ratio_full() {
        let b = make_bucket(0, 20, 0);
        assert!((b.fill_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_is_full_and_is_empty() {
        let empty = make_bucket(0, 0, 0);
        let full = make_bucket(1, 20, 0);
        let partial = make_bucket(2, 10, 0);

        assert!(empty.is_empty());
        assert!(!empty.is_full());

        assert!(full.is_full());
        assert!(!full.is_empty());

        assert!(!partial.is_empty());
        assert!(!partial.is_full());
    }

    // ── audit: insufficient_peers → Error ───────────────────────────────────

    #[test]
    fn test_audit_insufficient_peers_error() {
        let mut auditor = RoutingTableAuditor::new(AuditorConfig {
            min_total_peers: 5,
            max_empty_buckets: 100, // suppress empty-bucket warning
            stale_threshold_secs: 9999,
        });
        auditor.add_bucket(make_bucket(0, 2, 1000));
        let findings = auditor.audit(2000);
        let errors: Vec<_> = findings
            .iter()
            .filter(|f| f.severity == AuditSeverity::Error && f.category == "insufficient_peers")
            .collect();
        assert_eq!(errors.len(), 1, "expected one insufficient_peers error");
    }

    // ── audit: many empty buckets → Warning ──────────────────────────────────

    #[test]
    fn test_audit_many_empty_buckets_warning() {
        let mut auditor = RoutingTableAuditor::new(AuditorConfig {
            max_empty_buckets: 2,
            min_total_peers: 0, // suppress peer error
            stale_threshold_secs: 9999,
        });
        // Add 5 peers so insufficient_peers won't fire, but 3 empty buckets
        auditor.add_bucket(make_bucket(0, 5, 1000));
        auditor.add_bucket(make_bucket(1, 0, 1000));
        auditor.add_bucket(make_bucket(2, 0, 1000));
        auditor.add_bucket(make_bucket(3, 0, 1000));
        let findings = auditor.audit(2000);
        let warnings: Vec<_> = findings
            .iter()
            .filter(|f| f.severity == AuditSeverity::Warning && f.category == "empty_buckets")
            .collect();
        assert_eq!(warnings.len(), 1, "expected one empty_buckets warning");
    }

    // ── audit: stale bucket → Warning ───────────────────────────────────────

    #[test]
    fn test_audit_stale_bucket_warning() {
        let now = 10_000u64;
        let mut auditor = RoutingTableAuditor::new(AuditorConfig {
            stale_threshold_secs: 500,
            min_total_peers: 0,
            max_empty_buckets: 100,
        });
        auditor.add_bucket(make_bucket(0, 3, now - 600)); // stale
        auditor.add_bucket(make_bucket(1, 3, now - 100)); // fresh
        let findings = auditor.audit(now);
        let stale_warnings: Vec<_> = findings
            .iter()
            .filter(|f| f.category == "stale_bucket")
            .collect();
        assert_eq!(stale_warnings.len(), 1);
        assert_eq!(stale_warnings[0].affected_peers, vec!["bucket-0"]);
    }

    // ── audit: full bucket → Info ────────────────────────────────────────────

    #[test]
    fn test_audit_full_bucket_info() {
        let now = 5000u64;
        let mut auditor = RoutingTableAuditor::new(AuditorConfig {
            min_total_peers: 0,
            max_empty_buckets: 100,
            stale_threshold_secs: 9999,
        });
        auditor.add_bucket(make_bucket(3, DEFAULT_MAX_CAPACITY, now));
        let findings = auditor.audit(now);
        let info: Vec<_> = findings
            .iter()
            .filter(|f| f.severity == AuditSeverity::Info && f.category == "full_bucket")
            .collect();
        assert_eq!(info.len(), 1);
        assert_eq!(info[0].affected_peers, vec!["bucket-3"]);
    }

    // ── audit: sort by severity descending ───────────────────────────────────

    #[test]
    fn test_audit_sorted_by_severity_descending() {
        let now = 20_000u64;
        let mut auditor = RoutingTableAuditor::new(AuditorConfig {
            min_total_peers: 10,       // will trigger Error (total = 20 + full)
            max_empty_buckets: 0,      // will trigger Warning (1 empty bucket)
            stale_threshold_secs: 500, // will trigger Warning for old bucket
        });
        // 1 empty bucket → empty_buckets warning
        auditor.add_bucket(make_bucket(0, 0, now));
        // 1 stale bucket
        auditor.add_bucket(make_bucket(1, 2, now - 600));
        // 1 full bucket → full_bucket info
        auditor.add_bucket(make_bucket(2, DEFAULT_MAX_CAPACITY, now));
        // Enough peers so insufficient_peers fires: total = 0+2+20 = 22 >= 10, won't fire
        // Let's make total < 10 → use few peers
        let mut auditor2 = RoutingTableAuditor::new(AuditorConfig {
            min_total_peers: 50,
            max_empty_buckets: 0,
            stale_threshold_secs: 500,
        });
        auditor2.add_bucket(make_bucket(0, 0, now)); // empty → warning
        auditor2.add_bucket(make_bucket(1, 2, now - 600)); // stale → warning
        auditor2.add_bucket(make_bucket(2, DEFAULT_MAX_CAPACITY, now)); // full → info
                                                                        // total = 22 < 50 → error

        let findings = auditor2.audit(now);
        assert!(!findings.is_empty());

        // Verify that no finding has a severity higher than the previous one
        for window in findings.windows(2) {
            assert!(
                window[0].severity >= window[1].severity,
                "findings not sorted descending: {:?} < {:?}",
                window[0].severity,
                window[1].severity
            );
        }

        // First finding must be Error
        assert_eq!(findings[0].severity, AuditSeverity::Error);
        // Last finding must be Info
        assert_eq!(
            findings.last().map(|f| f.severity),
            Some(AuditSeverity::Info)
        );
    }

    // ── AuditSeverity ordering ────────────────────────────────────────────────

    #[test]
    fn test_audit_severity_ordering() {
        assert!(AuditSeverity::Error > AuditSeverity::Warning);
        assert!(AuditSeverity::Warning > AuditSeverity::Info);
        assert_eq!(AuditSeverity::Info, AuditSeverity::Info);
    }

    // ── fill_ratio with zero capacity ────────────────────────────────────────

    #[test]
    fn test_fill_ratio_zero_capacity() {
        let b = BucketInfo {
            bucket_index: 0,
            peer_count: 5,
            max_capacity: 0,
            last_refreshed_secs: 0,
        };
        assert!((b.fill_ratio() - 0.0).abs() < f64::EPSILON);
    }

    // ── no stale buckets when all fresh ──────────────────────────────────────

    #[test]
    fn test_no_stale_buckets_all_fresh() {
        let now = 5000u64;
        let mut auditor = RoutingTableAuditor::new(AuditorConfig {
            stale_threshold_secs: 3600,
            ..AuditorConfig::default()
        });
        auditor.add_bucket(make_bucket(0, 3, now - 100));
        auditor.add_bucket(make_bucket(1, 5, now - 200));
        assert_eq!(auditor.stale_bucket_count(now), 0);
    }
}
