//! DHT Routing Table Optimizer for Kademlia bucket health analysis.
//!
//! Periodically analyzes the Kademlia routing table to identify bucket
//! imbalances, dead peers, and suboptimal coverage, then emits remediation
//! suggestions via [`RoutingRecommendation`].
//!
//! ## Design
//!
//! The optimizer is a pure analytical component — it does **not** mutate any
//! state.  Callers feed it a snapshot of their routing table entries and
//! receive an [`OptimizationReport`] containing per-bucket analyses and a
//! prioritised list of actions to take.
//!
//! ### Health classification (per bucket)
//!
//! | Condition | Classification |
//! |---|---|
//! | `entry_count > k` (default k=20) | `Saturated` |
//! | any non-responsive entry | `Stale` |
//! | `responsive_count < target_fill` | `Sparse` |
//! | otherwise | `Healthy` |
//!
//! ### Recommendation priority
//!
//! 1. For each stale bucket: up to 3 `PingPeer` recommendations (or
//!    `EvictPeer` if the entry has not been seen for > 2× stale threshold).
//! 2. For each sparse bucket: one `RefreshBucket` recommendation.

use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────────────

/// Default number of seconds before a peer is considered stale (10 minutes).
pub const DEFAULT_STALE_THRESHOLD_SECS: u64 = 600;

/// Default target number of responsive peers per bucket.
pub const DEFAULT_TARGET_BUCKET_FILL: usize = 8;

/// Kademlia k-bucket capacity.
pub const DEFAULT_K_BUCKET_CAPACITY: usize = 20;

/// Maximum ping recommendations emitted per bucket per analysis cycle.
const MAX_PINGS_PER_BUCKET: usize = 3;

/// Multiplier applied to `stale_threshold_secs` to decide eviction vs ping.
const EVICT_MULTIPLIER: u64 = 2;

// ────────────────────────────────────────────────────────────────────────────
// BucketHealth
// ────────────────────────────────────────────────────────────────────────────

/// Health classification for a single Kademlia k-bucket.
#[derive(Debug, Clone, PartialEq)]
pub enum BucketHealth {
    /// Bucket is at or above the target fill ratio.
    Healthy,
    /// Bucket has fewer responsive peers than the target.
    Sparse {
        /// Fraction of target fill that is currently met (`responsive / target`).
        fill_ratio: f64,
    },
    /// Bucket contains one or more non-responsive (stale) peers.
    Stale {
        /// Number of non-responsive entries.
        stale_count: usize,
    },
    /// Bucket exceeds the k-bucket capacity (k = 20 by default).
    Saturated,
}

// ────────────────────────────────────────────────────────────────────────────
// RoutingEntry
// ────────────────────────────────────────────────────────────────────────────

/// A single peer entry in the Kademlia routing table snapshot.
#[derive(Debug, Clone)]
pub struct RoutingEntry {
    /// Peer identifier (e.g. libp2p PeerId as a string).
    pub peer_id: String,
    /// XOR-distance bit-prefix index (0 = closest bucket).
    pub bucket_index: u8,
    /// Unix timestamp (seconds) of the last successful response from this peer.
    pub last_seen_secs: u64,
    /// Observed round-trip latency in milliseconds.
    pub latency_ms: f64,
    /// Whether the peer responded to the most recent liveness probe.
    pub is_responsive: bool,
}

impl RoutingEntry {
    /// Convenience constructor.
    pub fn new(
        peer_id: impl Into<String>,
        bucket_index: u8,
        last_seen_secs: u64,
        latency_ms: f64,
        is_responsive: bool,
    ) -> Self {
        Self {
            peer_id: peer_id.into(),
            bucket_index,
            last_seen_secs,
            latency_ms,
            is_responsive,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// BucketAnalysis
// ────────────────────────────────────────────────────────────────────────────

/// Per-bucket analysis result produced by [`DhtRoutingOptimizer::analyze_bucket`].
#[derive(Debug, Clone)]
pub struct BucketAnalysis {
    /// Index of the analyzed bucket.
    pub bucket_index: u8,
    /// Total number of entries in this bucket.
    pub entry_count: usize,
    /// Number of entries that are currently responsive.
    pub responsive_count: usize,
    /// Average latency (ms) across responsive entries; `0.0` when none.
    pub avg_latency_ms: f64,
    /// Health classification for this bucket.
    pub health: BucketHealth,
    /// Age in seconds of the oldest entry: `now_secs - min(last_seen_secs)`.
    ///
    /// Returns `0` for an empty bucket.
    pub oldest_entry_age_secs: u64,
}

// ────────────────────────────────────────────────────────────────────────────
// RoutingRecommendation
// ────────────────────────────────────────────────────────────────────────────

/// Remediation action recommended by the optimizer.
#[derive(Debug, Clone, PartialEq)]
pub enum RoutingRecommendation {
    /// Trigger a Kademlia lookup to populate a sparse bucket.
    RefreshBucket(u8),
    /// Remove a stale / confirmed-unresponsive peer from the routing table.
    EvictPeer {
        /// Bucket containing the peer.
        bucket: u8,
        /// Peer to evict.
        peer_id: String,
    },
    /// Send a liveness ping before deciding whether to evict.
    PingPeer {
        /// Bucket containing the peer.
        bucket: u8,
        /// Peer to ping.
        peer_id: String,
    },
    /// No action required.
    NoAction,
}

impl RoutingRecommendation {
    /// Returns `true` if this recommendation requires an action to be taken
    /// (i.e. is not [`RoutingRecommendation::NoAction`]).
    pub fn is_actionable(&self) -> bool {
        !matches!(self, Self::NoAction)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// OptimizationReport
// ────────────────────────────────────────────────────────────────────────────

/// Summary report produced by [`DhtRoutingOptimizer::optimize`].
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    /// Unix timestamp (seconds) at which the analysis was performed.
    pub analyzed_at_secs: u64,
    /// Total number of routing entries across all buckets.
    pub total_entries: usize,
    /// Number of buckets classified as [`BucketHealth::Healthy`].
    pub healthy_buckets: usize,
    /// Number of buckets classified as [`BucketHealth::Sparse`].
    pub sparse_buckets: usize,
    /// Number of buckets classified as [`BucketHealth::Stale`].
    pub stale_buckets: usize,
    /// Ordered list of recommended actions.
    pub recommendations: Vec<RoutingRecommendation>,
}

impl OptimizationReport {
    /// Returns `true` if the report contains at least one actionable
    /// recommendation (anything other than [`RoutingRecommendation::NoAction`]).
    pub fn has_issues(&self) -> bool {
        self.recommendations.iter().any(|r| r.is_actionable())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// DhtRoutingOptimizer
// ────────────────────────────────────────────────────────────────────────────

/// Analyzes Kademlia routing table snapshots and emits remediation suggestions.
///
/// # Example
///
/// ```rust
/// use ipfrs_network::dht_optimizer::{DhtRoutingOptimizer, RoutingEntry};
///
/// let optimizer = DhtRoutingOptimizer::new(600, 8);
/// let entries = vec![
///     RoutingEntry::new("peer-1", 5, 1_700_000_000, 12.0, true),
///     RoutingEntry::new("peer-2", 5, 1_700_000_010, 18.5, true),
/// ];
/// let report = optimizer.optimize(&entries, 1_700_000_600);
/// assert!(!report.has_issues());
/// ```
#[derive(Debug, Clone)]
pub struct DhtRoutingOptimizer {
    /// Seconds of inactivity before a peer is considered stale.
    pub stale_threshold_secs: u64,
    /// Target number of responsive peers per bucket.
    pub target_bucket_fill: usize,
    /// Maximum number of entries allowed per bucket (Kademlia k).
    pub k_bucket_capacity: usize,
}

impl Default for DhtRoutingOptimizer {
    fn default() -> Self {
        Self::new(DEFAULT_STALE_THRESHOLD_SECS, DEFAULT_TARGET_BUCKET_FILL)
    }
}

impl DhtRoutingOptimizer {
    /// Creates a new optimizer with the given thresholds.
    ///
    /// # Arguments
    ///
    /// * `stale_threshold_secs` – seconds without a response before a peer is
    ///   labelled stale (default: [`DEFAULT_STALE_THRESHOLD_SECS`]).
    /// * `target_bucket_fill` – desired number of responsive entries per bucket
    ///   (default: [`DEFAULT_TARGET_BUCKET_FILL`]).
    pub fn new(stale_threshold_secs: u64, target_bucket_fill: usize) -> Self {
        Self {
            stale_threshold_secs,
            target_bucket_fill,
            k_bucket_capacity: DEFAULT_K_BUCKET_CAPACITY,
        }
    }

    /// Analyzes a slice of entries that all belong to the **same** bucket.
    ///
    /// The `bucket_index` field of the returned [`BucketAnalysis`] is taken
    /// from the first entry; callers should ensure all entries share the same
    /// bucket index (as guaranteed by [`optimize`][Self::optimize]).
    ///
    /// # Health classification order
    ///
    /// 1. **Saturated** — `entry_count > k_bucket_capacity`
    /// 2. **Stale** — any non-responsive entries present
    /// 3. **Sparse** — `responsive_count < target_bucket_fill`
    /// 4. **Healthy** — otherwise
    pub fn analyze_bucket(&self, entries: &[RoutingEntry], now_secs: u64) -> BucketAnalysis {
        let bucket_index = entries.first().map(|e| e.bucket_index).unwrap_or(0);
        let entry_count = entries.len();

        // Responsiveness counts
        let responsive_count = entries.iter().filter(|e| e.is_responsive).count();

        // Average latency from responsive entries only
        let avg_latency_ms = {
            let responsive_latencies: Vec<f64> = entries
                .iter()
                .filter(|e| e.is_responsive)
                .map(|e| e.latency_ms)
                .collect();
            if responsive_latencies.is_empty() {
                0.0
            } else {
                let sum: f64 = responsive_latencies.iter().sum();
                sum / responsive_latencies.len() as f64
            }
        };

        // Age of oldest entry
        let oldest_entry_age_secs = if entries.is_empty() {
            0
        } else {
            let min_last_seen = entries.iter().map(|e| e.last_seen_secs).min().unwrap_or(0);
            now_secs.saturating_sub(min_last_seen)
        };

        // Stale count: non-responsive entries
        let stale_count = entries.iter().filter(|e| !e.is_responsive).count();

        // Health classification — precedence: Saturated > Stale > Sparse > Healthy
        let health = if entry_count > self.k_bucket_capacity {
            BucketHealth::Saturated
        } else if stale_count > 0 {
            BucketHealth::Stale { stale_count }
        } else if responsive_count < self.target_bucket_fill {
            let fill_ratio = if self.target_bucket_fill == 0 {
                1.0
            } else {
                responsive_count as f64 / self.target_bucket_fill as f64
            };
            BucketHealth::Sparse { fill_ratio }
        } else {
            BucketHealth::Healthy
        };

        BucketAnalysis {
            bucket_index,
            entry_count,
            responsive_count,
            avg_latency_ms,
            health,
            oldest_entry_age_secs,
        }
    }

    /// Analyzes all routing entries and returns a full [`OptimizationReport`].
    ///
    /// # Algorithm
    ///
    /// 1. Group entries by `bucket_index`.
    /// 2. Run [`analyze_bucket`][Self::analyze_bucket] on each group.
    /// 3. For each bucket produce recommendations:
    ///    - **Stale** entries: emit `PingPeer` (or `EvictPeer` if age >
    ///      2× `stale_threshold_secs`), capped at
    ///      `MAX_PINGS_PER_BUCKET` per bucket.
    ///    - **Sparse** bucket: emit `RefreshBucket`.
    /// 4. Collate bucket-health counts and return the report.
    pub fn optimize(&self, all_entries: &[RoutingEntry], now_secs: u64) -> OptimizationReport {
        if all_entries.is_empty() {
            return OptimizationReport {
                analyzed_at_secs: now_secs,
                total_entries: 0,
                healthy_buckets: 0,
                sparse_buckets: 0,
                stale_buckets: 0,
                recommendations: vec![],
            };
        }

        // Group entries by bucket index
        let mut buckets: HashMap<u8, Vec<&RoutingEntry>> = HashMap::new();
        for entry in all_entries {
            buckets.entry(entry.bucket_index).or_default().push(entry);
        }

        let mut recommendations: Vec<RoutingRecommendation> = Vec::new();
        let mut healthy_buckets = 0usize;
        let mut sparse_buckets = 0usize;
        let mut stale_buckets = 0usize;

        // Sort bucket indices for deterministic output
        let mut bucket_indices: Vec<u8> = buckets.keys().copied().collect();
        bucket_indices.sort_unstable();

        for idx in bucket_indices {
            let entries: Vec<RoutingEntry> = buckets[&idx].iter().map(|e| (*e).clone()).collect();

            let analysis = self.analyze_bucket(&entries, now_secs);

            match &analysis.health {
                BucketHealth::Healthy | BucketHealth::Saturated => {
                    healthy_buckets += 1;
                }
                BucketHealth::Sparse { .. } => {
                    sparse_buckets += 1;
                    recommendations.push(RoutingRecommendation::RefreshBucket(idx));
                }
                BucketHealth::Stale { .. } => {
                    stale_buckets += 1;

                    // Emit up to MAX_PINGS_PER_BUCKET recommendations for
                    // non-responsive entries in this bucket.
                    let evict_threshold = self.stale_threshold_secs * EVICT_MULTIPLIER;
                    let mut ping_count = 0usize;

                    for entry in entries
                        .iter()
                        .filter(|e| !e.is_responsive)
                        .take(MAX_PINGS_PER_BUCKET)
                    {
                        let age = now_secs.saturating_sub(entry.last_seen_secs);
                        if age > evict_threshold {
                            recommendations.push(RoutingRecommendation::EvictPeer {
                                bucket: idx,
                                peer_id: entry.peer_id.clone(),
                            });
                        } else {
                            recommendations.push(RoutingRecommendation::PingPeer {
                                bucket: idx,
                                peer_id: entry.peer_id.clone(),
                            });
                            ping_count += 1;
                            if ping_count >= MAX_PINGS_PER_BUCKET {
                                break;
                            }
                        }
                    }
                }
            }
        }

        OptimizationReport {
            analyzed_at_secs: now_secs,
            total_entries: all_entries.len(),
            healthy_buckets,
            sparse_buckets,
            stale_buckets,
            recommendations,
        }
    }

    /// Returns the `n` responsive peers with the **highest** latency, sorted
    /// descending.  These are candidates for replacement by lower-latency
    /// peers discovered during bucket refresh.
    pub fn top_latency_peers<'a>(
        &self,
        entries: &'a [RoutingEntry],
        n: usize,
    ) -> Vec<&'a RoutingEntry> {
        let mut responsive: Vec<&'a RoutingEntry> =
            entries.iter().filter(|e| e.is_responsive).collect();

        // Sort descending by latency (NaN-safe: treat NaN as very large)
        responsive.sort_by(|a, b| {
            b.latency_ms
                .partial_cmp(&a.latency_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        responsive.into_iter().take(n).collect()
    }

    /// Returns the fraction of the 256 possible Kademlia bucket indices that
    /// have at least one responsive entry.
    ///
    /// Returns `0.0` for an empty entry list.
    pub fn coverage_score(&self, entries: &[RoutingEntry]) -> f64 {
        if entries.is_empty() {
            return 0.0;
        }

        let covered: std::collections::HashSet<u8> = entries
            .iter()
            .filter(|e| e.is_responsive)
            .map(|e| e.bucket_index)
            .collect();

        covered.len() as f64 / 256.0
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_700_000_000;

    fn make_entry(
        peer_id: &str,
        bucket: u8,
        last_seen_offset: i64,
        latency_ms: f64,
        responsive: bool,
    ) -> RoutingEntry {
        let last_seen = (NOW as i64 + last_seen_offset) as u64;
        RoutingEntry::new(peer_id, bucket, last_seen, latency_ms, responsive)
    }

    // ── Constructor ──────────────────────────────────────────────────────────

    #[test]
    fn test_new_with_custom_params() {
        let opt = DhtRoutingOptimizer::new(300, 5);
        assert_eq!(opt.stale_threshold_secs, 300);
        assert_eq!(opt.target_bucket_fill, 5);
        assert_eq!(opt.k_bucket_capacity, DEFAULT_K_BUCKET_CAPACITY);
    }

    // ── analyze_bucket ───────────────────────────────────────────────────────

    #[test]
    fn test_analyze_bucket_all_healthy() {
        let opt = DhtRoutingOptimizer::new(600, 3);
        let entries: Vec<RoutingEntry> = (0..5)
            .map(|i| make_entry(&format!("peer-{i}"), 7, 0, 10.0, true))
            .collect();
        let analysis = opt.analyze_bucket(&entries, NOW);
        assert_eq!(analysis.entry_count, 5);
        assert_eq!(analysis.responsive_count, 5);
        assert_eq!(analysis.health, BucketHealth::Healthy);
        assert!((analysis.avg_latency_ms - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_analyze_bucket_sparse() {
        let opt = DhtRoutingOptimizer::new(600, 8);
        // Only 2 responsive out of target 8 → Sparse
        let entries: Vec<RoutingEntry> = (0..2)
            .map(|i| make_entry(&format!("peer-{i}"), 3, 0, 5.0, true))
            .collect();
        let analysis = opt.analyze_bucket(&entries, NOW);
        assert_eq!(analysis.responsive_count, 2);
        if let BucketHealth::Sparse { fill_ratio } = analysis.health {
            assert!((fill_ratio - 0.25).abs() < 1e-9); // 2/8
        } else {
            panic!("expected Sparse, got {:?}", analysis.health);
        }
    }

    #[test]
    fn test_analyze_bucket_stale() {
        let opt = DhtRoutingOptimizer::new(600, 2);
        let entries = vec![
            make_entry("peer-a", 1, 0, 10.0, true),
            make_entry("peer-b", 1, 0, 10.0, false), // non-responsive
        ];
        let analysis = opt.analyze_bucket(&entries, NOW);
        assert!(matches!(
            analysis.health,
            BucketHealth::Stale { stale_count: 1 }
        ));
    }

    #[test]
    fn test_analyze_bucket_saturated() {
        let opt = DhtRoutingOptimizer::new(600, 8);
        // 21 entries > k=20 → Saturated
        let entries: Vec<RoutingEntry> = (0..21)
            .map(|i| make_entry(&format!("peer-{i}"), 9, 0, 5.0, true))
            .collect();
        let analysis = opt.analyze_bucket(&entries, NOW);
        assert_eq!(analysis.health, BucketHealth::Saturated);
    }

    #[test]
    fn test_analyze_bucket_empty() {
        let opt = DhtRoutingOptimizer::default();
        let analysis = opt.analyze_bucket(&[], NOW);
        assert_eq!(analysis.entry_count, 0);
        assert_eq!(analysis.responsive_count, 0);
        assert_eq!(analysis.oldest_entry_age_secs, 0);
        assert_eq!(analysis.avg_latency_ms, 0.0);
        // Empty → responsive_count (0) < target_fill (8) → Sparse
        assert!(matches!(analysis.health, BucketHealth::Sparse { .. }));
    }

    #[test]
    fn test_analyze_bucket_oldest_entry_age() {
        let opt = DhtRoutingOptimizer::default();
        let entries = vec![
            make_entry("peer-x", 2, -500, 10.0, true), // 500 s ago
            make_entry("peer-y", 2, -100, 10.0, true), // 100 s ago
        ];
        let analysis = opt.analyze_bucket(&entries, NOW);
        assert_eq!(analysis.oldest_entry_age_secs, 500);
    }

    // ── optimize ─────────────────────────────────────────────────────────────

    #[test]
    fn test_optimize_empty_returns_empty_report() {
        let opt = DhtRoutingOptimizer::default();
        let report = opt.optimize(&[], NOW);
        assert_eq!(report.total_entries, 0);
        assert_eq!(report.recommendations.len(), 0);
        assert!(!report.has_issues());
    }

    #[test]
    fn test_optimize_single_healthy_bucket_no_recommendations() {
        let opt = DhtRoutingOptimizer::new(600, 3);
        let entries: Vec<RoutingEntry> = (0..5)
            .map(|i| make_entry(&format!("peer-{i}"), 4, 0, 8.0, true))
            .collect();
        let report = opt.optimize(&entries, NOW);
        assert!(!report.has_issues());
        assert_eq!(report.healthy_buckets, 1);
    }

    #[test]
    fn test_optimize_sparse_bucket_refresh_recommendation() {
        let opt = DhtRoutingOptimizer::new(600, 8);
        // Only 2 responsive peers in bucket 10 → Sparse → RefreshBucket(10)
        let entries: Vec<RoutingEntry> = (0..2)
            .map(|i| make_entry(&format!("peer-{i}"), 10, 0, 5.0, true))
            .collect();
        let report = opt.optimize(&entries, NOW);
        assert!(report.has_issues());
        assert!(report
            .recommendations
            .contains(&RoutingRecommendation::RefreshBucket(10)));
        assert_eq!(report.sparse_buckets, 1);
    }

    #[test]
    fn test_optimize_stale_peer_ping_recommendation() {
        let opt = DhtRoutingOptimizer::new(600, 1);
        // One responsive + one non-responsive (age = 400 s < 2×600) → PingPeer
        let entries = vec![
            make_entry("peer-ok", 6, 0, 5.0, true),
            make_entry("peer-bad", 6, -400, 5.0, false),
        ];
        let report = opt.optimize(&entries, NOW);
        let has_ping = report.recommendations.iter().any(|r| {
            matches!(r, RoutingRecommendation::PingPeer { peer_id, .. } if peer_id == "peer-bad")
        });
        assert!(has_ping, "expected PingPeer for peer-bad");
    }

    #[test]
    fn test_optimize_very_old_peer_evict_recommendation() {
        let opt = DhtRoutingOptimizer::new(600, 1);
        // Non-responsive peer not seen for 2000 s > 2×600 → EvictPeer
        let entries = vec![
            make_entry("peer-ok", 6, 0, 5.0, true),
            make_entry("peer-old", 6, -2000, 5.0, false),
        ];
        let report = opt.optimize(&entries, NOW);
        let has_evict = report.recommendations.iter().any(|r| {
            matches!(r, RoutingRecommendation::EvictPeer { peer_id, .. } if peer_id == "peer-old")
        });
        assert!(has_evict, "expected EvictPeer for peer-old");
    }

    #[test]
    fn test_optimize_counts_bucket_health_correctly() {
        let opt = DhtRoutingOptimizer::new(600, 8);
        // Bucket 0: 10 responsive → Healthy
        let healthy: Vec<RoutingEntry> = (0..10)
            .map(|i| make_entry(&format!("h-{i}"), 0, 0, 5.0, true))
            .collect();
        // Bucket 1: 2 responsive → Sparse
        let sparse: Vec<RoutingEntry> = (0..2)
            .map(|i| make_entry(&format!("s-{i}"), 1, 0, 5.0, true))
            .collect();
        // Bucket 2: 1 non-responsive + 5 responsive → Stale
        let mut stale: Vec<RoutingEntry> = (0..5)
            .map(|i| make_entry(&format!("st-{i}"), 2, 0, 5.0, true))
            .collect();
        stale.push(make_entry("stale-peer", 2, -100, 5.0, false));

        let all: Vec<RoutingEntry> = [healthy, sparse, stale].concat();
        let report = opt.optimize(&all, NOW);
        assert_eq!(report.healthy_buckets, 1);
        assert_eq!(report.sparse_buckets, 1);
        assert_eq!(report.stale_buckets, 1);
    }

    #[test]
    fn test_optimize_at_most_3_ping_recommendations_per_bucket() {
        let opt = DhtRoutingOptimizer::new(600, 1);
        // 6 non-responsive peers in same bucket, age within threshold
        let mut entries: Vec<RoutingEntry> = (0..6)
            .map(|i| make_entry(&format!("bad-{i}"), 5, -100, 5.0, false))
            .collect();
        // Add one responsive so the bucket is classified as Stale
        entries.push(make_entry("ok", 5, 0, 5.0, true));

        let report = opt.optimize(&entries, NOW);
        let ping_count = report
            .recommendations
            .iter()
            .filter(|r| matches!(r, RoutingRecommendation::PingPeer { .. }))
            .count();
        assert!(
            ping_count <= MAX_PINGS_PER_BUCKET,
            "expected ≤ {MAX_PINGS_PER_BUCKET} pings, got {ping_count}"
        );
    }

    // ── top_latency_peers ────────────────────────────────────────────────────

    #[test]
    fn test_top_latency_peers_sorted_descending() {
        let opt = DhtRoutingOptimizer::default();
        let entries = vec![
            make_entry("peer-a", 0, 0, 10.0, true),
            make_entry("peer-b", 0, 0, 50.0, true),
            make_entry("peer-c", 0, 0, 30.0, true),
            make_entry("peer-d", 0, 0, 5.0, false), // non-responsive, excluded
        ];
        let top = opt.top_latency_peers(&entries, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].peer_id, "peer-b");
        assert_eq!(top[1].peer_id, "peer-c");
    }

    // ── coverage_score ───────────────────────────────────────────────────────

    #[test]
    fn test_coverage_score_zero_for_empty() {
        let opt = DhtRoutingOptimizer::default();
        assert_eq!(opt.coverage_score(&[]), 0.0);
    }

    #[test]
    fn test_coverage_score_fractional_for_partial_coverage() {
        let opt = DhtRoutingOptimizer::default();
        // 2 distinct responsive bucket indices out of 256
        let entries = vec![
            make_entry("peer-1", 0, 0, 5.0, true),
            make_entry("peer-2", 1, 0, 5.0, true),
            make_entry("peer-3", 1, 0, 5.0, true), // duplicate bucket
            make_entry("peer-4", 2, 0, 5.0, false), // non-responsive, not counted
        ];
        let score = opt.coverage_score(&entries);
        // 2 responsive unique buckets / 256
        assert!((score - 2.0 / 256.0).abs() < 1e-12);
    }

    // ── has_issues ───────────────────────────────────────────────────────────

    #[test]
    fn test_has_issues_true_when_recommendations_exist() {
        let opt = DhtRoutingOptimizer::new(600, 8);
        // Only 1 responsive peer → Sparse → RefreshBucket
        let entries = vec![make_entry("peer-1", 0, 0, 5.0, true)];
        let report = opt.optimize(&entries, NOW);
        assert!(report.has_issues());
    }
}
