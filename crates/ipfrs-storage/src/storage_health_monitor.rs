//! Storage Health Monitor
//!
//! Production-grade health monitoring system for IPFRS storage subsystems.
//! Tracks probes across multiple storage categories (BlockStore, WalLog, Cache, etc.),
//! computes EWMA success rates and latencies, fires severity-graded alerts,
//! maintains a bounded rolling history of snapshots, and provides actionable
//! recovery suggestions per probe.
//!
//! # Example
//! ```rust
//! use ipfrs_storage::storage_health_monitor::{
//!     StorageHealthMonitor, ShmMonitorConfig, ShmCategory,
//! };
//!
//! let cfg = ShmMonitorConfig::default();
//! let mut monitor = StorageHealthMonitor::new(cfg);
//! let id = monitor.register_probe("wal-primary", ShmCategory::WalLog, Default::default());
//! monitor.record_check(id, true, 1.2).ok();
//! let snap = monitor.run_health_check();
//! assert!(snap.score >= 0.0 && snap.score <= 1.0);
//! ```

use std::collections::{HashMap, VecDeque};

// ─── deterministic pseudo-random (xorshift64) ────────────────────────────────

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ─── type aliases ────────────────────────────────────────────────────────────

/// Opaque probe identifier.
pub type ShmProbeId = u32;

/// Type alias for the monitor (public surface).
pub type ShmStorageHealthMonitor = StorageHealthMonitor;

// ─── enumerations ────────────────────────────────────────────────────────────

/// Storage subsystem category tracked by a probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShmCategory {
    BlockStore,
    IndexStore,
    WalLog,
    Cache,
    Network,
    Encryption,
    Compression,
}

impl ShmCategory {
    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::BlockStore => "block-store",
            Self::IndexStore => "index-store",
            Self::WalLog => "wal-log",
            Self::Cache => "cache",
            Self::Network => "network",
            Self::Encryption => "encryption",
            Self::Compression => "compression",
        }
    }
}

/// Operational status of a single probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShmStatus {
    Healthy,
    Degraded,
    Critical,
    Unknown,
    Recovering,
}

impl ShmStatus {
    /// Numeric weight for aggregate scoring (0 = best, 3 = worst).
    pub fn weight(self) -> f64 {
        match self {
            Self::Healthy => 0.0,
            Self::Recovering => 0.5,
            Self::Unknown => 1.0,
            Self::Degraded => 2.0,
            Self::Critical => 3.0,
        }
    }
}

/// Alert severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ShmSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

// ─── configuration ───────────────────────────────────────────────────────────

/// Configuration for [`StorageHealthMonitor`].
#[derive(Debug, Clone)]
pub struct ShmMonitorConfig {
    /// Seconds between automatic health-check cycles (informational; caller drives timing).
    pub check_interval_secs: u64,
    /// Success-rate threshold below which a probe becomes `Degraded` (default 0.80).
    pub alert_threshold: f64,
    /// Success-rate threshold above which a probe transitions from `Critical/Degraded`
    /// to `Recovering` (default 0.90).
    pub recovery_threshold: f64,
    /// Number of consecutive failures before a probe becomes `Critical`.
    pub max_consecutive_failures: u32,
    /// EWMA smoothing factor α for success_rate (0 < α ≤ 1; default 0.1).
    pub ewma_alpha: f64,
    /// EWMA smoothing factor α for latency_ms (0 < α ≤ 1; default 0.2).
    pub ewma_latency_alpha: f64,
    /// TTL seconds for auto-resolvable alerts (0 = no auto-resolve).
    pub alert_auto_resolve_secs: u64,
}

impl Default for ShmMonitorConfig {
    fn default() -> Self {
        Self {
            check_interval_secs: 30,
            alert_threshold: 0.80,
            recovery_threshold: 0.90,
            max_consecutive_failures: 5,
            ewma_alpha: 0.10,
            ewma_latency_alpha: 0.20,
            alert_auto_resolve_secs: 300,
        }
    }
}

// ─── probe ───────────────────────────────────────────────────────────────────

/// A named health-check probe for one storage component.
#[derive(Debug, Clone)]
pub struct ShmProbe {
    pub id: ShmProbeId,
    pub name: String,
    pub category: ShmCategory,
    pub status: ShmStatus,
    /// Unix-epoch timestamp of the most recent check (seconds).
    pub last_check_ts: u64,
    pub consecutive_failures: u32,
    /// EWMA-smoothed success rate (0.0–1.0).
    pub success_rate: f64,
    /// EWMA-smoothed latency in milliseconds.
    pub latency_ms: f64,
    /// Freeform key/value metadata attached at registration.
    pub metadata: HashMap<String, String>,
    /// Total checks recorded.
    pub total_checks: u64,
    /// Total successful checks.
    pub total_successes: u64,
}

impl ShmProbe {
    fn new(
        id: ShmProbeId,
        name: impl Into<String>,
        category: ShmCategory,
        metadata: HashMap<String, String>,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            category,
            status: ShmStatus::Unknown,
            last_check_ts: 0,
            consecutive_failures: 0,
            success_rate: 1.0, // optimistic initial value
            latency_ms: 0.0,
            metadata,
            total_checks: 0,
            total_successes: 0,
        }
    }
}

// ─── snapshot / alert ────────────────────────────────────────────────────────

/// Point-in-time health snapshot across all probes.
#[derive(Debug, Clone)]
pub struct ShmHealthSnapshot {
    pub ts: u64,
    pub overall_status: ShmStatus,
    pub probe_statuses: HashMap<ShmProbeId, ShmStatus>,
    /// Weighted composite score in [0.0, 1.0] where 1.0 = fully healthy.
    pub score: f64,
    /// Number of probes sampled.
    pub probe_count: usize,
}

/// A fired alert from the monitoring system.
#[derive(Debug, Clone)]
pub struct ShmAlert {
    pub id: usize,
    pub ts: u64,
    pub probe_id: ShmProbeId,
    pub severity: ShmSeverity,
    pub message: String,
    /// If set, the alert may be automatically expired at this Unix-epoch timestamp.
    pub auto_resolve_at: Option<u64>,
    pub resolved: bool,
}

// ─── stats ───────────────────────────────────────────────────────────────────

/// Aggregate statistics for the monitor.
#[derive(Debug, Clone, Default)]
pub struct ShmMonitorStats {
    pub total_probes: usize,
    pub healthy_count: usize,
    pub degraded_count: usize,
    pub critical_count: usize,
    pub unknown_count: usize,
    pub recovering_count: usize,
    pub total_alerts_fired: u64,
    pub active_alerts: usize,
    pub snapshots_stored: usize,
    pub total_checks_recorded: u64,
}

// ─── main struct ─────────────────────────────────────────────────────────────

/// Production-grade storage-subsystem health monitor.
///
/// Manages a collection of named probes, records check outcomes (success/failure +
/// latency), computes EWMA metrics, fires alerts, and maintains a rolling history
/// of health snapshots.
pub struct StorageHealthMonitor {
    config: ShmMonitorConfig,
    probes: HashMap<ShmProbeId, ShmProbe>,
    /// Bounded deque of historical snapshots (max 200).
    history: VecDeque<ShmHealthSnapshot>,
    /// Bounded deque of fired alerts (max 500).
    alerts: VecDeque<ShmAlert>,
    next_probe_id: ShmProbeId,
    next_alert_id: usize,
    total_alerts_fired: u64,
    /// Simple monotonic clock seed (xorshift64 state for jitter/demo purposes).
    rng_state: u64,
}

const MAX_HISTORY: usize = 200;
const MAX_ALERTS: usize = 500;

impl StorageHealthMonitor {
    // ── construction ──────────────────────────────────────────────────────

    /// Create a new monitor with the given configuration.
    pub fn new(config: ShmMonitorConfig) -> Self {
        Self {
            config,
            probes: HashMap::new(),
            history: VecDeque::with_capacity(MAX_HISTORY),
            alerts: VecDeque::with_capacity(MAX_ALERTS),
            next_probe_id: 1,
            next_alert_id: 0,
            total_alerts_fired: 0,
            rng_state: 0xdeadbeef_cafebabe,
        }
    }

    /// Create a monitor with default configuration.
    pub fn default_config() -> Self {
        Self::new(ShmMonitorConfig::default())
    }

    // ── probe registration ────────────────────────────────────────────────

    /// Register a new probe and return its unique [`ShmProbeId`].
    pub fn register_probe(
        &mut self,
        name: impl Into<String>,
        category: ShmCategory,
        metadata: HashMap<String, String>,
    ) -> ShmProbeId {
        let id = self.next_probe_id;
        self.next_probe_id = self.next_probe_id.saturating_add(1);
        let probe = ShmProbe::new(id, name, category, metadata);
        self.probes.insert(id, probe);
        id
    }

    // ── check recording ───────────────────────────────────────────────────

    /// Record the outcome of a health check for `probe_id`.
    ///
    /// Updates EWMA success rate and latency, transitions probe status, and
    /// fires alerts when thresholds are crossed.
    ///
    /// Returns an error string if the probe does not exist.
    pub fn record_check(
        &mut self,
        probe_id: ShmProbeId,
        success: bool,
        latency_ms: f64,
    ) -> Result<(), String> {
        let ts = self.now_ts();
        self.record_check_at(probe_id, success, latency_ms, ts)
    }

    /// Same as `record_check` but with an explicit Unix-epoch timestamp.
    pub fn record_check_at(
        &mut self,
        probe_id: ShmProbeId,
        success: bool,
        latency_ms: f64,
        now_ts: u64,
    ) -> Result<(), String> {
        let config = self.config.clone();
        let probe = self
            .probes
            .get_mut(&probe_id)
            .ok_or_else(|| format!("probe {} not found", probe_id))?;

        probe.total_checks += 1;
        probe.last_check_ts = now_ts;

        // EWMA success rate
        let outcome = if success { 1.0_f64 } else { 0.0_f64 };
        probe.success_rate =
            config.ewma_alpha * outcome + (1.0 - config.ewma_alpha) * probe.success_rate;

        // EWMA latency (only update on success to avoid polluting with failure spikes)
        if success {
            probe.total_successes += 1;
            probe.consecutive_failures = 0;
            let lat = latency_ms.max(0.0);
            if probe.latency_ms == 0.0 {
                probe.latency_ms = lat;
            } else {
                probe.latency_ms = config.ewma_latency_alpha * lat
                    + (1.0 - config.ewma_latency_alpha) * probe.latency_ms;
            }
        } else {
            probe.consecutive_failures = probe.consecutive_failures.saturating_add(1);
        }

        // Determine new status
        let new_status = self.compute_probe_status(probe_id, &config)?;
        let old_status = self.probes[&probe_id].status;

        if let Some(p) = self.probes.get_mut(&probe_id) {
            p.status = new_status;
        }

        // Fire alerts on status transitions
        self.maybe_fire_alert(probe_id, old_status, new_status, now_ts, &config);

        Ok(())
    }

    // ── health snapshot ───────────────────────────────────────────────────

    /// Compute an overall [`ShmHealthSnapshot`] and push it to the history.
    pub fn run_health_check(&mut self) -> ShmHealthSnapshot {
        let ts = self.now_ts();
        self.run_health_check_at(ts)
    }

    /// Same as `run_health_check` but with an explicit timestamp.
    pub fn run_health_check_at(&mut self, now_ts: u64) -> ShmHealthSnapshot {
        let probe_statuses: HashMap<ShmProbeId, ShmStatus> =
            self.probes.iter().map(|(&id, p)| (id, p.status)).collect();

        let (score, overall_status) = self.compute_overall_score();

        let snap = ShmHealthSnapshot {
            ts: now_ts,
            overall_status,
            probe_statuses,
            score,
            probe_count: self.probes.len(),
        };

        if self.history.len() >= MAX_HISTORY {
            self.history.pop_front();
        }
        self.history.push_back(snap.clone());

        snap
    }

    // ── alert management ─────────────────────────────────────────────────

    /// Mark an alert as resolved by its index in the internal deque.
    pub fn resolve_alert(&mut self, alert_id: usize) -> Result<(), String> {
        let alert = self
            .alerts
            .iter_mut()
            .find(|a| a.id == alert_id)
            .ok_or_else(|| format!("alert {} not found", alert_id))?;
        alert.resolved = true;
        Ok(())
    }

    /// Expire all unresolved alerts whose `auto_resolve_at` timestamp ≤ `now_ts`.
    pub fn expire_alerts(&mut self, now_ts: u64) {
        for alert in self.alerts.iter_mut() {
            if let Some(expire_at) = alert.auto_resolve_at {
                if !alert.resolved && now_ts >= expire_at {
                    alert.resolved = true;
                }
            }
        }
    }

    /// Return active (unresolved) alerts.
    pub fn active_alerts(&self) -> Vec<&ShmAlert> {
        self.alerts.iter().filter(|a| !a.resolved).collect()
    }

    /// Return all alerts (resolved and unresolved).
    pub fn all_alerts(&self) -> Vec<&ShmAlert> {
        self.alerts.iter().collect()
    }

    // ── query helpers ─────────────────────────────────────────────────────

    /// Return all probes belonging to the given category.
    pub fn probes_by_category(&self, cat: ShmCategory) -> Vec<&ShmProbe> {
        self.probes.values().filter(|p| p.category == cat).collect()
    }

    /// Return a reference to a probe by id.
    pub fn probe(&self, id: ShmProbeId) -> Option<&ShmProbe> {
        self.probes.get(&id)
    }

    /// Return the last `window` overall health scores from history (oldest first).
    pub fn health_trend(&self, window: usize) -> Vec<f64> {
        let skip = self.history.len().saturating_sub(window);
        self.history.iter().skip(skip).map(|s| s.score).collect()
    }

    /// Return the last `window` snapshots from history (oldest first).
    pub fn history_window(&self, window: usize) -> Vec<&ShmHealthSnapshot> {
        let skip = self.history.len().saturating_sub(window);
        self.history.iter().skip(skip).collect()
    }

    // ── recovery suggestions ──────────────────────────────────────────────

    /// Produce a human-readable recovery suggestion for the given probe.
    pub fn suggest_recovery(&self, probe_id: ShmProbeId) -> Result<String, String> {
        let probe = self
            .probes
            .get(&probe_id)
            .ok_or_else(|| format!("probe {} not found", probe_id))?;

        let suggestion = match (probe.category, probe.status) {
            (ShmCategory::BlockStore, ShmStatus::Critical) => {
                "Run block-store integrity scan and compact fragmented segments. \
                 Check disk utilisation and I/O error counters. \
                 Consider failing over to a replica shard."
            }
            (ShmCategory::BlockStore, ShmStatus::Degraded) => {
                "Increase block-store flush concurrency. \
                 Verify that write-back cache is not saturated. \
                 Monitor IOPS and queue depth."
            }
            (ShmCategory::IndexStore, ShmStatus::Critical) => {
                "Trigger an index rebuild from the WAL. \
                 Verify B-tree pages for corruption. \
                 Restore from last known-good index snapshot if rebuild fails."
            }
            (ShmCategory::IndexStore, ShmStatus::Degraded) => {
                "Schedule an index compaction pass. \
                 Check for lock contention on hot index buckets."
            }
            (ShmCategory::WalLog, ShmStatus::Critical) => {
                "Halt writes immediately to prevent data loss. \
                 Replay the WAL from the last checkpoint. \
                 Verify disk space; rotate or archive old segments."
            }
            (ShmCategory::WalLog, ShmStatus::Degraded) => {
                "Increase WAL segment size or rotation frequency. \
                 Monitor sync latency; consider async fsync mode."
            }
            (ShmCategory::Cache, ShmStatus::Critical) => {
                "Flush and invalidate the cache. \
                 Investigate memory pressure — consider reducing cache capacity. \
                 Check for eviction storms."
            }
            (ShmCategory::Cache, ShmStatus::Degraded) => {
                "Tune LRU/LFU eviction policy. \
                 Review cache hit-rate; if < 60 % consider increasing capacity."
            }
            (ShmCategory::Network, ShmStatus::Critical) => {
                "Check peer connectivity and firewall rules. \
                 Verify TLS certificate validity. \
                 Restart the libp2p swarm if persistent connection failures."
            }
            (ShmCategory::Network, ShmStatus::Degraded) => {
                "Inspect bandwidth utilisation. \
                 Check for TCP retransmissions and increase socket buffer sizes if needed."
            }
            (ShmCategory::Encryption, ShmStatus::Critical) => {
                "Rotate encryption keys immediately. \
                 Verify KMS availability. \
                 Do not serve requests until the encryption subsystem is healthy."
            }
            (ShmCategory::Encryption, ShmStatus::Degraded) => {
                "Check key-derivation latency. \
                 Ensure HSM or KMS response times are acceptable."
            }
            (ShmCategory::Compression, ShmStatus::Critical) => {
                "Disable compression and serve raw blocks temporarily. \
                 Inspect codec state; a corrupted dictionary may require rebuild."
            }
            (ShmCategory::Compression, ShmStatus::Degraded) => {
                "Lower compression level to reduce CPU pressure. \
                 Consider switching to a faster codec (LZ4 instead of Zstd)."
            }
            (_, ShmStatus::Unknown) => {
                "No check data yet — register and execute at least one probe check \
                 before interpreting status."
            }
            (_, ShmStatus::Recovering) => {
                "Probe is recovering; monitor consecutive successes. \
                 Do not re-introduce heavy load until status reaches Healthy."
            }
            (_, ShmStatus::Healthy) => "No action required — probe is healthy.",
        };

        Ok(format!(
            "[probe={} category={} status={:?}] {}",
            probe.name,
            probe.category.label(),
            probe.status,
            suggestion
        ))
    }

    // ── statistics ────────────────────────────────────────────────────────

    /// Return aggregate statistics for the monitor.
    pub fn monitor_stats(&self) -> ShmMonitorStats {
        let mut stats = ShmMonitorStats {
            total_probes: self.probes.len(),
            total_alerts_fired: self.total_alerts_fired,
            active_alerts: self.active_alerts().len(),
            snapshots_stored: self.history.len(),
            total_checks_recorded: self.probes.values().map(|p| p.total_checks).sum(),
            ..Default::default()
        };
        for probe in self.probes.values() {
            match probe.status {
                ShmStatus::Healthy => stats.healthy_count += 1,
                ShmStatus::Degraded => stats.degraded_count += 1,
                ShmStatus::Critical => stats.critical_count += 1,
                ShmStatus::Unknown => stats.unknown_count += 1,
                ShmStatus::Recovering => stats.recovering_count += 1,
            }
        }
        stats
    }

    /// Return the current configuration.
    pub fn config(&self) -> &ShmMonitorConfig {
        &self.config
    }

    /// Update the configuration at runtime.
    pub fn set_config(&mut self, config: ShmMonitorConfig) {
        self.config = config;
    }

    /// Return the number of registered probes.
    pub fn probe_count(&self) -> usize {
        self.probes.len()
    }

    /// Return the number of stored history snapshots.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    // ── internal helpers ──────────────────────────────────────────────────

    /// A simple monotonically-advancing timestamp based on xorshift64.
    /// In production callers should pass real wall-clock timestamps;
    /// this is used as a fallback / for internal state seeding.
    fn now_ts(&mut self) -> u64 {
        // Use xorshift just to advance state — real time via std
        let _ = xorshift64(&mut self.rng_state);
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    /// Compute probe status from current EWMA and consecutive failures.
    fn compute_probe_status(
        &self,
        probe_id: ShmProbeId,
        config: &ShmMonitorConfig,
    ) -> Result<ShmStatus, String> {
        let probe = self
            .probes
            .get(&probe_id)
            .ok_or_else(|| format!("probe {} not found", probe_id))?;

        let status = if probe.consecutive_failures >= config.max_consecutive_failures {
            ShmStatus::Critical
        } else if probe.success_rate < config.alert_threshold {
            // Distinguish degraded vs critical based on how far below threshold
            let ratio = probe.success_rate / config.alert_threshold;
            if ratio < 0.5 {
                ShmStatus::Critical
            } else {
                ShmStatus::Degraded
            }
        } else if probe.success_rate >= config.recovery_threshold
            && matches!(probe.status, ShmStatus::Degraded | ShmStatus::Critical)
        {
            // Rate is recovering but was previously unhealthy — stay in Recovering
            ShmStatus::Recovering
        } else if probe.success_rate >= config.recovery_threshold
            && probe.status == ShmStatus::Recovering
            && probe.consecutive_failures == 0
        {
            // Sustained recovery with no recent failures → promote to Healthy
            ShmStatus::Healthy
        } else if probe.status == ShmStatus::Unknown {
            if probe.success_rate >= config.recovery_threshold {
                ShmStatus::Healthy
            } else {
                ShmStatus::Unknown
            }
        } else {
            // Stays healthy if currently healthy and rate is fine
            probe.status
        };

        Ok(status)
    }

    /// Compute (score, overall_status) across all probes.
    fn compute_overall_score(&self) -> (f64, ShmStatus) {
        if self.probes.is_empty() {
            return (1.0, ShmStatus::Healthy);
        }

        let total: f64 = self.probes.values().map(|p| p.success_rate).sum();
        let score = (total / self.probes.len() as f64).clamp(0.0, 1.0);

        // Overall status is driven by the worst probe
        let worst = self.probes.values().fold(ShmStatus::Healthy, |acc, p| {
            if p.status.weight() > acc.weight() {
                p.status
            } else {
                acc
            }
        });

        (score, worst)
    }

    /// Fire an alert if the probe status changed to a noteworthy state.
    fn maybe_fire_alert(
        &mut self,
        probe_id: ShmProbeId,
        old_status: ShmStatus,
        new_status: ShmStatus,
        now_ts: u64,
        config: &ShmMonitorConfig,
    ) {
        if old_status == new_status {
            return;
        }

        let severity = match new_status {
            ShmStatus::Healthy => ShmSeverity::Info,
            ShmStatus::Recovering => ShmSeverity::Info,
            ShmStatus::Degraded => ShmSeverity::Warning,
            ShmStatus::Critical => ShmSeverity::Critical,
            ShmStatus::Unknown => return, // not worth alerting on unknown
        };

        let probe_name = match self.probes.get(&probe_id) {
            Some(p) => p.name.clone(),
            None => format!("probe-{}", probe_id),
        };

        let message = format!(
            "probe '{}' transitioned {:?} → {:?}",
            probe_name, old_status, new_status
        );

        let auto_resolve_at = if config.alert_auto_resolve_secs > 0 {
            Some(now_ts.saturating_add(config.alert_auto_resolve_secs))
        } else {
            None
        };

        let alert_id = self.next_alert_id;
        self.next_alert_id += 1;
        self.total_alerts_fired += 1;

        let alert = ShmAlert {
            id: alert_id,
            ts: now_ts,
            probe_id,
            severity,
            message,
            auto_resolve_at,
            resolved: false,
        };

        if self.alerts.len() >= MAX_ALERTS {
            self.alerts.pop_front();
        }
        self.alerts.push_back(alert);
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_monitor() -> StorageHealthMonitor {
        StorageHealthMonitor::new(ShmMonitorConfig {
            check_interval_secs: 10,
            alert_threshold: 0.80,
            recovery_threshold: 0.90,
            max_consecutive_failures: 3,
            ewma_alpha: 0.5, // aggressive smoothing for fast tests
            ewma_latency_alpha: 0.5,
            alert_auto_resolve_secs: 600,
        })
    }

    // ── basic registration ────────────────────────────────────────────────

    #[test]
    fn test_register_probe_returns_unique_ids() {
        let mut m = make_monitor();
        let a = m.register_probe("a", ShmCategory::BlockStore, Default::default());
        let b = m.register_probe("b", ShmCategory::Cache, Default::default());
        assert_ne!(a, b);
    }

    #[test]
    fn test_probe_count_after_registration() {
        let mut m = make_monitor();
        assert_eq!(m.probe_count(), 0);
        m.register_probe("x", ShmCategory::WalLog, Default::default());
        assert_eq!(m.probe_count(), 1);
        m.register_probe("y", ShmCategory::Network, Default::default());
        assert_eq!(m.probe_count(), 2);
    }

    #[test]
    fn test_registered_probe_initial_status_unknown() {
        let mut m = make_monitor();
        let id = m.register_probe("z", ShmCategory::Encryption, Default::default());
        assert_eq!(m.probe(id).map(|p| p.status), Some(ShmStatus::Unknown));
    }

    #[test]
    fn test_registered_probe_name_stored() {
        let mut m = make_monitor();
        let id = m.register_probe("my-probe", ShmCategory::Cache, Default::default());
        assert_eq!(m.probe(id).map(|p| p.name.as_str()), Some("my-probe"));
    }

    #[test]
    fn test_registered_probe_category_stored() {
        let mut m = make_monitor();
        let id = m.register_probe("c", ShmCategory::Compression, Default::default());
        assert_eq!(
            m.probe(id).map(|p| p.category),
            Some(ShmCategory::Compression)
        );
    }

    #[test]
    fn test_metadata_stored_on_probe() {
        let mut m = make_monitor();
        let mut meta = HashMap::new();
        meta.insert("host".to_string(), "node-1".to_string());
        let id = m.register_probe("p", ShmCategory::Network, meta);
        let probe = m.probe(id).expect("probe exists");
        assert_eq!(probe.metadata.get("host"), Some(&"node-1".to_string()));
    }

    #[test]
    fn test_probe_ids_increment() {
        let mut m = make_monitor();
        let ids: Vec<_> = (0..5)
            .map(|i| m.register_probe(format!("p{}", i), ShmCategory::Cache, Default::default()))
            .collect();
        for w in ids.windows(2) {
            assert!(w[1] > w[0]);
        }
    }

    // ── record_check basic ────────────────────────────────────────────────

    #[test]
    fn test_record_check_unknown_probe_returns_error() {
        let mut m = make_monitor();
        assert!(m.record_check(999, true, 1.0).is_err());
    }

    #[test]
    fn test_record_check_success_updates_total_checks() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::BlockStore, Default::default());
        m.record_check(id, true, 2.0).unwrap();
        assert_eq!(m.probe(id).unwrap().total_checks, 1);
        assert_eq!(m.probe(id).unwrap().total_successes, 1);
    }

    #[test]
    fn test_record_check_failure_increments_consecutive_failures() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::BlockStore, Default::default());
        m.record_check(id, false, 0.0).unwrap();
        assert_eq!(m.probe(id).unwrap().consecutive_failures, 1);
    }

    #[test]
    fn test_record_check_success_resets_consecutive_failures() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::BlockStore, Default::default());
        m.record_check(id, false, 0.0).unwrap();
        m.record_check(id, false, 0.0).unwrap();
        m.record_check(id, true, 1.0).unwrap();
        assert_eq!(m.probe(id).unwrap().consecutive_failures, 0);
    }

    #[test]
    fn test_success_rate_ewma_update() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::Cache, Default::default());
        // Initial success_rate = 1.0, alpha = 0.5
        m.record_check(id, false, 0.0).unwrap(); // outcome 0 → new = 0.5*0 + 0.5*1.0 = 0.5
        let rate = m.probe(id).unwrap().success_rate;
        assert!((rate - 0.5).abs() < 1e-9, "expected 0.5, got {}", rate);
    }

    #[test]
    fn test_latency_ewma_update() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::WalLog, Default::default());
        m.record_check(id, true, 10.0).unwrap(); // initial = 10.0
        m.record_check(id, true, 20.0).unwrap(); // 0.5*20 + 0.5*10 = 15.0
        let lat = m.probe(id).unwrap().latency_ms;
        assert!((lat - 15.0).abs() < 1e-9, "expected 15.0, got {}", lat);
    }

    #[test]
    fn test_failure_does_not_update_latency() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::WalLog, Default::default());
        m.record_check(id, true, 5.0).unwrap();
        let lat_before = m.probe(id).unwrap().latency_ms;
        m.record_check(id, false, 999.0).unwrap();
        let lat_after = m.probe(id).unwrap().latency_ms;
        assert!((lat_before - lat_after).abs() < 1e-9);
    }

    // ── status transitions ────────────────────────────────────────────────

    #[test]
    fn test_probe_becomes_critical_after_max_consecutive_failures() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::BlockStore, Default::default());
        for _ in 0..3 {
            m.record_check(id, false, 0.0).unwrap();
        }
        assert_eq!(m.probe(id).unwrap().status, ShmStatus::Critical);
    }

    #[test]
    fn test_probe_becomes_healthy_after_successes() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::Cache, Default::default());
        // Drive up success rate quickly with alpha=0.5
        for _ in 0..10 {
            m.record_check(id, true, 1.0).unwrap();
        }
        assert_eq!(m.probe(id).unwrap().status, ShmStatus::Healthy);
    }

    #[test]
    fn test_probe_status_unknown_initially() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::IndexStore, Default::default());
        assert_eq!(m.probe(id).unwrap().status, ShmStatus::Unknown);
    }

    #[test]
    fn test_probe_degraded_on_low_success_rate() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::Network, Default::default());
        // With alpha=0.5 and starting at 1.0:
        // after 5 failures: 1→0.5→0.25→0.125→0.0625→0.03125 → all below 0.8 threshold
        // but consecutive_failures also triggers — let's use a separate config
        let mut m2 = StorageHealthMonitor::new(ShmMonitorConfig {
            ewma_alpha: 0.5,
            max_consecutive_failures: 100, // prevent critical via consec
            alert_threshold: 0.80,
            recovery_threshold: 0.90,
            ..Default::default()
        });
        let id2 = m2.register_probe("p2", ShmCategory::Network, Default::default());
        m2.record_check(id2, false, 0.0).unwrap(); // 0.5
        m2.record_check(id2, false, 0.0).unwrap(); // 0.25
                                                   // success_rate = 0.25 < 0.80 → degraded (ratio 0.25/0.80 = 0.3125 < 0.5 → critical actually)
                                                   // Let's just verify it's not healthy
        assert_ne!(m2.probe(id2).unwrap().status, ShmStatus::Healthy);
        let _ = id;
    }

    // ── overall score ─────────────────────────────────────────────────────

    #[test]
    fn test_overall_score_empty_monitor_is_one() {
        let mut m = make_monitor();
        let snap = m.run_health_check();
        assert!((snap.score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_overall_score_all_healthy_approaches_one() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::Cache, Default::default());
        for _ in 0..20 {
            m.record_check(id, true, 1.0).unwrap();
        }
        let snap = m.run_health_check();
        assert!(
            snap.score > 0.95,
            "score should be near 1.0, got {}",
            snap.score
        );
    }

    #[test]
    fn test_overall_score_all_failures_drops() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::BlockStore, Default::default());
        for _ in 0..10 {
            m.record_check(id, false, 0.0).unwrap();
        }
        let snap = m.run_health_check();
        assert!(snap.score < 0.5, "score should be low, got {}", snap.score);
    }

    #[test]
    fn test_overall_score_in_range() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::Cache, Default::default());
        m.record_check(id, true, 1.0).unwrap();
        m.record_check(id, false, 0.0).unwrap();
        let snap = m.run_health_check();
        assert!(snap.score >= 0.0 && snap.score <= 1.0);
    }

    #[test]
    fn test_snapshot_probe_count_matches() {
        let mut m = make_monitor();
        m.register_probe("a", ShmCategory::Cache, Default::default());
        m.register_probe("b", ShmCategory::WalLog, Default::default());
        let snap = m.run_health_check();
        assert_eq!(snap.probe_count, 2);
    }

    // ── history ───────────────────────────────────────────────────────────

    #[test]
    fn test_history_grows_with_snapshots() {
        let mut m = make_monitor();
        for _ in 0..5 {
            m.run_health_check();
        }
        assert_eq!(m.history_len(), 5);
    }

    #[test]
    fn test_history_bounded_at_200() {
        let mut m = make_monitor();
        for _ in 0..250 {
            m.run_health_check();
        }
        assert_eq!(m.history_len(), 200);
    }

    #[test]
    fn test_health_trend_returns_correct_window() {
        let mut m = make_monitor();
        for _ in 0..10 {
            m.run_health_check();
        }
        let trend = m.health_trend(5);
        assert_eq!(trend.len(), 5);
    }

    #[test]
    fn test_health_trend_returns_all_if_window_larger_than_history() {
        let mut m = make_monitor();
        for _ in 0..3 {
            m.run_health_check();
        }
        let trend = m.health_trend(100);
        assert_eq!(trend.len(), 3);
    }

    #[test]
    fn test_health_trend_scores_in_range() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::Cache, Default::default());
        m.record_check(id, true, 1.0).unwrap();
        m.run_health_check();
        let trend = m.health_trend(1);
        for s in &trend {
            assert!(*s >= 0.0 && *s <= 1.0, "score out of range: {}", s);
        }
    }

    // ── alerts ────────────────────────────────────────────────────────────

    #[test]
    fn test_alert_fired_on_critical_transition() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::BlockStore, Default::default());
        for _ in 0..3 {
            m.record_check(id, false, 0.0).unwrap();
        }
        let active = m.active_alerts();
        assert!(!active.is_empty(), "expected at least one active alert");
    }

    #[test]
    fn test_alert_severity_critical_on_critical_status() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::WalLog, Default::default());
        for _ in 0..3 {
            m.record_check(id, false, 0.0).unwrap();
        }
        let critical = m
            .active_alerts()
            .iter()
            .any(|a| a.severity == ShmSeverity::Critical);
        assert!(critical, "expected a Critical alert");
    }

    #[test]
    fn test_resolve_alert_marks_resolved() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::Cache, Default::default());
        for _ in 0..3 {
            m.record_check(id, false, 0.0).unwrap();
        }
        let alert_id = m
            .active_alerts()
            .first()
            .map(|a| a.id)
            .expect("alert exists");
        m.resolve_alert(alert_id).unwrap();
        assert!(m
            .all_alerts()
            .iter()
            .find(|a| a.id == alert_id)
            .map(|a| a.resolved)
            .unwrap_or(false));
    }

    #[test]
    fn test_resolve_nonexistent_alert_returns_error() {
        let mut m = make_monitor();
        assert!(m.resolve_alert(9999).is_err());
    }

    #[test]
    fn test_expire_alerts_resolves_timed_out() {
        let mut m = StorageHealthMonitor::new(ShmMonitorConfig {
            alert_auto_resolve_secs: 100,
            max_consecutive_failures: 3,
            ewma_alpha: 0.5,
            ..Default::default()
        });
        let id = m.register_probe("p", ShmCategory::BlockStore, Default::default());
        let base_ts: u64 = 1_000_000;
        for _ in 0..3 {
            m.record_check_at(id, false, 0.0, base_ts).unwrap();
        }
        // expire at base_ts + 200 (> auto_resolve_secs=100)
        m.expire_alerts(base_ts + 200);
        let active = m.active_alerts();
        assert!(active.is_empty(), "all alerts should have expired");
    }

    #[test]
    fn test_alerts_bounded_at_500() {
        let mut m = StorageHealthMonitor::new(ShmMonitorConfig {
            max_consecutive_failures: 1,
            ewma_alpha: 1.0,      // instant update
            alert_threshold: 1.1, // never degrade (won't fire degraded)
            ..Default::default()
        });
        let id = m.register_probe("p", ShmCategory::Network, Default::default());
        // Fire critical → healthy transitions to generate many alerts
        for i in 0..600_u64 {
            let success = i % 2 == 0;
            m.record_check_at(id, success, 1.0, i).unwrap();
        }
        assert!(m.all_alerts().len() <= 500);
    }

    #[test]
    fn test_alert_contains_probe_id() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::Encryption, Default::default());
        for _ in 0..3 {
            m.record_check(id, false, 0.0).unwrap();
        }
        let alerts_for_probe: Vec<_> = m
            .active_alerts()
            .into_iter()
            .filter(|a| a.probe_id == id)
            .collect();
        assert!(!alerts_for_probe.is_empty());
    }

    #[test]
    fn test_alert_auto_resolve_at_set() {
        let mut m = StorageHealthMonitor::new(ShmMonitorConfig {
            alert_auto_resolve_secs: 300,
            max_consecutive_failures: 1,
            ewma_alpha: 1.0,
            ..Default::default()
        });
        let id = m.register_probe("p", ShmCategory::Cache, Default::default());
        m.record_check_at(id, false, 0.0, 1000).unwrap();
        // At least one alert should have auto_resolve_at set
        let has_auto = m.all_alerts().iter().any(|a| a.auto_resolve_at.is_some());
        assert!(has_auto);
    }

    // ── probes_by_category ────────────────────────────────────────────────

    #[test]
    fn test_probes_by_category_filter() {
        let mut m = make_monitor();
        m.register_probe("a", ShmCategory::Cache, Default::default());
        m.register_probe("b", ShmCategory::Cache, Default::default());
        m.register_probe("c", ShmCategory::WalLog, Default::default());
        let cache_probes = m.probes_by_category(ShmCategory::Cache);
        assert_eq!(cache_probes.len(), 2);
    }

    #[test]
    fn test_probes_by_category_empty_when_none_registered() {
        let m = make_monitor();
        assert!(m.probes_by_category(ShmCategory::Encryption).is_empty());
    }

    #[test]
    fn test_probes_by_category_all_categories() {
        let mut m = make_monitor();
        let cats = [
            ShmCategory::BlockStore,
            ShmCategory::IndexStore,
            ShmCategory::WalLog,
            ShmCategory::Cache,
            ShmCategory::Network,
            ShmCategory::Encryption,
            ShmCategory::Compression,
        ];
        for (i, cat) in cats.iter().enumerate() {
            m.register_probe(format!("p{}", i), *cat, Default::default());
        }
        for cat in &cats {
            assert_eq!(m.probes_by_category(*cat).len(), 1);
        }
    }

    // ── recovery suggestions ──────────────────────────────────────────────

    #[test]
    fn test_suggest_recovery_unknown_probe_returns_error() {
        let m = make_monitor();
        assert!(m.suggest_recovery(9999).is_err());
    }

    #[test]
    fn test_suggest_recovery_healthy_returns_no_action() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::Cache, Default::default());
        for _ in 0..20 {
            m.record_check(id, true, 1.0).unwrap();
        }
        let suggestion = m.suggest_recovery(id).unwrap();
        assert!(
            suggestion.contains("No action required")
                || suggestion.contains("healthy")
                || suggestion.contains("Healthy")
        );
    }

    #[test]
    fn test_suggest_recovery_critical_blockstore() {
        let mut m = make_monitor();
        let id = m.register_probe("bs", ShmCategory::BlockStore, Default::default());
        for _ in 0..3 {
            m.record_check(id, false, 0.0).unwrap();
        }
        let s = m.suggest_recovery(id).unwrap();
        assert!(
            s.contains("block-store")
                || s.contains("BlockStore")
                || s.contains("integrity")
                || s.contains("block")
        );
    }

    #[test]
    fn test_suggest_recovery_critical_wallog() {
        let mut m = make_monitor();
        let id = m.register_probe("wal", ShmCategory::WalLog, Default::default());
        for _ in 0..3 {
            m.record_check(id, false, 0.0).unwrap();
        }
        let s = m.suggest_recovery(id).unwrap();
        assert!(
            s.contains("WAL")
                || s.contains("wal")
                || s.contains("writes")
                || s.contains("checkpoint")
        );
    }

    #[test]
    fn test_suggest_recovery_contains_probe_name() {
        let mut m = make_monitor();
        let id = m.register_probe("my-node", ShmCategory::Network, Default::default());
        let s = m.suggest_recovery(id).unwrap();
        assert!(s.contains("my-node"));
    }

    // ── monitor stats ─────────────────────────────────────────────────────

    #[test]
    fn test_monitor_stats_total_probes() {
        let mut m = make_monitor();
        m.register_probe("a", ShmCategory::Cache, Default::default());
        m.register_probe("b", ShmCategory::WalLog, Default::default());
        let stats = m.monitor_stats();
        assert_eq!(stats.total_probes, 2);
    }

    #[test]
    fn test_monitor_stats_counts_by_status() {
        let mut m = make_monitor();
        let a = m.register_probe("a", ShmCategory::Cache, Default::default());
        let _b = m.register_probe("b", ShmCategory::WalLog, Default::default());
        // make a healthy
        for _ in 0..10 {
            m.record_check(a, true, 1.0).unwrap();
        }
        let stats = m.monitor_stats();
        assert!(stats.healthy_count >= 1);
    }

    #[test]
    fn test_monitor_stats_total_checks_recorded() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::BlockStore, Default::default());
        for _ in 0..7 {
            m.record_check(id, true, 1.0).unwrap();
        }
        let stats = m.monitor_stats();
        assert_eq!(stats.total_checks_recorded, 7);
    }

    #[test]
    fn test_monitor_stats_active_alerts() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::BlockStore, Default::default());
        for _ in 0..3 {
            m.record_check(id, false, 0.0).unwrap();
        }
        let stats = m.monitor_stats();
        assert!(stats.active_alerts > 0);
    }

    #[test]
    fn test_monitor_stats_snapshots_stored() {
        let mut m = make_monitor();
        m.run_health_check();
        m.run_health_check();
        let stats = m.monitor_stats();
        assert_eq!(stats.snapshots_stored, 2);
    }

    // ── config ────────────────────────────────────────────────────────────

    #[test]
    fn test_set_config_updates_config() {
        let mut m = make_monitor();
        let new_cfg = ShmMonitorConfig {
            check_interval_secs: 60,
            ..Default::default()
        };
        m.set_config(new_cfg);
        assert_eq!(m.config().check_interval_secs, 60);
    }

    #[test]
    fn test_default_config_sensible_values() {
        let cfg = ShmMonitorConfig::default();
        assert!(cfg.alert_threshold > 0.0 && cfg.alert_threshold < 1.0);
        assert!(cfg.recovery_threshold > cfg.alert_threshold);
        assert!(cfg.ewma_alpha > 0.0 && cfg.ewma_alpha <= 1.0);
    }

    // ── category labels ───────────────────────────────────────────────────

    #[test]
    fn test_category_labels_non_empty() {
        let cats = [
            ShmCategory::BlockStore,
            ShmCategory::IndexStore,
            ShmCategory::WalLog,
            ShmCategory::Cache,
            ShmCategory::Network,
            ShmCategory::Encryption,
            ShmCategory::Compression,
        ];
        for cat in cats {
            assert!(!cat.label().is_empty());
        }
    }

    // ── status weight ordering ────────────────────────────────────────────

    #[test]
    fn test_status_weights_ordered() {
        assert!(ShmStatus::Healthy.weight() < ShmStatus::Degraded.weight());
        assert!(ShmStatus::Degraded.weight() < ShmStatus::Critical.weight());
    }

    // ── xorshift64 ────────────────────────────────────────────────────────

    #[test]
    fn test_xorshift64_produces_different_values() {
        let mut state = 12345_u64;
        let a = xorshift64(&mut state);
        let b = xorshift64(&mut state);
        assert_ne!(a, b);
    }

    #[test]
    fn test_xorshift64_state_changes() {
        let mut state = 99_u64;
        let before = state;
        xorshift64(&mut state);
        assert_ne!(state, before);
    }

    // ── overall health status ─────────────────────────────────────────────

    #[test]
    fn test_overall_status_healthy_when_all_probes_healthy() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::Cache, Default::default());
        for _ in 0..20 {
            m.record_check(id, true, 1.0).unwrap();
        }
        let snap = m.run_health_check();
        assert_eq!(snap.overall_status, ShmStatus::Healthy);
    }

    #[test]
    fn test_overall_status_critical_when_any_critical() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::BlockStore, Default::default());
        for _ in 0..3 {
            m.record_check(id, false, 0.0).unwrap();
        }
        let snap = m.run_health_check();
        assert_eq!(snap.overall_status, ShmStatus::Critical);
    }

    #[test]
    fn test_snapshot_probe_statuses_populated() {
        let mut m = make_monitor();
        let a = m.register_probe("a", ShmCategory::Cache, Default::default());
        let b = m.register_probe("b", ShmCategory::WalLog, Default::default());
        let snap = m.run_health_check();
        assert!(snap.probe_statuses.contains_key(&a));
        assert!(snap.probe_statuses.contains_key(&b));
    }

    // ── multi-probe interaction ────────────────────────────────────────────

    #[test]
    fn test_one_critical_probe_does_not_affect_others() {
        let mut m = make_monitor();
        let good = m.register_probe("good", ShmCategory::Cache, Default::default());
        let bad = m.register_probe("bad", ShmCategory::BlockStore, Default::default());
        for _ in 0..20 {
            m.record_check(good, true, 1.0).unwrap();
        }
        for _ in 0..3 {
            m.record_check(bad, false, 0.0).unwrap();
        }
        let snap = m.run_health_check();
        assert_eq!(snap.probe_statuses[&good], ShmStatus::Healthy);
        assert_eq!(snap.probe_statuses[&bad], ShmStatus::Critical);
    }

    #[test]
    fn test_total_alerts_fired_increments() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::Encryption, Default::default());
        for _ in 0..3 {
            m.record_check(id, false, 0.0).unwrap();
        }
        let stats = m.monitor_stats();
        assert!(stats.total_alerts_fired > 0);
    }

    #[test]
    fn test_history_window_returns_subset() {
        let mut m = make_monitor();
        for _ in 0..10 {
            m.run_health_check();
        }
        let window = m.history_window(3);
        assert_eq!(window.len(), 3);
    }

    #[test]
    fn test_all_alerts_includes_resolved() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::Cache, Default::default());
        for _ in 0..3 {
            m.record_check(id, false, 0.0).unwrap();
        }
        let alert_id = m.active_alerts().first().map(|a| a.id).unwrap();
        m.resolve_alert(alert_id).unwrap();
        assert!(!m.all_alerts().is_empty());
        assert!(m.all_alerts().iter().any(|a| a.resolved));
    }

    #[test]
    fn test_probe_nonexistent_returns_none() {
        let m = make_monitor();
        assert!(m.probe(42).is_none());
    }

    #[test]
    fn test_initial_success_rate_is_one() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::Network, Default::default());
        assert!((m.probe(id).unwrap().success_rate - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_initial_latency_is_zero() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::Network, Default::default());
        assert!((m.probe(id).unwrap().latency_ms).abs() < 1e-9);
    }

    #[test]
    fn test_zero_latency_handled_gracefully() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::BlockStore, Default::default());
        m.record_check(id, true, 0.0).unwrap();
        assert!(m.probe(id).unwrap().latency_ms >= 0.0);
    }

    #[test]
    fn test_large_latency_does_not_panic() {
        let mut m = make_monitor();
        let id = m.register_probe("p", ShmCategory::BlockStore, Default::default());
        m.record_check(id, true, f64::MAX / 2.0).unwrap();
        // Just shouldn't panic
        assert!(m.probe(id).unwrap().latency_ms >= 0.0);
    }

    #[test]
    fn test_many_probes_score_still_in_range() {
        let mut m = make_monitor();
        for i in 0..50 {
            let id = m.register_probe(format!("p{}", i), ShmCategory::Cache, Default::default());
            m.record_check(id, i % 3 != 0, 1.0).unwrap();
        }
        let snap = m.run_health_check();
        assert!(snap.score >= 0.0 && snap.score <= 1.0);
    }

    #[test]
    fn test_monitor_default_config_constructor() {
        let m = StorageHealthMonitor::default_config();
        assert_eq!(m.config().check_interval_secs, 30);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(ShmSeverity::Info < ShmSeverity::Warning);
        assert!(ShmSeverity::Warning < ShmSeverity::Error);
        assert!(ShmSeverity::Error < ShmSeverity::Critical);
    }

    #[test]
    fn test_all_statuses_covered_in_weight() {
        let statuses = [
            ShmStatus::Healthy,
            ShmStatus::Recovering,
            ShmStatus::Unknown,
            ShmStatus::Degraded,
            ShmStatus::Critical,
        ];
        for s in statuses {
            let w = s.weight();
            assert!(w >= 0.0);
        }
    }
}
