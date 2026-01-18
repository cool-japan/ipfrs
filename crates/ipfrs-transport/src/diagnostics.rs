//! Diagnostic utilities for troubleshooting transport issues
//!
//! This module provides comprehensive diagnostic tools to help identify
//! and troubleshoot issues in the transport layer.

use crate::{ConcurrentPeerManager, ConcurrentWantList, Session};
use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

/// Comprehensive diagnostic report for the transport layer
#[derive(Debug, Clone)]
pub struct DiagnosticReport {
    /// Timestamp when the report was generated
    pub timestamp: std::time::SystemTime,
    /// Overall health status
    pub health_status: HealthStatus,
    /// Want list diagnostics
    pub want_list: WantListDiagnostics,
    /// Peer manager diagnostics
    pub peer_manager: PeerManagerDiagnostics,
    /// Session diagnostics (if available)
    pub sessions: Vec<SessionDiagnostics>,
    /// Identified issues
    pub issues: Vec<DiagnosticIssue>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Overall health status of the transport layer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Everything is functioning normally
    Healthy,
    /// Minor issues detected but system is functional
    Degraded,
    /// Significant issues affecting performance
    Warning,
    /// Critical issues requiring immediate attention
    Critical,
}

/// Diagnostic information about the want list
#[derive(Debug, Clone)]
pub struct WantListDiagnostics {
    /// Total number of wants
    pub total_wants: usize,
    /// Wants by priority distribution
    pub priority_distribution: HashMap<String, usize>,
    /// Number of expired wants
    pub expired_wants: usize,
    /// Number of wants awaiting retry
    pub retry_pending: usize,
    /// Average time wants are in the queue
    pub avg_queue_time: Duration,
    /// Oldest want age
    pub oldest_want_age: Option<Duration>,
}

/// Diagnostic information about peer management
#[derive(Debug, Clone)]
pub struct PeerManagerDiagnostics {
    /// Total number of peers
    pub total_peers: usize,
    /// Active peers (available for requests)
    pub active_peers: usize,
    /// Blacklisted peers
    pub blacklisted_peers: usize,
    /// Average peer score
    pub avg_peer_score: f64,
    /// Peers with circuit breaker open
    pub circuit_breaker_open: usize,
    /// Average latency across all peers
    pub avg_latency: Option<Duration>,
    /// Total bandwidth (bytes/sec)
    pub total_bandwidth: u64,
}

/// Diagnostic information about a session
#[derive(Debug, Clone)]
pub struct SessionDiagnostics {
    /// Session ID
    pub session_id: u64,
    /// Current state
    pub state: String,
    /// Number of blocks requested
    pub blocks_requested: usize,
    /// Number of blocks received
    pub blocks_received: usize,
    /// Progress percentage
    pub progress_percent: f64,
    /// Time elapsed since session start
    pub elapsed_time: Duration,
    /// Estimated time remaining
    pub estimated_remaining: Option<Duration>,
    /// Current throughput (bytes/sec)
    pub throughput: u64,
}

/// Identified issue in the transport layer
#[derive(Debug, Clone)]
pub struct DiagnosticIssue {
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue category
    pub category: IssueCategory,
    /// Human-readable description
    pub description: String,
    /// Detailed information
    pub details: Option<String>,
}

/// Severity of a diagnostic issue
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    /// Informational, no action required
    Info,
    /// Warning, may need attention
    Warning,
    /// Error, should be addressed
    Error,
    /// Critical, requires immediate action
    Critical,
}

/// Category of diagnostic issue
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueCategory {
    /// Issue with want list management
    WantList,
    /// Issue with peer connectivity or selection
    PeerManagement,
    /// Issue with session handling
    Session,
    /// Issue with network performance
    Performance,
    /// Configuration issue
    Configuration,
}

/// Main diagnostic engine for analyzing transport layer health
pub struct DiagnosticEngine {
    /// Thresholds for issue detection
    config: DiagnosticConfig,
}

/// Configuration for diagnostic analysis
#[derive(Debug, Clone)]
pub struct DiagnosticConfig {
    /// Max acceptable queue time before warning
    pub max_queue_time: Duration,
    /// Minimum acceptable number of active peers
    pub min_active_peers: usize,
    /// Minimum acceptable average peer score
    pub min_avg_score: f64,
    /// Max acceptable expired wants ratio
    pub max_expired_ratio: f64,
    /// Min acceptable session progress rate (blocks/sec)
    pub min_progress_rate: f64,
}

impl Default for DiagnosticConfig {
    fn default() -> Self {
        Self {
            max_queue_time: Duration::from_secs(60),
            min_active_peers: 3,
            min_avg_score: 0.5,
            max_expired_ratio: 0.1, // 10%
            min_progress_rate: 1.0, // At least 1 block per second
        }
    }
}

impl DiagnosticEngine {
    /// Create a new diagnostic engine with default configuration
    pub fn new() -> Self {
        Self {
            config: DiagnosticConfig::default(),
        }
    }

    /// Create a new diagnostic engine with custom configuration
    pub fn with_config(config: DiagnosticConfig) -> Self {
        Self { config }
    }

    /// Generate a comprehensive diagnostic report
    pub fn generate_report(
        &self,
        want_list: &ConcurrentWantList,
        peer_manager: &ConcurrentPeerManager,
        sessions: &[&Session],
    ) -> DiagnosticReport {
        let want_diag = self.diagnose_want_list(want_list);
        let peer_diag = self.diagnose_peer_manager(peer_manager);
        let session_diags: Vec<_> = sessions.iter().map(|s| self.diagnose_session(s)).collect();

        let mut issues = Vec::new();
        issues.extend(self.detect_want_list_issues(&want_diag));
        issues.extend(self.detect_peer_issues(&peer_diag));
        issues.extend(self.detect_session_issues(&session_diags));

        let health_status = self.determine_health_status(&issues);
        let recommendations = self.generate_recommendations(&issues, &want_diag, &peer_diag);

        DiagnosticReport {
            timestamp: std::time::SystemTime::now(),
            health_status,
            want_list: want_diag,
            peer_manager: peer_diag,
            sessions: session_diags,
            issues,
            recommendations,
        }
    }

    /// Diagnose want list health
    fn diagnose_want_list(&self, want_list: &ConcurrentWantList) -> WantListDiagnostics {
        // Get all CIDs and then get batch of entries
        let cids = want_list.cids();
        let all_wants = want_list.get_batch(&cids);
        let total_wants = all_wants.len();

        let mut priority_distribution = HashMap::new();
        let mut expired_count = 0;
        let mut retry_count = 0;
        let mut total_age = Duration::ZERO;
        let mut oldest_age = None;

        let now = std::time::Instant::now();

        for want in &all_wants {
            // Priority distribution
            let priority_label = match want.priority {
                p if p >= 900 => "Critical",
                p if p >= 700 => "Urgent",
                p if p >= 500 => "High",
                p if p >= 300 => "Normal",
                _ => "Low",
            }
            .to_string();
            *priority_distribution.entry(priority_label).or_insert(0) += 1;

            // Check if expired
            if let Some(deadline) = want.deadline {
                if now >= deadline {
                    expired_count += 1;
                }
            }

            // Check retry status
            if want.retry_count > 0 {
                retry_count += 1;
            }

            // Calculate age
            let age = now.duration_since(want.created_at);
            total_age += age;
            oldest_age = Some(oldest_age.map_or(age, |old: Duration| old.max(age)));
        }

        let avg_queue_time = if total_wants > 0 {
            total_age / total_wants as u32
        } else {
            Duration::ZERO
        };

        WantListDiagnostics {
            total_wants,
            priority_distribution,
            expired_wants: expired_count,
            retry_pending: retry_count,
            avg_queue_time,
            oldest_want_age: oldest_age,
        }
    }

    /// Diagnose peer manager health
    fn diagnose_peer_manager(
        &self,
        peer_manager: &ConcurrentPeerManager,
    ) -> PeerManagerDiagnostics {
        let stats = peer_manager.stats();

        PeerManagerDiagnostics {
            total_peers: stats.total_peers,
            active_peers: stats.connected_peers,
            blacklisted_peers: stats.blacklisted_peers,
            avg_peer_score: stats.avg_score,
            circuit_breaker_open: 0, // Would need to query each peer's circuit breaker
            avg_latency: Some(Duration::from_millis(stats.avg_latency_ms as u64)),
            total_bandwidth: 0, // PeerManagerStats doesn't track total bytes
        }
    }

    /// Diagnose session health
    fn diagnose_session(&self, session: &Session) -> SessionDiagnostics {
        let stats = session.stats();
        let state = format!("{:?}", session.state());

        let progress_percent = if stats.total_blocks > 0 {
            (stats.blocks_received as f64 / stats.total_blocks as f64) * 100.0
        } else {
            0.0
        };

        // Calculate elapsed time from timestamps
        let elapsed_time = if let Some(start) = stats.started_at {
            if let Some(end) = stats.completed_at {
                end.duration_since(start)
            } else {
                std::time::Instant::now().duration_since(start)
            }
        } else {
            Duration::ZERO
        };

        // Estimate remaining time based on current throughput
        let estimated_remaining = if stats.bytes_transferred > 0
            && stats.total_blocks > stats.blocks_received
        {
            let remaining_blocks = stats.total_blocks - stats.blocks_received;
            let blocks_per_sec = stats.blocks_received as f64 / elapsed_time.as_secs_f64().max(1.0);
            if blocks_per_sec > 0.0 {
                Some(Duration::from_secs_f64(
                    remaining_blocks as f64 / blocks_per_sec,
                ))
            } else {
                None
            }
        } else {
            None
        };

        let throughput = if elapsed_time.as_secs() > 0 {
            stats.bytes_transferred / elapsed_time.as_secs()
        } else {
            0
        };

        SessionDiagnostics {
            session_id: session.id(),
            state,
            blocks_requested: stats.total_blocks,
            blocks_received: stats.blocks_received,
            progress_percent,
            elapsed_time,
            estimated_remaining,
            throughput,
        }
    }

    /// Detect issues in want list
    fn detect_want_list_issues(&self, diag: &WantListDiagnostics) -> Vec<DiagnosticIssue> {
        let mut issues = Vec::new();

        // Check for excessive queue time
        if diag.avg_queue_time > self.config.max_queue_time {
            issues.push(DiagnosticIssue {
                severity: IssueSeverity::Warning,
                category: IssueCategory::WantList,
                description: "High average queue time for wants".to_string(),
                details: Some(format!(
                    "Average queue time is {:?}, exceeds threshold of {:?}",
                    diag.avg_queue_time, self.config.max_queue_time
                )),
            });
        }

        // Check for high expired ratio
        if diag.total_wants > 0 {
            let expired_ratio = diag.expired_wants as f64 / diag.total_wants as f64;
            if expired_ratio > self.config.max_expired_ratio {
                issues.push(DiagnosticIssue {
                    severity: IssueSeverity::Error,
                    category: IssueCategory::WantList,
                    description: "High ratio of expired wants".to_string(),
                    details: Some(format!(
                        "{:.1}% of wants have expired (threshold: {:.1}%)",
                        expired_ratio * 100.0,
                        self.config.max_expired_ratio * 100.0
                    )),
                });
            }
        }

        // Check for many retries
        if diag.retry_pending > diag.total_wants / 2 {
            issues.push(DiagnosticIssue {
                severity: IssueSeverity::Warning,
                category: IssueCategory::Performance,
                description: "High number of wants awaiting retry".to_string(),
                details: Some(format!(
                    "{} out of {} wants are awaiting retry",
                    diag.retry_pending, diag.total_wants
                )),
            });
        }

        issues
    }

    /// Detect issues in peer management
    fn detect_peer_issues(&self, diag: &PeerManagerDiagnostics) -> Vec<DiagnosticIssue> {
        let mut issues = Vec::new();

        // Check for insufficient active peers
        if diag.active_peers < self.config.min_active_peers {
            issues.push(DiagnosticIssue {
                severity: IssueSeverity::Critical,
                category: IssueCategory::PeerManagement,
                description: "Insufficient active peers".to_string(),
                details: Some(format!(
                    "Only {} active peers (minimum recommended: {})",
                    diag.active_peers, self.config.min_active_peers
                )),
            });
        }

        // Check average peer score
        if diag.avg_peer_score < self.config.min_avg_score {
            issues.push(DiagnosticIssue {
                severity: IssueSeverity::Warning,
                category: IssueCategory::PeerManagement,
                description: "Low average peer score".to_string(),
                details: Some(format!(
                    "Average peer score is {:.2} (threshold: {:.2})",
                    diag.avg_peer_score, self.config.min_avg_score
                )),
            });
        }

        // Check for high blacklist ratio
        if diag.total_peers > 0 {
            let blacklist_ratio = diag.blacklisted_peers as f64 / diag.total_peers as f64;
            if blacklist_ratio > 0.3 {
                issues.push(DiagnosticIssue {
                    severity: IssueSeverity::Warning,
                    category: IssueCategory::PeerManagement,
                    description: "High percentage of blacklisted peers".to_string(),
                    details: Some(format!(
                        "{:.1}% of peers are blacklisted",
                        blacklist_ratio * 100.0
                    )),
                });
            }
        }

        issues
    }

    /// Detect issues in sessions
    fn detect_session_issues(&self, sessions: &[SessionDiagnostics]) -> Vec<DiagnosticIssue> {
        let mut issues = Vec::new();

        for session in sessions {
            // Check for stalled sessions
            if session.blocks_requested > 0
                && session.progress_percent < 10.0
                && session.elapsed_time > Duration::from_secs(30)
            {
                issues.push(DiagnosticIssue {
                    severity: IssueSeverity::Warning,
                    category: IssueCategory::Session,
                    description: format!("Session {} appears stalled", session.session_id),
                    details: Some(format!(
                        "Only {:.1}% progress after {:?}",
                        session.progress_percent, session.elapsed_time
                    )),
                });
            }

            // Check for low throughput
            if session.elapsed_time > Duration::from_secs(10) && session.throughput < 10_000 {
                // < 10 KB/s
                issues.push(DiagnosticIssue {
                    severity: IssueSeverity::Warning,
                    category: IssueCategory::Performance,
                    description: format!("Low throughput for session {}", session.session_id),
                    details: Some(format!(
                        "Current throughput: {} bytes/sec",
                        session.throughput
                    )),
                });
            }
        }

        issues
    }

    /// Determine overall health status based on issues
    fn determine_health_status(&self, issues: &[DiagnosticIssue]) -> HealthStatus {
        let mut max_severity = IssueSeverity::Info;

        for issue in issues {
            if issue.severity > max_severity {
                max_severity = issue.severity;
            }
        }

        match max_severity {
            IssueSeverity::Info => HealthStatus::Healthy,
            IssueSeverity::Warning => HealthStatus::Degraded,
            IssueSeverity::Error => HealthStatus::Warning,
            IssueSeverity::Critical => HealthStatus::Critical,
        }
    }

    /// Generate recommendations based on detected issues
    fn generate_recommendations(
        &self,
        issues: &[DiagnosticIssue],
        want_diag: &WantListDiagnostics,
        peer_diag: &PeerManagerDiagnostics,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Recommendations based on issues
        for issue in issues {
            match issue.category {
                IssueCategory::PeerManagement
                    if peer_diag.active_peers < self.config.min_active_peers =>
                {
                    recommendations.push(
                        "Consider connecting to more peers to improve redundancy and performance"
                            .to_string(),
                    );
                }
                IssueCategory::WantList if want_diag.expired_wants > 0 => {
                    recommendations.push(
                        "Consider increasing timeout duration or improving network connectivity"
                            .to_string(),
                    );
                }
                IssueCategory::Performance => {
                    recommendations.push(
                        "Check network conditions and consider adjusting batch sizes or concurrency limits".to_string()
                    );
                }
                _ => {}
            }
        }

        // General recommendations based on stats
        if peer_diag.avg_peer_score < 0.7 {
            recommendations.push(
                "Peer quality is suboptimal. Consider finding better peers or adjusting scoring weights".to_string()
            );
        }

        if want_diag.total_wants > 1000 {
            recommendations.push(
                "Large number of pending wants. Consider increasing max_concurrent_blocks or adding more peers".to_string()
            );
        }

        recommendations.dedup();
        recommendations
    }
}

impl Default for DiagnosticEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for DiagnosticReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Transport Diagnostic Report ===")?;
        writeln!(f, "Generated: {:?}", self.timestamp)?;
        writeln!(f, "Health Status: {:?}", self.health_status)?;
        writeln!(f)?;

        writeln!(f, "Want List:")?;
        writeln!(f, "  Total wants: {}", self.want_list.total_wants)?;
        writeln!(f, "  Expired: {}", self.want_list.expired_wants)?;
        writeln!(f, "  Retry pending: {}", self.want_list.retry_pending)?;
        writeln!(f, "  Avg queue time: {:?}", self.want_list.avg_queue_time)?;
        writeln!(f)?;

        writeln!(f, "Peer Manager:")?;
        writeln!(f, "  Total peers: {}", self.peer_manager.total_peers)?;
        writeln!(f, "  Active: {}", self.peer_manager.active_peers)?;
        writeln!(f, "  Blacklisted: {}", self.peer_manager.blacklisted_peers)?;
        writeln!(f, "  Avg score: {:.2}", self.peer_manager.avg_peer_score)?;
        writeln!(f)?;

        if !self.sessions.is_empty() {
            writeln!(f, "Sessions:")?;
            for session in &self.sessions {
                writeln!(
                    f,
                    "  Session {}: {:.1}% complete ({}/{})",
                    session.session_id,
                    session.progress_percent,
                    session.blocks_received,
                    session.blocks_requested
                )?;
            }
            writeln!(f)?;
        }

        if !self.issues.is_empty() {
            writeln!(f, "Issues:")?;
            for issue in &self.issues {
                writeln!(
                    f,
                    "  [{:?}] {}: {}",
                    issue.severity, issue.category, issue.description
                )?;
                if let Some(details) = &issue.details {
                    writeln!(f, "      {}", details)?;
                }
            }
            writeln!(f)?;
        }

        if !self.recommendations.is_empty() {
            writeln!(f, "Recommendations:")?;
            for (i, rec) in self.recommendations.iter().enumerate() {
                writeln!(f, "  {}. {}", i + 1, rec)?;
            }
        }

        Ok(())
    }
}

impl fmt::Display for IssueCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IssueCategory::WantList => write!(f, "WantList"),
            IssueCategory::PeerManagement => write!(f, "PeerManagement"),
            IssueCategory::Session => write!(f, "Session"),
            IssueCategory::Performance => write!(f, "Performance"),
            IssueCategory::Configuration => write!(f, "Configuration"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Priority, WantListConfig};
    use ipfrs_core::Cid;
    use std::time::Duration;

    #[test]
    fn test_diagnostic_engine_creation() {
        let engine = DiagnosticEngine::new();
        assert_eq!(engine.config.min_active_peers, 3);

        let custom_config = DiagnosticConfig {
            min_active_peers: 5,
            ..Default::default()
        };
        let engine = DiagnosticEngine::with_config(custom_config);
        assert_eq!(engine.config.min_active_peers, 5);
    }

    #[test]
    fn test_empty_report() {
        let engine = DiagnosticEngine::new();
        let want_list = ConcurrentWantList::new(WantListConfig::default());
        let peer_manager = ConcurrentPeerManager::new(Default::default());

        let report = engine.generate_report(&want_list, &peer_manager, &[]);

        assert_eq!(report.want_list.total_wants, 0);
        assert_eq!(report.peer_manager.total_peers, 0);
        assert_eq!(report.sessions.len(), 0);
    }

    #[test]
    fn test_detect_low_peer_count() {
        let engine = DiagnosticEngine::new();
        let want_list = ConcurrentWantList::new(WantListConfig::default());
        let peer_manager = ConcurrentPeerManager::new(Default::default());

        let report = engine.generate_report(&want_list, &peer_manager, &[]);

        // Should detect insufficient peers
        assert!(report
            .issues
            .iter()
            .any(|i| matches!(i.category, IssueCategory::PeerManagement)));
        assert_eq!(report.health_status, HealthStatus::Critical);
    }

    #[test]
    fn test_want_list_diagnostics() {
        let engine = DiagnosticEngine::new();
        let want_list = ConcurrentWantList::new(WantListConfig::default());

        // Add a want (note: Cid::default() creates the same CID each time)
        let cid1 = Cid::default();
        want_list.add_simple(cid1, Priority::High as i32);

        let peer_manager = ConcurrentPeerManager::new(Default::default());
        let report = engine.generate_report(&want_list, &peer_manager, &[]);

        assert_eq!(report.want_list.total_wants, 1);
        assert!(!report.want_list.priority_distribution.is_empty());
    }

    #[test]
    fn test_health_status_determination() {
        let engine = DiagnosticEngine::new();

        // No issues - healthy
        let issues = vec![];
        assert_eq!(
            engine.determine_health_status(&issues),
            HealthStatus::Healthy
        );

        // Warning issue
        let issues = vec![DiagnosticIssue {
            severity: IssueSeverity::Warning,
            category: IssueCategory::Performance,
            description: "Test".to_string(),
            details: None,
        }];
        assert_eq!(
            engine.determine_health_status(&issues),
            HealthStatus::Degraded
        );

        // Critical issue
        let issues = vec![DiagnosticIssue {
            severity: IssueSeverity::Critical,
            category: IssueCategory::PeerManagement,
            description: "Test".to_string(),
            details: None,
        }];
        assert_eq!(
            engine.determine_health_status(&issues),
            HealthStatus::Critical
        );
    }

    #[test]
    fn test_report_display() {
        let engine = DiagnosticEngine::new();
        let want_list = ConcurrentWantList::new(WantListConfig::default());
        let peer_manager = ConcurrentPeerManager::new(Default::default());

        let report = engine.generate_report(&want_list, &peer_manager, &[]);
        let display = format!("{}", report);

        assert!(display.contains("Transport Diagnostic Report"));
        assert!(display.contains("Health Status"));
        assert!(display.contains("Want List"));
        assert!(display.contains("Peer Manager"));
    }

    #[test]
    fn test_issue_severity_ordering() {
        assert!(IssueSeverity::Critical > IssueSeverity::Error);
        assert!(IssueSeverity::Error > IssueSeverity::Warning);
        assert!(IssueSeverity::Warning > IssueSeverity::Info);
    }

    #[test]
    fn test_recommendations_generation() {
        let engine = DiagnosticEngine::new();
        let want_list = ConcurrentWantList::new(WantListConfig::default());
        let peer_manager = ConcurrentPeerManager::new(Default::default());

        let report = engine.generate_report(&want_list, &peer_manager, &[]);

        // Should have recommendations due to low peer count
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_diagnostic_config_default() {
        let config = DiagnosticConfig::default();
        assert_eq!(config.min_active_peers, 3);
        assert_eq!(config.max_queue_time, Duration::from_secs(60));
        assert_eq!(config.min_avg_score, 0.5);
    }
}
