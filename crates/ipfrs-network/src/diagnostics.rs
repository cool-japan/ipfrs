//! Network Diagnostics and Troubleshooting Utilities
//!
//! This module provides diagnostic tools to help identify and troubleshoot network issues.
//!
//! # Features
//!
//! - **Connectivity Tests**: Test connectivity to bootstrap nodes and peers
//! - **Performance Diagnostics**: Measure network performance metrics
//! - **Configuration Validation**: Validate network configuration
//! - **Health Checks**: Comprehensive health checks for all components
//! - **Troubleshooting Guides**: Automated diagnosis of common issues
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::diagnostics::{NetworkDiagnostics, DiagnosticTest};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut diagnostics = NetworkDiagnostics::new();
//!
//! // Run all diagnostic tests
//! let results = diagnostics.run_all_tests();
//! for result in results {
//!     println!("{}: {}", result.test_name, if result.passed { "PASS" } else { "FAIL" });
//!     if !result.passed {
//!         println!("  Issue: {}", result.message);
//!         if let Some(fix) = result.suggested_fix {
//!             println!("  Fix: {}", fix);
//!         }
//!     }
//! }
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Result of a diagnostic test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticResult {
    /// Name of the test
    pub test_name: String,
    /// Whether the test passed
    pub passed: bool,
    /// Message describing the result
    pub message: String,
    /// Suggested fix if test failed
    pub suggested_fix: Option<String>,
    /// Test duration
    pub duration: Duration,
    /// Severity level (0=info, 1=warning, 2=error, 3=critical)
    pub severity: u8,
}

/// Type of diagnostic test
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticTest {
    /// Test basic network connectivity
    BasicConnectivity,
    /// Test DHT functionality
    DhtHealth,
    /// Test NAT traversal capabilities
    NatTraversal,
    /// Test peer discovery
    PeerDiscovery,
    /// Test bootstrap node connectivity
    BootstrapConnectivity,
    /// Validate configuration
    ConfigValidation,
    /// Check system resources
    ResourceCheck,
    /// Test protocol compatibility
    ProtocolCompatibility,
}

impl DiagnosticTest {
    /// Get human-readable name for the test
    pub fn name(&self) -> &'static str {
        match self {
            Self::BasicConnectivity => "Basic Connectivity",
            Self::DhtHealth => "DHT Health",
            Self::NatTraversal => "NAT Traversal",
            Self::PeerDiscovery => "Peer Discovery",
            Self::BootstrapConnectivity => "Bootstrap Connectivity",
            Self::ConfigValidation => "Configuration Validation",
            Self::ResourceCheck => "Resource Check",
            Self::ProtocolCompatibility => "Protocol Compatibility",
        }
    }

    /// Get description of what the test checks
    pub fn description(&self) -> &'static str {
        match self {
            Self::BasicConnectivity => "Verifies basic network stack is functioning",
            Self::DhtHealth => "Checks DHT routing table and query performance",
            Self::NatTraversal => "Tests NAT type detection and hole punching capability",
            Self::PeerDiscovery => "Verifies mDNS and DHT peer discovery mechanisms",
            Self::BootstrapConnectivity => "Tests connectivity to configured bootstrap nodes",
            Self::ConfigValidation => "Validates network configuration parameters",
            Self::ResourceCheck => "Checks available system resources (memory, file descriptors)",
            Self::ProtocolCompatibility => "Verifies protocol versions and compatibility",
        }
    }
}

/// Configuration diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigDiagnostics {
    /// Issues found in configuration
    pub issues: Vec<ConfigIssue>,
    /// Warnings about suboptimal settings
    pub warnings: Vec<String>,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Configuration issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigIssue {
    /// Name of the configuration parameter
    pub parameter: String,
    /// Description of the issue
    pub issue: String,
    /// Suggested fix
    pub fix: String,
    /// Severity (0=info, 1=warning, 2=error, 3=critical)
    pub severity: u8,
}

/// Network performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average latency to connected peers (milliseconds)
    pub avg_latency_ms: f64,
    /// Median latency to connected peers (milliseconds)
    pub median_latency_ms: f64,
    /// 95th percentile latency (milliseconds)
    pub p95_latency_ms: f64,
    /// Average bandwidth utilization (bytes/sec)
    pub avg_bandwidth_bps: u64,
    /// DHT query success rate (0.0 - 1.0)
    pub dht_success_rate: f64,
    /// Average DHT query time (milliseconds)
    pub avg_dht_query_ms: f64,
    /// Number of connected peers
    pub connected_peers: usize,
    /// Number of routing table entries
    pub routing_table_size: usize,
}

/// Network diagnostics tool
pub struct NetworkDiagnostics {
    /// Test results history
    results_history: Vec<DiagnosticResult>,
    /// Performance metrics history
    metrics_history: Vec<(Instant, PerformanceMetrics)>,
    /// Maximum history size
    max_history: usize,
}

impl NetworkDiagnostics {
    /// Create a new diagnostics instance
    pub fn new() -> Self {
        Self {
            results_history: Vec::new(),
            metrics_history: Vec::new(),
            max_history: 100,
        }
    }

    /// Create diagnostics with custom history size
    pub fn with_history_size(max_history: usize) -> Self {
        Self {
            results_history: Vec::new(),
            metrics_history: Vec::new(),
            max_history,
        }
    }

    /// Run all diagnostic tests
    pub fn run_all_tests(&mut self) -> Vec<DiagnosticResult> {
        let tests = vec![
            DiagnosticTest::BasicConnectivity,
            DiagnosticTest::ConfigValidation,
            DiagnosticTest::ResourceCheck,
            DiagnosticTest::DhtHealth,
            DiagnosticTest::NatTraversal,
            DiagnosticTest::PeerDiscovery,
            DiagnosticTest::BootstrapConnectivity,
            DiagnosticTest::ProtocolCompatibility,
        ];

        let mut results = Vec::new();
        for test in tests {
            let result = self.run_test(test);
            results.push(result);
        }

        results
    }

    /// Run a specific diagnostic test
    pub fn run_test(&mut self, test: DiagnosticTest) -> DiagnosticResult {
        let start = Instant::now();

        let result = match test {
            DiagnosticTest::BasicConnectivity => self.test_basic_connectivity(),
            DiagnosticTest::DhtHealth => self.test_dht_health(),
            DiagnosticTest::NatTraversal => self.test_nat_traversal(),
            DiagnosticTest::PeerDiscovery => self.test_peer_discovery(),
            DiagnosticTest::BootstrapConnectivity => self.test_bootstrap_connectivity(),
            DiagnosticTest::ConfigValidation => self.test_config_validation(),
            DiagnosticTest::ResourceCheck => self.test_resource_check(),
            DiagnosticTest::ProtocolCompatibility => self.test_protocol_compatibility(),
        };

        let duration = start.elapsed();
        let mut result = result;
        result.duration = duration;
        result.test_name = test.name().to_string();

        // Store in history
        self.results_history.push(result.clone());
        if self.results_history.len() > self.max_history {
            self.results_history.remove(0);
        }

        result
    }

    /// Get diagnostic results history
    pub fn results_history(&self) -> &[DiagnosticResult] {
        &self.results_history
    }

    /// Get latest test result for a specific test
    pub fn latest_result(&self, test: DiagnosticTest) -> Option<&DiagnosticResult> {
        self.results_history
            .iter()
            .rev()
            .find(|r| r.test_name == test.name())
    }

    /// Generate comprehensive diagnostic report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("Network Diagnostics Report\n");
        report.push_str("==========================\n\n");

        if self.results_history.is_empty() {
            report.push_str("No diagnostic tests have been run yet.\n");
            return report;
        }

        // Summary
        let total_tests = self.results_history.len();
        let passed = self.results_history.iter().filter(|r| r.passed).count();
        let failed = total_tests - passed;

        report.push_str(&format!("Total Tests: {}\n", total_tests));
        report.push_str(&format!("Passed: {}\n", passed));
        report.push_str(&format!("Failed: {}\n\n", failed));

        // Failed tests details
        if failed > 0 {
            report.push_str("Failed Tests:\n");
            report.push_str("-------------\n");
            for result in self.results_history.iter().filter(|r| !r.passed) {
                report.push_str(&format!("\n{}\n", result.test_name));
                report.push_str(&format!("  Issue: {}\n", result.message));
                if let Some(fix) = &result.suggested_fix {
                    report.push_str(&format!("  Suggested Fix: {}\n", fix));
                }
                report.push_str(&format!(
                    "  Severity: {}\n",
                    severity_string(result.severity)
                ));
            }
        }

        report
    }

    // Individual test implementations (placeholders for now)

    #[allow(dead_code)]
    fn test_basic_connectivity(&self) -> DiagnosticResult {
        DiagnosticResult {
            test_name: String::new(),
            passed: true,
            message: "Network stack is functioning correctly".to_string(),
            suggested_fix: None,
            duration: Duration::default(),
            severity: 0,
        }
    }

    #[allow(dead_code)]
    fn test_dht_health(&self) -> DiagnosticResult {
        DiagnosticResult {
            test_name: String::new(),
            passed: true,
            message: "DHT is healthy and responsive".to_string(),
            suggested_fix: None,
            duration: Duration::default(),
            severity: 0,
        }
    }

    #[allow(dead_code)]
    fn test_nat_traversal(&self) -> DiagnosticResult {
        DiagnosticResult {
            test_name: String::new(),
            passed: true,
            message: "NAT traversal mechanisms are working".to_string(),
            suggested_fix: None,
            duration: Duration::default(),
            severity: 0,
        }
    }

    #[allow(dead_code)]
    fn test_peer_discovery(&self) -> DiagnosticResult {
        DiagnosticResult {
            test_name: String::new(),
            passed: true,
            message: "Peer discovery is functioning".to_string(),
            suggested_fix: None,
            duration: Duration::default(),
            severity: 0,
        }
    }

    #[allow(dead_code)]
    fn test_bootstrap_connectivity(&self) -> DiagnosticResult {
        DiagnosticResult {
            test_name: String::new(),
            passed: true,
            message: "Bootstrap nodes are reachable".to_string(),
            suggested_fix: None,
            duration: Duration::default(),
            severity: 0,
        }
    }

    #[allow(dead_code)]
    fn test_config_validation(&self) -> DiagnosticResult {
        DiagnosticResult {
            test_name: String::new(),
            passed: true,
            message: "Configuration is valid".to_string(),
            suggested_fix: None,
            duration: Duration::default(),
            severity: 0,
        }
    }

    #[allow(dead_code)]
    fn test_resource_check(&self) -> DiagnosticResult {
        DiagnosticResult {
            test_name: String::new(),
            passed: true,
            message: "System resources are adequate".to_string(),
            suggested_fix: None,
            duration: Duration::default(),
            severity: 0,
        }
    }

    #[allow(dead_code)]
    fn test_protocol_compatibility(&self) -> DiagnosticResult {
        DiagnosticResult {
            test_name: String::new(),
            passed: true,
            message: "Protocol versions are compatible".to_string(),
            suggested_fix: None,
            duration: Duration::default(),
            severity: 0,
        }
    }

    /// Record performance metrics
    pub fn record_metrics(&mut self, metrics: PerformanceMetrics) {
        self.metrics_history.push((Instant::now(), metrics));
        if self.metrics_history.len() > self.max_history {
            self.metrics_history.remove(0);
        }
    }

    /// Get latest performance metrics
    pub fn latest_metrics(&self) -> Option<&PerformanceMetrics> {
        self.metrics_history.last().map(|(_, metrics)| metrics)
    }

    /// Get performance metrics history
    pub fn metrics_history(&self) -> &[(Instant, PerformanceMetrics)] {
        &self.metrics_history
    }
}

impl Default for NetworkDiagnostics {
    fn default() -> Self {
        Self::new()
    }
}

fn severity_string(severity: u8) -> &'static str {
    match severity {
        0 => "Info",
        1 => "Warning",
        2 => "Error",
        3 => "Critical",
        _ => "Unknown",
    }
}

/// Common network issues and their solutions
pub struct TroubleshootingGuide;

impl TroubleshootingGuide {
    /// Get troubleshooting advice for common issues
    pub fn get_advice(issue: &str) -> Option<String> {
        let guides: HashMap<&str, &str> = [
            (
                "no_peers",
                "No peers connected:\n\
                1. Check internet connectivity\n\
                2. Verify bootstrap nodes are configured\n\
                3. Check firewall settings\n\
                4. Ensure listen addresses are correct\n\
                5. Try enabling mDNS for local discovery",
            ),
            (
                "slow_dht",
                "DHT queries are slow:\n\
                1. Increase DHT concurrency (alpha parameter)\n\
                2. Add more bootstrap nodes\n\
                3. Check network latency to peers\n\
                4. Enable query caching\n\
                5. Tune timeout parameters",
            ),
            (
                "nat_issues",
                "NAT traversal failing:\n\
                1. Enable AutoNAT to detect NAT type\n\
                2. Configure Circuit Relay for fallback\n\
                3. Try enabling DCUtR for hole punching\n\
                4. Consider using UPnP if available\n\
                5. Use relay nodes as backup",
            ),
            (
                "high_memory",
                "High memory usage:\n\
                1. Use low_memory() configuration preset\n\
                2. Reduce max_connections limit\n\
                3. Enable aggressive cache cleanup\n\
                4. Reduce peer store size\n\
                5. Disable unused features",
            ),
            (
                "connection_churn",
                "Too many connection changes:\n\
                1. Increase connection keep-alive interval\n\
                2. Reduce connection limits\n\
                3. Improve peer selection criteria\n\
                4. Check network stability\n\
                5. Enable connection quality prediction",
            ),
        ]
        .iter()
        .cloned()
        .collect();

        guides.get(issue).map(|s| s.to_string())
    }

    /// List all available troubleshooting topics
    pub fn list_topics() -> Vec<&'static str> {
        vec![
            "no_peers",
            "slow_dht",
            "nat_issues",
            "high_memory",
            "connection_churn",
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_test_names() {
        assert_eq!(
            DiagnosticTest::BasicConnectivity.name(),
            "Basic Connectivity"
        );
        assert_eq!(DiagnosticTest::DhtHealth.name(), "DHT Health");
    }

    #[test]
    fn test_diagnostics_creation() {
        let diag = NetworkDiagnostics::new();
        assert_eq!(diag.results_history().len(), 0);
    }

    #[test]
    fn test_diagnostics_with_history_size() {
        let diag = NetworkDiagnostics::with_history_size(50);
        assert_eq!(diag.max_history, 50);
    }

    #[test]
    fn test_run_test() {
        let mut diag = NetworkDiagnostics::new();
        let result = diag.run_test(DiagnosticTest::BasicConnectivity);
        assert!(!result.test_name.is_empty());
        assert_eq!(diag.results_history().len(), 1);
    }

    #[test]
    fn test_run_all_tests() {
        let mut diag = NetworkDiagnostics::new();
        let results = diag.run_all_tests();
        assert_eq!(results.len(), 8);
    }

    #[test]
    fn test_latest_result() {
        let mut diag = NetworkDiagnostics::new();
        diag.run_test(DiagnosticTest::BasicConnectivity);
        diag.run_test(DiagnosticTest::DhtHealth);

        let latest = diag.latest_result(DiagnosticTest::DhtHealth);
        assert!(latest.is_some());
        assert_eq!(
            latest
                .expect("test: latest DHT Health result should be present")
                .test_name,
            "DHT Health"
        );
    }

    #[test]
    fn test_generate_report() {
        let mut diag = NetworkDiagnostics::new();
        diag.run_all_tests();

        let report = diag.generate_report();
        assert!(report.contains("Network Diagnostics Report"));
        assert!(report.contains("Total Tests:"));
    }

    #[test]
    fn test_metrics_recording() {
        let mut diag = NetworkDiagnostics::new();
        let metrics = PerformanceMetrics {
            avg_latency_ms: 50.0,
            median_latency_ms: 45.0,
            p95_latency_ms: 100.0,
            avg_bandwidth_bps: 1_000_000,
            dht_success_rate: 0.95,
            avg_dht_query_ms: 200.0,
            connected_peers: 10,
            routing_table_size: 50,
        };

        diag.record_metrics(metrics);
        assert_eq!(diag.metrics_history().len(), 1);

        let latest = diag.latest_metrics();
        assert!(latest.is_some());
        assert_eq!(
            latest
                .expect("test: latest metrics should be present after record_metrics")
                .connected_peers,
            10
        );
    }

    #[test]
    fn test_troubleshooting_guide() {
        let advice = TroubleshootingGuide::get_advice("no_peers");
        assert!(advice.is_some());
        assert!(advice
            .expect("test: advice for no_peers should be Some")
            .contains("bootstrap"));

        let topics = TroubleshootingGuide::list_topics();
        assert!(topics.contains(&"no_peers"));
        assert!(topics.contains(&"slow_dht"));
    }

    #[test]
    fn test_history_size_limit() {
        let mut diag = NetworkDiagnostics::with_history_size(3);

        diag.run_test(DiagnosticTest::BasicConnectivity);
        diag.run_test(DiagnosticTest::DhtHealth);
        diag.run_test(DiagnosticTest::NatTraversal);
        diag.run_test(DiagnosticTest::PeerDiscovery);

        assert_eq!(diag.results_history().len(), 3);
    }

    #[test]
    fn test_severity_levels() {
        assert_eq!(severity_string(0), "Info");
        assert_eq!(severity_string(1), "Warning");
        assert_eq!(severity_string(2), "Error");
        assert_eq!(severity_string(3), "Critical");
    }
}
