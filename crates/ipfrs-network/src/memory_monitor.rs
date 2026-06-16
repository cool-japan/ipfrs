//! Memory usage monitoring for network components
//!
//! This module provides memory tracking and monitoring capabilities for:
//! - Peer store memory usage
//! - Connection buffer sizes
//! - Cache memory consumption
//! - DHT routing table memory
//! - Memory budgets and limits

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Errors that can occur during memory monitoring
#[derive(Error, Debug, Clone)]
pub enum MemoryMonitorError {
    #[error("Memory budget exceeded: {0} bytes over limit")]
    BudgetExceeded(usize),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Component not found: {0}")]
    ComponentNotFound(String),
}

/// Configuration for memory monitoring
#[derive(Debug, Clone)]
pub struct MemoryMonitorConfig {
    /// Enable memory monitoring
    pub enabled: bool,

    /// Total memory budget in bytes (None = unlimited)
    pub total_budget: Option<usize>,

    /// Per-component memory budgets
    pub component_budgets: HashMap<String, usize>,

    /// Enable automatic memory cleanup when approaching limits
    pub enable_auto_cleanup: bool,

    /// Cleanup threshold (fraction of budget, 0.0-1.0)
    pub cleanup_threshold: f64,

    /// Monitoring interval
    pub monitoring_interval: Duration,

    /// Enable memory leak detection
    pub enable_leak_detection: bool,

    /// Growth rate threshold for leak detection (bytes per second)
    pub leak_detection_threshold: f64,
}

impl Default for MemoryMonitorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            total_budget: None,
            component_budgets: HashMap::new(),
            enable_auto_cleanup: true,
            cleanup_threshold: 0.9,
            monitoring_interval: Duration::from_secs(10),
            enable_leak_detection: false,
            leak_detection_threshold: 1_000_000.0, // 1 MB/s growth
        }
    }
}

impl MemoryMonitorConfig {
    /// Configuration for low-memory devices (128 MB budget)
    pub fn low_memory() -> Self {
        let mut budgets = HashMap::new();
        budgets.insert("peer_store".to_string(), 10 * 1024 * 1024); // 10 MB
        budgets.insert("dht_cache".to_string(), 20 * 1024 * 1024); // 20 MB
        budgets.insert("provider_cache".to_string(), 10 * 1024 * 1024); // 10 MB
        budgets.insert("connections".to_string(), 30 * 1024 * 1024); // 30 MB
        budgets.insert("other".to_string(), 58 * 1024 * 1024); // 58 MB

        Self {
            enabled: true,
            total_budget: Some(128 * 1024 * 1024), // 128 MB
            component_budgets: budgets,
            enable_auto_cleanup: true,
            cleanup_threshold: 0.85,
            monitoring_interval: Duration::from_secs(5),
            enable_leak_detection: true,
            leak_detection_threshold: 100_000.0,
        }
    }

    /// Configuration for IoT devices (64 MB budget)
    pub fn iot() -> Self {
        let mut budgets = HashMap::new();
        budgets.insert("peer_store".to_string(), 5 * 1024 * 1024); // 5 MB
        budgets.insert("dht_cache".to_string(), 10 * 1024 * 1024); // 10 MB
        budgets.insert("provider_cache".to_string(), 5 * 1024 * 1024); // 5 MB
        budgets.insert("connections".to_string(), 20 * 1024 * 1024); // 20 MB
        budgets.insert("other".to_string(), 24 * 1024 * 1024); // 24 MB

        Self {
            enabled: true,
            total_budget: Some(64 * 1024 * 1024), // 64 MB
            component_budgets: budgets,
            enable_auto_cleanup: true,
            cleanup_threshold: 0.8,
            monitoring_interval: Duration::from_secs(3),
            enable_leak_detection: true,
            leak_detection_threshold: 50_000.0,
        }
    }

    /// Configuration for mobile devices (256 MB budget)
    pub fn mobile() -> Self {
        let mut budgets = HashMap::new();
        budgets.insert("peer_store".to_string(), 20 * 1024 * 1024); // 20 MB
        budgets.insert("dht_cache".to_string(), 50 * 1024 * 1024); // 50 MB
        budgets.insert("provider_cache".to_string(), 20 * 1024 * 1024); // 20 MB
        budgets.insert("connections".to_string(), 100 * 1024 * 1024); // 100 MB
        budgets.insert("other".to_string(), 66 * 1024 * 1024); // 66 MB

        Self {
            enabled: true,
            total_budget: Some(256 * 1024 * 1024), // 256 MB
            component_budgets: budgets,
            enable_auto_cleanup: true,
            cleanup_threshold: 0.9,
            monitoring_interval: Duration::from_secs(10),
            enable_leak_detection: true,
            leak_detection_threshold: 500_000.0,
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), MemoryMonitorError> {
        if self.cleanup_threshold < 0.0 || self.cleanup_threshold > 1.0 {
            return Err(MemoryMonitorError::InvalidConfig(
                "cleanup_threshold must be in [0.0, 1.0]".to_string(),
            ));
        }

        if let Some(total) = self.total_budget {
            let component_total: usize = self.component_budgets.values().sum();
            if component_total > total {
                return Err(MemoryMonitorError::InvalidConfig(format!(
                    "Component budgets ({}) exceed total budget ({})",
                    component_total, total
                )));
            }
        }

        Ok(())
    }
}

/// Memory usage for a component
#[derive(Debug, Clone, Default)]
pub struct ComponentMemory {
    /// Component name
    pub name: String,
    /// Current memory usage in bytes
    pub current_usage: usize,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Number of allocations
    pub allocation_count: u64,
    /// Last update time
    pub last_updated: Option<Instant>,
    /// Memory budget for this component
    pub budget: Option<usize>,
}

impl ComponentMemory {
    fn new(name: String, budget: Option<usize>) -> Self {
        Self {
            name,
            current_usage: 0,
            peak_usage: 0,
            allocation_count: 0,
            last_updated: Some(Instant::now()),
            budget,
        }
    }

    /// Check if over budget
    pub fn is_over_budget(&self) -> bool {
        if let Some(budget) = self.budget {
            self.current_usage > budget
        } else {
            false
        }
    }

    /// Get budget utilization (0.0-1.0+)
    pub fn budget_utilization(&self) -> Option<f64> {
        self.budget
            .map(|budget| self.current_usage as f64 / budget as f64)
    }
}

/// Memory monitoring state
struct MonitorState {
    /// Memory usage per component
    components: HashMap<String, ComponentMemory>,
    /// Total memory usage
    total_usage: usize,
    /// Peak total usage
    peak_total_usage: usize,
    /// Last cleanup time
    last_cleanup: Instant,
    /// Memory samples for leak detection
    memory_samples: Vec<(Instant, usize)>,
    /// Number of cleanup operations performed
    cleanup_count: u64,
}

impl MonitorState {
    fn new() -> Self {
        Self {
            components: HashMap::new(),
            total_usage: 0,
            peak_total_usage: 0,
            last_cleanup: Instant::now(),
            memory_samples: Vec::new(),
            cleanup_count: 0,
        }
    }
}

/// Memory monitor for network components
pub struct MemoryMonitor {
    config: MemoryMonitorConfig,
    state: Arc<RwLock<MonitorState>>,
}

impl MemoryMonitor {
    /// Create a new memory monitor
    pub fn new(config: MemoryMonitorConfig) -> Result<Self, MemoryMonitorError> {
        config.validate()?;

        let mut state = MonitorState::new();

        // Initialize component budgets
        for (name, budget) in &config.component_budgets {
            state.components.insert(
                name.clone(),
                ComponentMemory::new(name.clone(), Some(*budget)),
            );
        }

        Ok(Self {
            config,
            state: Arc::new(RwLock::new(state)),
        })
    }

    /// Record memory usage for a component
    pub fn record_usage(&self, component: &str, bytes: usize) -> Result<(), MemoryMonitorError> {
        if !self.config.enabled {
            return Ok(());
        }

        let mut state = self.state.write();
        let now = Instant::now();

        // Get or create component
        let comp = state
            .components
            .entry(component.to_string())
            .or_insert_with(|| {
                let budget = self.config.component_budgets.get(component).copied();
                ComponentMemory::new(component.to_string(), budget)
            });

        // Get values we need before mutating
        let old_usage = comp.current_usage;
        let comp_budget = comp.budget;

        // Update component stats
        comp.current_usage = bytes;
        comp.peak_usage = comp.peak_usage.max(bytes);
        comp.allocation_count += 1;
        comp.last_updated = Some(now);

        // Update total
        let old_total = state.total_usage;
        state.total_usage = old_total - old_usage + bytes;
        state.peak_total_usage = state.peak_total_usage.max(state.total_usage);

        // Check budgets
        if let Some(budget) = comp_budget {
            if bytes > budget {
                return Err(MemoryMonitorError::BudgetExceeded(bytes - budget));
            }
        }

        if let Some(total_budget) = self.config.total_budget {
            if state.total_usage > total_budget {
                return Err(MemoryMonitorError::BudgetExceeded(
                    state.total_usage - total_budget,
                ));
            }
        }

        // Record sample for leak detection
        if self.config.enable_leak_detection {
            let total_usage = state.total_usage;
            state.memory_samples.push((now, total_usage));
            // Keep last 100 samples
            if state.memory_samples.len() > 100 {
                state.memory_samples.remove(0);
            }
        }

        Ok(())
    }

    /// Get current memory usage for a component
    pub fn get_usage(&self, component: &str) -> Result<usize, MemoryMonitorError> {
        let state = self.state.read();
        state
            .components
            .get(component)
            .map(|c| c.current_usage)
            .ok_or_else(|| MemoryMonitorError::ComponentNotFound(component.to_string()))
    }

    /// Get total memory usage
    pub fn total_usage(&self) -> usize {
        self.state.read().total_usage
    }

    /// Check if cleanup is needed
    pub fn needs_cleanup(&self) -> bool {
        if !self.config.enable_auto_cleanup {
            return false;
        }

        let state = self.state.read();

        if let Some(total_budget) = self.config.total_budget {
            let usage_ratio = state.total_usage as f64 / total_budget as f64;
            if usage_ratio >= self.config.cleanup_threshold {
                return true;
            }
        }

        // Check component budgets
        for comp in state.components.values() {
            if let Some(util) = comp.budget_utilization() {
                if util >= self.config.cleanup_threshold {
                    return true;
                }
            }
        }

        false
    }

    /// Detect memory leaks based on growth rate
    pub fn detect_leak(&self) -> Option<f64> {
        if !self.config.enable_leak_detection {
            return None;
        }

        let state = self.state.read();

        if state.memory_samples.len() < 10 {
            return None; // Not enough data
        }

        // Calculate growth rate (linear regression)
        let samples = &state.memory_samples;
        let n = samples.len();
        let first = &samples[0];
        let last = &samples[n - 1];

        let time_diff = last.0.duration_since(first.0).as_secs_f64();
        if time_diff < 1.0 {
            return None;
        }

        let growth = (last.1 as i64 - first.1 as i64) as f64;
        let growth_rate = growth / time_diff;

        if growth_rate.abs() > self.config.leak_detection_threshold {
            Some(growth_rate)
        } else {
            None
        }
    }

    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        let state = self.state.read();

        let components: Vec<ComponentMemory> = state.components.values().cloned().collect();

        MemoryStats {
            total_usage: state.total_usage,
            peak_usage: state.peak_total_usage,
            total_budget: self.config.total_budget,
            components,
            cleanup_count: state.cleanup_count,
            potential_leak: self.detect_leak(),
        }
    }

    /// Mark that cleanup was performed
    pub fn mark_cleanup(&self) {
        let mut state = self.state.write();
        state.last_cleanup = Instant::now();
        state.cleanup_count += 1;
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        let mut state = self.state.write();
        for comp in state.components.values_mut() {
            comp.peak_usage = comp.current_usage;
            comp.allocation_count = 0;
        }
        state.peak_total_usage = state.total_usage;
        state.memory_samples.clear();
    }

    /// Get component names
    pub fn component_names(&self) -> Vec<String> {
        self.state.read().components.keys().cloned().collect()
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total memory usage
    pub total_usage: usize,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Total memory budget
    pub total_budget: Option<usize>,
    /// Per-component stats
    pub components: Vec<ComponentMemory>,
    /// Number of cleanups performed
    pub cleanup_count: u64,
    /// Potential memory leak (bytes per second)
    pub potential_leak: Option<f64>,
}

impl MemoryStats {
    /// Get budget utilization (0.0-1.0+)
    pub fn budget_utilization(&self) -> Option<f64> {
        self.total_budget
            .map(|budget| self.total_usage as f64 / budget as f64)
    }

    /// Check if any component is over budget
    pub fn has_budget_violation(&self) -> bool {
        if let Some(budget) = self.total_budget {
            if self.total_usage > budget {
                return true;
            }
        }

        self.components.iter().any(|c| c.is_over_budget())
    }

    /// Format memory size as human-readable string
    pub fn format_bytes(bytes: usize) -> String {
        const KB: usize = 1024;
        const MB: usize = KB * 1024;
        const GB: usize = MB * 1024;

        if bytes >= GB {
            format!("{:.2} GB", bytes as f64 / GB as f64)
        } else if bytes >= MB {
            format!("{:.2} MB", bytes as f64 / MB as f64)
        } else if bytes >= KB {
            format!("{:.2} KB", bytes as f64 / KB as f64)
        } else {
            format!("{} B", bytes)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = MemoryMonitorConfig::default();
        assert!(config.validate().is_ok());
        assert!(config.enabled);
    }

    #[test]
    fn test_config_low_memory() {
        let config = MemoryMonitorConfig::low_memory();
        assert!(config.validate().is_ok());
        assert_eq!(config.total_budget, Some(128 * 1024 * 1024));
    }

    #[test]
    fn test_config_iot() {
        let config = MemoryMonitorConfig::iot();
        assert!(config.validate().is_ok());
        assert_eq!(config.total_budget, Some(64 * 1024 * 1024));
    }

    #[test]
    fn test_config_mobile() {
        let config = MemoryMonitorConfig::mobile();
        assert!(config.validate().is_ok());
        assert_eq!(config.total_budget, Some(256 * 1024 * 1024));
    }

    #[test]
    fn test_record_usage() {
        let config = MemoryMonitorConfig::default();
        let monitor = MemoryMonitor::new(config)
            .expect("test: MemoryMonitor::new should succeed with default config");

        let result = monitor.record_usage("test", 1000);
        assert!(result.is_ok());

        assert_eq!(
            monitor
                .get_usage("test")
                .expect("test: get_usage should return recorded value"),
            1000
        );
        assert_eq!(monitor.total_usage(), 1000);
    }

    #[test]
    fn test_budget_exceeded() {
        let mut config = MemoryMonitorConfig::default();
        config.component_budgets.insert("test".to_string(), 500);
        let monitor = MemoryMonitor::new(config)
            .expect("test: MemoryMonitor::new should succeed with component budget config");

        let result = monitor.record_usage("test", 1000);
        assert!(matches!(result, Err(MemoryMonitorError::BudgetExceeded(_))));
    }

    #[test]
    fn test_total_budget_exceeded() {
        let config = MemoryMonitorConfig {
            total_budget: Some(1000),
            ..Default::default()
        };
        let monitor = MemoryMonitor::new(config)
            .expect("test: MemoryMonitor::new should succeed with total_budget config");

        monitor
            .record_usage("test1", 500)
            .expect("test: record_usage test1 500 should succeed within budget");
        let result = monitor.record_usage("test2", 600);
        assert!(matches!(result, Err(MemoryMonitorError::BudgetExceeded(_))));
    }

    #[test]
    fn test_needs_cleanup() {
        let config = MemoryMonitorConfig {
            total_budget: Some(1000),
            cleanup_threshold: 0.8,
            ..Default::default()
        };
        let monitor = MemoryMonitor::new(config)
            .expect("test: MemoryMonitor::new should succeed with cleanup threshold config");

        assert!(!monitor.needs_cleanup());

        monitor
            .record_usage("test", 850)
            .expect("test: record_usage test 850 should succeed (under budget)");
        assert!(monitor.needs_cleanup());
    }

    #[test]
    fn test_component_utilization() {
        let mut comp = ComponentMemory::new("test".to_string(), Some(1000));
        comp.current_usage = 500;

        assert_eq!(comp.budget_utilization(), Some(0.5));
        assert!(!comp.is_over_budget());

        comp.current_usage = 1500;
        assert!(comp.is_over_budget());
    }

    #[test]
    fn test_stats() {
        let config = MemoryMonitorConfig::default();
        let monitor = MemoryMonitor::new(config)
            .expect("test: MemoryMonitor::new should succeed with default config in test_stats");

        monitor
            .record_usage("test1", 500)
            .expect("test: record_usage test1 500 should succeed in test_stats");
        monitor
            .record_usage("test2", 300)
            .expect("test: record_usage test2 300 should succeed in test_stats");

        let stats = monitor.stats();
        assert_eq!(stats.total_usage, 800);
        assert_eq!(stats.components.len(), 2);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(MemoryStats::format_bytes(500), "500 B");
        assert_eq!(MemoryStats::format_bytes(2048), "2.00 KB");
        assert_eq!(MemoryStats::format_bytes(2 * 1024 * 1024), "2.00 MB");
        assert_eq!(MemoryStats::format_bytes(3 * 1024 * 1024 * 1024), "3.00 GB");
    }

    #[test]
    fn test_component_names() {
        let config = MemoryMonitorConfig::low_memory();
        let monitor = MemoryMonitor::new(config)
            .expect("test: MemoryMonitor::new should succeed with low_memory config");

        let names = monitor.component_names();
        assert!(names.contains(&"peer_store".to_string()));
        assert!(names.contains(&"dht_cache".to_string()));
    }

    #[test]
    fn test_reset_stats() {
        let config = MemoryMonitorConfig::default();
        let monitor = MemoryMonitor::new(config).expect(
            "test: MemoryMonitor::new should succeed with default config in test_reset_stats",
        );

        monitor
            .record_usage("test", 1000)
            .expect("test: record_usage test 1000 should succeed in test_reset_stats");
        let stats1 = monitor.stats();
        assert_eq!(stats1.peak_usage, 1000);

        monitor.reset_stats();
        let stats2 = monitor.stats();
        assert_eq!(stats2.peak_usage, 1000); // Current usage, not reset
    }

    #[test]
    fn test_mark_cleanup() {
        let config = MemoryMonitorConfig::default();
        let monitor = MemoryMonitor::new(config).expect(
            "test: MemoryMonitor::new should succeed with default config in test_mark_cleanup",
        );

        let stats1 = monitor.stats();
        assert_eq!(stats1.cleanup_count, 0);

        monitor.mark_cleanup();
        let stats2 = monitor.stats();
        assert_eq!(stats2.cleanup_count, 1);
    }
}
