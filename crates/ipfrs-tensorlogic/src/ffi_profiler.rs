//! FFI Overhead Profiling
//!
//! This module provides utilities for profiling FFI call overhead and identifying
//! performance bottlenecks in cross-language boundaries.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// FFI call statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiCallStats {
    /// Function name
    pub name: String,
    /// Total number of calls
    pub call_count: u64,
    /// Total time spent in calls
    pub total_duration: Duration,
    /// Minimum call duration
    pub min_duration: Duration,
    /// Maximum call duration
    pub max_duration: Duration,
    /// Average call duration
    pub avg_duration: Duration,
}

impl FfiCallStats {
    fn new(name: String) -> Self {
        Self {
            name,
            call_count: 0,
            total_duration: Duration::ZERO,
            min_duration: Duration::MAX,
            max_duration: Duration::ZERO,
            avg_duration: Duration::ZERO,
        }
    }

    fn record(&mut self, duration: Duration) {
        self.call_count += 1;
        self.total_duration += duration;
        self.min_duration = self.min_duration.min(duration);
        self.max_duration = self.max_duration.max(duration);
        self.avg_duration = self.total_duration / self.call_count as u32;
    }

    /// Check if call overhead exceeds target
    pub fn exceeds_target(&self, target_micros: u64) -> bool {
        self.avg_duration.as_micros() > target_micros as u128
    }

    /// Get overhead percentage relative to target
    pub fn overhead_percentage(&self, target_micros: u64) -> f64 {
        let avg_micros = self.avg_duration.as_micros() as f64;
        ((avg_micros - target_micros as f64) / target_micros as f64) * 100.0
    }
}

/// FFI profiler for measuring call overhead
pub struct FfiProfiler {
    stats: Arc<RwLock<HashMap<String, FfiCallStats>>>,
    enabled: Arc<RwLock<bool>>,
}

impl FfiProfiler {
    /// Create a new FFI profiler
    pub fn new() -> Self {
        Self {
            stats: Arc::new(RwLock::new(HashMap::new())),
            enabled: Arc::new(RwLock::new(true)),
        }
    }

    /// Enable profiling
    pub fn enable(&self) {
        *self.enabled.write() = true;
    }

    /// Disable profiling
    pub fn disable(&self) {
        *self.enabled.write() = false;
    }

    /// Check if profiling is enabled
    pub fn is_enabled(&self) -> bool {
        *self.enabled.read()
    }

    /// Start profiling a function call
    pub fn start(&self, name: &str) -> FfiCallGuard {
        FfiCallGuard {
            name: name.to_string(),
            start: Instant::now(),
            profiler: self.clone(),
        }
    }

    /// Record a call duration
    fn record(&self, name: String, duration: Duration) {
        if !self.is_enabled() {
            return;
        }

        let mut stats = self.stats.write();
        stats
            .entry(name.clone())
            .or_insert_with(|| FfiCallStats::new(name))
            .record(duration);
    }

    /// Get statistics for a specific function
    pub fn get_stats(&self, name: &str) -> Option<FfiCallStats> {
        self.stats.read().get(name).cloned()
    }

    /// Get all statistics
    pub fn get_all_stats(&self) -> Vec<FfiCallStats> {
        self.stats.read().values().cloned().collect()
    }

    /// Reset all statistics
    pub fn reset(&self) {
        self.stats.write().clear();
    }

    /// Get statistics sorted by average duration
    pub fn get_hotspots(&self) -> Vec<FfiCallStats> {
        let mut stats = self.get_all_stats();
        stats.sort_by_key(|s| std::cmp::Reverse(s.avg_duration));
        stats
    }

    /// Get total overhead
    pub fn total_overhead(&self) -> Duration {
        self.stats.read().values().map(|s| s.total_duration).sum()
    }

    /// Generate profiling report
    pub fn report(&self) -> ProfilingReport {
        let stats = self.get_all_stats();
        let total_calls: u64 = stats.iter().map(|s| s.call_count).sum();
        let total_duration = self.total_overhead();

        ProfilingReport {
            total_calls,
            total_duration,
            function_stats: stats,
        }
    }
}

impl Default for FfiProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for FfiProfiler {
    fn clone(&self) -> Self {
        Self {
            stats: Arc::clone(&self.stats),
            enabled: Arc::clone(&self.enabled),
        }
    }
}

/// RAII guard for profiling FFI calls
pub struct FfiCallGuard {
    name: String,
    start: Instant,
    profiler: FfiProfiler,
}

impl Drop for FfiCallGuard {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        self.profiler.record(self.name.clone(), duration);
    }
}

/// Profiling report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingReport {
    /// Total number of FFI calls
    pub total_calls: u64,
    /// Total time spent in FFI calls
    pub total_duration: Duration,
    /// Per-function statistics
    pub function_stats: Vec<FfiCallStats>,
}

impl ProfilingReport {
    /// Print report to stdout
    pub fn print(&self) {
        println!("\n=== FFI Profiling Report ===");
        println!("Total calls: {}", self.total_calls);
        println!("Total duration: {:?}", self.total_duration);
        println!("\nFunction statistics:");
        println!(
            "{:<30} {:>10} {:>15} {:>15} {:>15}",
            "Function", "Calls", "Avg (μs)", "Min (μs)", "Max (μs)"
        );
        println!("{}", "-".repeat(85));

        let mut sorted_stats = self.function_stats.clone();
        sorted_stats.sort_by_key(|s| std::cmp::Reverse(s.avg_duration));

        for stat in sorted_stats {
            println!(
                "{:<30} {:>10} {:>15.2} {:>15.2} {:>15.2}",
                stat.name,
                stat.call_count,
                stat.avg_duration.as_micros() as f64,
                stat.min_duration.as_micros() as f64,
                stat.max_duration.as_micros() as f64,
            );
        }
    }

    /// Identify functions exceeding target overhead
    pub fn identify_bottlenecks(&self, target_micros: u64) -> Vec<String> {
        self.function_stats
            .iter()
            .filter(|s| s.exceeds_target(target_micros))
            .map(|s| s.name.clone())
            .collect()
    }

    /// Get overhead summary
    pub fn summary(&self) -> OverheadSummary {
        let avg_call_duration = if self.total_calls > 0 {
            self.total_duration / self.total_calls as u32
        } else {
            Duration::ZERO
        };

        let max_duration = self
            .function_stats
            .iter()
            .map(|s| s.max_duration)
            .max()
            .unwrap_or(Duration::ZERO);

        OverheadSummary {
            total_calls: self.total_calls,
            total_duration: self.total_duration,
            avg_call_duration,
            max_call_duration: max_duration,
            functions_profiled: self.function_stats.len(),
        }
    }
}

/// Overhead summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverheadSummary {
    pub total_calls: u64,
    pub total_duration: Duration,
    pub avg_call_duration: Duration,
    pub max_call_duration: Duration,
    pub functions_profiled: usize,
}

impl OverheadSummary {
    /// Check if average overhead meets target
    pub fn meets_target(&self, target_micros: u64) -> bool {
        self.avg_call_duration.as_micros() <= target_micros as u128
    }
}

/// Global FFI profiler instance
static GLOBAL_PROFILER: once_cell::sync::Lazy<FfiProfiler> =
    once_cell::sync::Lazy::new(FfiProfiler::new);

/// Get the global FFI profiler
pub fn global_profiler() -> &'static FfiProfiler {
    &GLOBAL_PROFILER
}

/// Profile an FFI function call
#[macro_export]
macro_rules! profile_ffi {
    ($name:expr, $body:expr) => {{
        let _guard = $crate::ffi_profiler::global_profiler().start($name);
        $body
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_ffi_profiler_basic() {
        let profiler = FfiProfiler::new();

        // Profile a function
        {
            let _guard = profiler.start("test_function");
            thread::sleep(Duration::from_millis(10));
        }

        let stats = profiler.get_stats("test_function");
        assert!(stats.is_some());

        let stats = stats.expect("test: should succeed");
        assert_eq!(stats.call_count, 1);
        assert!(stats.avg_duration >= Duration::from_millis(10));
    }

    #[test]
    fn test_multiple_calls() {
        let profiler = FfiProfiler::new();

        for _ in 0..5 {
            let _guard = profiler.start("multi_call");
            thread::sleep(Duration::from_millis(5));
        }

        let stats = profiler
            .get_stats("multi_call")
            .expect("test: should succeed");
        assert_eq!(stats.call_count, 5);
        assert!(stats.avg_duration >= Duration::from_millis(5));
    }

    #[test]
    fn test_enable_disable() {
        let profiler = FfiProfiler::new();

        profiler.disable();
        {
            let _guard = profiler.start("disabled");
            thread::sleep(Duration::from_millis(5));
        }

        assert!(profiler.get_stats("disabled").is_none());

        profiler.enable();
        {
            let _guard = profiler.start("enabled");
            thread::sleep(Duration::from_millis(5));
        }

        assert!(profiler.get_stats("enabled").is_some());
    }

    #[test]
    fn test_reset() {
        let profiler = FfiProfiler::new();

        {
            let _guard = profiler.start("test");
            thread::sleep(Duration::from_millis(5));
        }

        assert!(profiler.get_stats("test").is_some());

        profiler.reset();
        assert!(profiler.get_stats("test").is_none());
    }

    #[test]
    fn test_hotspots() {
        let profiler = FfiProfiler::new();

        {
            let _guard = profiler.start("fast");
            thread::sleep(Duration::from_millis(1));
        }

        {
            let _guard = profiler.start("slow");
            thread::sleep(Duration::from_millis(10));
        }

        let hotspots = profiler.get_hotspots();
        assert_eq!(hotspots.len(), 2);
        assert_eq!(hotspots[0].name, "slow");
        assert_eq!(hotspots[1].name, "fast");
    }

    #[test]
    fn test_profiling_report() {
        let profiler = FfiProfiler::new();

        for i in 0..3 {
            let _guard = profiler.start(&format!("func_{}", i));
            thread::sleep(Duration::from_millis(5));
        }

        let report = profiler.report();
        assert_eq!(report.total_calls, 3);
        assert_eq!(report.function_stats.len(), 3);

        let summary = report.summary();
        assert_eq!(summary.total_calls, 3);
        assert_eq!(summary.functions_profiled, 3);
    }

    #[test]
    fn test_exceeds_target() {
        let mut stats = FfiCallStats::new("test".to_string());
        stats.record(Duration::from_micros(500));

        assert!(!stats.exceeds_target(1000));
        assert!(stats.exceeds_target(100));
    }

    #[test]
    fn test_identify_bottlenecks() {
        let profiler = FfiProfiler::new();

        // Use manually injected durations rather than sleeping, so the test
        // is not sensitive to OS scheduling jitter.
        {
            let mut stats = profiler.stats.write();
            let mut fast_stat = FfiCallStats::new("fast".to_string());
            fast_stat.record(Duration::from_micros(100));
            stats.insert("fast".to_string(), fast_stat);

            let mut slow_stat = FfiCallStats::new("slow".to_string());
            slow_stat.record(Duration::from_millis(10));
            stats.insert("slow".to_string(), slow_stat);
        }

        let report = profiler.report();
        // Target: 1ms (1000 µs). "fast" avg=100µs, "slow" avg=10000µs.
        let bottlenecks = report.identify_bottlenecks(1000);

        assert!(bottlenecks.contains(&"slow".to_string()));
        assert!(!bottlenecks.contains(&"fast".to_string()));
    }

    #[test]
    fn test_global_profiler() {
        let profiler = global_profiler();

        profiler.reset(); // Clear any previous stats

        {
            let _guard = profiler.start("global_test");
            thread::sleep(Duration::from_millis(5));
        }

        let stats = profiler.get_stats("global_test");
        assert!(stats.is_some());
    }
}
