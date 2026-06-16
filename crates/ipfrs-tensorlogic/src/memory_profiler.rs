//! Memory profiling utilities for tracking allocations and memory usage.
//!
//! This module provides tools for:
//! - Tracking heap allocations
//! - Monitoring shared memory usage
//! - Detecting potential memory leaks
//! - Measuring peak memory consumption
//!
//! # Examples
//!
//! ```
//! use ipfrs_tensorlogic::MemoryProfiler;
//!
//! let profiler = MemoryProfiler::new();
//!
//! {
//!     let _guard = profiler.start_tracking("my_operation");
//!     // Your operation here
//!     let data = vec![0u8; 1024 * 1024]; // 1 MB allocation
//!     drop(data);
//! }
//!
//! let stats = profiler.get_stats("my_operation").expect("example: should succeed in docs");
//! println!("Peak memory: {} bytes", stats.peak_bytes);
//! ```

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Memory usage statistics for a tracked operation
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Number of times this operation was tracked
    pub track_count: usize,
    /// Total bytes allocated (cumulative across all tracks)
    pub total_bytes: usize,
    /// Peak bytes used during any single track
    pub peak_bytes: usize,
    /// Average bytes per track
    pub avg_bytes: usize,
    /// Total duration tracked
    pub total_duration: Duration,
    /// Average duration per track
    pub avg_duration: Duration,
}

impl MemoryStats {
    fn new() -> Self {
        Self {
            track_count: 0,
            total_bytes: 0,
            peak_bytes: 0,
            avg_bytes: 0,
            total_duration: Duration::ZERO,
            avg_duration: Duration::ZERO,
        }
    }

    fn update(&mut self, bytes: usize, duration: Duration) {
        self.track_count += 1;
        self.total_bytes += bytes;
        self.peak_bytes = self.peak_bytes.max(bytes);
        self.total_duration += duration;

        self.avg_bytes = self.total_bytes.checked_div(self.track_count).unwrap_or(0);
        self.avg_duration = self
            .total_duration
            .checked_div(self.track_count as u32)
            .unwrap_or(Duration::ZERO);
    }
}

/// A guard that tracks memory usage for the duration of its lifetime
pub struct MemoryTrackingGuard {
    profiler: Arc<MemoryProfiler>,
    operation: String,
    start_time: Instant,
    initial_memory: usize,
}

impl Drop for MemoryTrackingGuard {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed();
        let current_memory = self.profiler.get_current_memory_usage();
        let bytes_used = current_memory.saturating_sub(self.initial_memory);

        let mut stats = self.profiler.stats.write();
        let entry = stats
            .entry(self.operation.clone())
            .or_insert_with(MemoryStats::new);
        entry.update(bytes_used, duration);
    }
}

/// Memory profiler for tracking allocations and usage
pub struct MemoryProfiler {
    stats: Arc<RwLock<HashMap<String, MemoryStats>>>,
}

impl MemoryProfiler {
    /// Create a new memory profiler
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            stats: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Start tracking memory usage for an operation
    ///
    /// Returns a guard that will record statistics when dropped.
    pub fn start_tracking(self: &Arc<Self>, operation: &str) -> MemoryTrackingGuard {
        MemoryTrackingGuard {
            profiler: Arc::clone(self),
            operation: operation.to_string(),
            start_time: Instant::now(),
            initial_memory: self.get_current_memory_usage(),
        }
    }

    /// Get statistics for a specific operation
    pub fn get_stats(&self, operation: &str) -> Option<MemoryStats> {
        self.stats.read().get(operation).cloned()
    }

    /// Get all tracked statistics
    pub fn get_all_stats(&self) -> HashMap<String, MemoryStats> {
        self.stats.read().clone()
    }

    /// Clear all statistics
    pub fn clear(&self) {
        self.stats.write().clear();
    }

    /// Get current memory usage in bytes
    ///
    /// This is a platform-specific approximation based on available system information.
    #[cfg(target_os = "linux")]
    fn get_current_memory_usage(&self) -> usize {
        // On Linux, read from /proc/self/statm
        if let Ok(contents) = std::fs::read_to_string("/proc/self/statm") {
            if let Some(first) = contents.split_whitespace().next() {
                if let Ok(pages) = first.parse::<usize>() {
                    // Each page is typically 4096 bytes
                    return pages * 4096;
                }
            }
        }
        0
    }

    #[cfg(not(target_os = "linux"))]
    fn get_current_memory_usage(&self) -> usize {
        // For non-Linux systems, we can't easily get RSS without platform-specific code
        // Return 0 as a placeholder
        0
    }

    /// Generate a memory profiling report
    pub fn generate_report(&self) -> MemoryProfilingReport {
        let stats = self.get_all_stats();
        let total_operations = stats.len();
        let total_tracked = stats.values().map(|s| s.track_count).sum();
        let total_bytes: usize = stats.values().map(|s| s.total_bytes).sum();
        let max_peak = stats.values().map(|s| s.peak_bytes).max().unwrap_or(0);

        let mut operations: Vec<_> = stats.into_iter().collect();
        operations.sort_by_key(|a| std::cmp::Reverse(a.1.peak_bytes));

        MemoryProfilingReport {
            total_operations,
            total_tracked,
            total_bytes,
            max_peak_bytes: max_peak,
            operations,
        }
    }
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self {
            stats: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

/// A comprehensive memory profiling report
#[derive(Debug)]
pub struct MemoryProfilingReport {
    /// Total number of distinct operations tracked
    pub total_operations: usize,
    /// Total number of tracking instances
    pub total_tracked: usize,
    /// Total bytes allocated across all operations
    pub total_bytes: usize,
    /// Maximum peak memory usage across all operations
    pub max_peak_bytes: usize,
    /// Operations sorted by peak memory usage (descending)
    pub operations: Vec<(String, MemoryStats)>,
}

impl MemoryProfilingReport {
    /// Print a formatted report to stdout
    pub fn print(&self) {
        println!("=== Memory Profiling Report ===");
        println!("Total operations: {}", self.total_operations);
        println!("Total tracks: {}", self.total_tracked);
        println!(
            "Total bytes: {} ({:.2} MB)",
            self.total_bytes,
            self.total_bytes as f64 / 1024.0 / 1024.0
        );
        println!(
            "Max peak: {} ({:.2} MB)",
            self.max_peak_bytes,
            self.max_peak_bytes as f64 / 1024.0 / 1024.0
        );
        println!("\nTop memory-consuming operations:");
        println!(
            "{:<40} {:>12} {:>12} {:>12} {:>10}",
            "Operation", "Tracks", "Peak", "Avg", "Avg Time"
        );
        println!(
            "{:-<40} {:-<12} {:-<12} {:-<12} {:-<10}",
            "", "", "", "", ""
        );

        for (i, (name, stats)) in self.operations.iter().enumerate().take(10) {
            println!(
                "{:<40} {:>12} {:>12} {:>12} {:>10?}",
                if name.len() > 40 {
                    format!("{}...", &name[..37])
                } else {
                    name.clone()
                },
                stats.track_count,
                format_bytes(stats.peak_bytes),
                format_bytes(stats.avg_bytes),
                stats.avg_duration
            );
            if i >= 9 {
                break;
            }
        }
    }
}

/// Format bytes in human-readable form
fn format_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / 1024.0 / 1024.0)
    } else {
        format!("{:.1} GB", bytes as f64 / 1024.0 / 1024.0 / 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_profiler_basic() {
        let profiler = MemoryProfiler::new();

        {
            let _guard = profiler.start_tracking("test_operation");
            // Simulate some work
            std::thread::sleep(Duration::from_millis(10));
        }

        let stats = profiler.get_stats("test_operation");
        assert!(stats.is_some());

        let stats = stats.expect("test: should succeed");
        assert_eq!(stats.track_count, 1);
        assert!(stats.total_duration >= Duration::from_millis(10));
    }

    #[test]
    fn test_memory_profiler_multiple_tracks() {
        let profiler = MemoryProfiler::new();

        for _ in 0..5 {
            let _guard = profiler.start_tracking("repeated_op");
            std::thread::sleep(Duration::from_millis(5));
        }

        let stats = profiler
            .get_stats("repeated_op")
            .expect("test: should succeed");
        assert_eq!(stats.track_count, 5);
        assert!(stats.avg_duration >= Duration::from_millis(5));
    }

    #[test]
    fn test_memory_profiler_multiple_operations() {
        let profiler = MemoryProfiler::new();

        {
            let _guard1 = profiler.start_tracking("op1");
            std::thread::sleep(Duration::from_millis(5));
        }

        {
            let _guard2 = profiler.start_tracking("op2");
            std::thread::sleep(Duration::from_millis(10));
        }

        let all_stats = profiler.get_all_stats();
        assert_eq!(all_stats.len(), 2);
        assert!(all_stats.contains_key("op1"));
        assert!(all_stats.contains_key("op2"));
    }

    #[test]
    fn test_memory_profiler_clear() {
        let profiler = MemoryProfiler::new();

        {
            let _guard = profiler.start_tracking("test");
        }

        assert_eq!(profiler.get_all_stats().len(), 1);

        profiler.clear();
        assert_eq!(profiler.get_all_stats().len(), 0);
    }

    #[test]
    fn test_memory_profiler_report() {
        let profiler = MemoryProfiler::new();

        {
            let _guard = profiler.start_tracking("op1");
        }

        {
            let _guard = profiler.start_tracking("op2");
        }

        let report = profiler.generate_report();
        assert_eq!(report.total_operations, 2);
        assert_eq!(report.total_tracked, 2);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
    }
}
