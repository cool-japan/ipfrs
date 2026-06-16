//! Performance Profiling Utilities
//!
//! Tools for measuring and analyzing IPFRS performance.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Operation profiler for tracking performance
#[derive(Debug, Clone)]
pub struct Profiler {
    measurements: Arc<Mutex<HashMap<String, Vec<Duration>>>>,
}

impl Profiler {
    /// Create a new profiler
    pub fn new() -> Self {
        Self {
            measurements: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Start timing an operation
    pub fn start(&self, operation: impl Into<String>) -> ProfilerGuard {
        ProfilerGuard {
            operation: operation.into(),
            start: Instant::now(),
            profiler: self.clone(),
        }
    }

    /// Record a duration for an operation
    fn record(&self, operation: String, duration: Duration) {
        let mut measurements = self.measurements.lock().unwrap_or_else(|e| e.into_inner());
        measurements.entry(operation).or_default().push(duration);
    }

    /// Get statistics for an operation
    pub fn stats(&self, operation: &str) -> Option<OperationStats> {
        let measurements = self.measurements.lock().unwrap_or_else(|e| e.into_inner());
        let durations = measurements.get(operation)?;

        if durations.is_empty() {
            return None;
        }

        let total: Duration = durations.iter().sum();
        let count = durations.len();
        let avg = total / count as u32;

        let mut sorted = durations.clone();
        sorted.sort();

        let min = sorted.first().copied()?;
        let max = sorted.last().copied()?;
        let median = sorted[count / 2];
        let p95 = sorted[(count as f64 * 0.95) as usize];
        let p99 = sorted[(count as f64 * 0.99) as usize];

        Some(OperationStats {
            count,
            total,
            avg,
            min,
            max,
            median,
            p95,
            p99,
        })
    }

    /// Get all recorded operations
    pub fn operations(&self) -> Vec<String> {
        let measurements = self.measurements.lock().unwrap_or_else(|e| e.into_inner());
        measurements.keys().cloned().collect()
    }

    /// Generate a report of all operations
    pub fn report(&self) -> String {
        let mut output = String::from("Performance Report\n");
        output.push_str("==================\n\n");

        let operations = self.operations();
        if operations.is_empty() {
            output.push_str("No operations recorded.\n");
            return output;
        }

        for op in operations {
            if let Some(stats) = self.stats(&op) {
                output.push_str(&format!("{}\n", op));
                output.push_str(&format!("  Count:  {}\n", stats.count));
                output.push_str(&format!("  Total:  {:?}\n", stats.total));
                output.push_str(&format!("  Avg:    {:?}\n", stats.avg));
                output.push_str(&format!("  Min:    {:?}\n", stats.min));
                output.push_str(&format!("  Max:    {:?}\n", stats.max));
                output.push_str(&format!("  Median: {:?}\n", stats.median));
                output.push_str(&format!("  P95:    {:?}\n", stats.p95));
                output.push_str(&format!("  P99:    {:?}\n", stats.p99));
                output.push('\n');
            }
        }

        output
    }

    /// Clear all measurements
    pub fn clear(&self) {
        let mut measurements = self.measurements.lock().unwrap_or_else(|e| e.into_inner());
        measurements.clear();
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Guard that automatically records duration when dropped
pub struct ProfilerGuard {
    operation: String,
    start: Instant,
    profiler: Profiler,
}

impl Drop for ProfilerGuard {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        self.profiler.record(self.operation.clone(), duration);
    }
}

/// Statistics for an operation
#[derive(Debug, Clone)]
pub struct OperationStats {
    /// Number of measurements
    pub count: usize,
    /// Total time
    pub total: Duration,
    /// Average time
    pub avg: Duration,
    /// Minimum time
    pub min: Duration,
    /// Maximum time
    pub max: Duration,
    /// Median time
    pub median: Duration,
    /// 95th percentile
    pub p95: Duration,
    /// 99th percentile
    pub p99: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_profiler_basic() {
        let profiler = Profiler::new();

        // Profile some operations
        {
            let _guard = profiler.start("test_op");
            thread::sleep(Duration::from_millis(10));
        }

        {
            let _guard = profiler.start("test_op");
            thread::sleep(Duration::from_millis(20));
        }

        // Check stats
        let stats = profiler
            .stats("test_op")
            .expect("test: stats for test_op should exist");
        assert_eq!(stats.count, 2);
        assert!(stats.total >= Duration::from_millis(30));
        assert!(stats.avg >= Duration::from_millis(15));
    }

    #[test]
    fn test_profiler_multiple_operations() {
        let profiler = Profiler::new();

        {
            let _guard = profiler.start("op1");
            thread::sleep(Duration::from_millis(5));
        }

        {
            let _guard = profiler.start("op2");
            thread::sleep(Duration::from_millis(10));
        }

        let ops = profiler.operations();
        assert_eq!(ops.len(), 2);
        assert!(ops.contains(&"op1".to_string()));
        assert!(ops.contains(&"op2".to_string()));
    }

    #[test]
    fn test_profiler_report() {
        let profiler = Profiler::new();

        {
            let _guard = profiler.start("test");
            thread::sleep(Duration::from_millis(5));
        }

        let report = profiler.report();
        assert!(report.contains("Performance Report"));
        assert!(report.contains("test"));
        assert!(report.contains("Count:"));
    }

    #[test]
    fn test_profiler_clear() {
        let profiler = Profiler::new();

        {
            let _guard = profiler.start("test");
            thread::sleep(Duration::from_millis(5));
        }

        assert!(profiler.stats("test").is_some());

        profiler.clear();
        assert!(profiler.stats("test").is_none());
    }
}
