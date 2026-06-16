//! TensorProfiler — operation profiling for tensor computations.
//!
//! Records per-operation timing, element counts, and aggregates
//! into per-op-name profiles for throughput analysis and hotspot
//! identification.

use std::collections::HashMap;

/// A single recorded profiling entry.
#[derive(Debug, Clone)]
pub struct ProfileEntry {
    /// Name of the operation (e.g. "matmul", "conv2d").
    pub op_name: String,
    /// Wall-clock duration of the operation in nanoseconds.
    pub duration_ns: u64,
    /// Number of input elements consumed.
    pub input_elements: u64,
    /// Number of output elements produced.
    pub output_elements: u64,
    /// Monotonic tick at which the entry was recorded.
    pub tick: u64,
}

/// Aggregate profile for a single operation name.
#[derive(Debug, Clone)]
pub struct OpProfile {
    /// Operation name.
    pub op_name: String,
    /// Total number of calls recorded.
    pub call_count: u64,
    /// Cumulative wall-clock time in nanoseconds.
    pub total_ns: u64,
    /// Minimum single-call duration in nanoseconds.
    pub min_ns: u64,
    /// Maximum single-call duration in nanoseconds.
    pub max_ns: u64,
    /// Cumulative input element count.
    pub total_input_elements: u64,
    /// Cumulative output element count.
    pub total_output_elements: u64,
}

/// Summary statistics for the profiler.
#[derive(Debug, Clone)]
pub struct ProfilerStats {
    /// Total number of entries stored.
    pub total_entries: usize,
    /// Number of distinct operation names.
    pub unique_ops: usize,
    /// Sum of `duration_ns` across all entries.
    pub total_ns: u64,
    /// Whether the profiler is currently enabled.
    pub enabled: bool,
}

/// Operation profiler for tensor computations.
///
/// Records individual operation entries and maintains running
/// aggregates per operation name.  Supports enable/disable toggling,
/// configurable maximum entry count (oldest entries evicted on
/// overflow), and tick-based ordering.
pub struct TensorProfiler {
    entries: Vec<ProfileEntry>,
    op_profiles: HashMap<String, OpProfile>,
    enabled: bool,
    current_tick: u64,
    max_entries: usize,
}

impl TensorProfiler {
    /// Create a new profiler with the given maximum entry capacity.
    ///
    /// The profiler starts **enabled** by default.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            op_profiles: HashMap::new(),
            enabled: true,
            current_tick: 0,
            max_entries,
        }
    }

    /// Record an operation.
    ///
    /// If the profiler is disabled the call is a no-op.  When the
    /// entry buffer is full the oldest entry is evicted (FIFO).
    pub fn record(
        &mut self,
        op_name: &str,
        duration_ns: u64,
        input_elements: u64,
        output_elements: u64,
    ) {
        if !self.enabled {
            return;
        }

        // Only store in the entry buffer if capacity allows.
        if self.max_entries > 0 {
            // Evict oldest if at capacity.
            if self.entries.len() >= self.max_entries {
                self.entries.remove(0);
            }

            let entry = ProfileEntry {
                op_name: op_name.to_string(),
                duration_ns,
                input_elements,
                output_elements,
                tick: self.current_tick,
            };
            self.entries.push(entry);
        }

        // Update aggregate profile.
        let profile = self
            .op_profiles
            .entry(op_name.to_string())
            .or_insert_with(|| OpProfile {
                op_name: op_name.to_string(),
                call_count: 0,
                total_ns: 0,
                min_ns: u64::MAX,
                max_ns: 0,
                total_input_elements: 0,
                total_output_elements: 0,
            });

        profile.call_count += 1;
        profile.total_ns += duration_ns;
        if duration_ns < profile.min_ns {
            profile.min_ns = duration_ns;
        }
        if duration_ns > profile.max_ns {
            profile.max_ns = duration_ns;
        }
        profile.total_input_elements += input_elements;
        profile.total_output_elements += output_elements;
    }

    /// Enable recording.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable recording.  Subsequent `record` calls become no-ops.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Returns `true` if the profiler is currently recording.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Look up the aggregate profile for a given operation name.
    pub fn get_profile(&self, op_name: &str) -> Option<&OpProfile> {
        self.op_profiles.get(op_name)
    }

    /// Average duration per call in nanoseconds for the given op.
    pub fn avg_ns(&self, op_name: &str) -> Option<f64> {
        self.op_profiles.get(op_name).map(|p| {
            if p.call_count == 0 {
                0.0
            } else {
                p.total_ns as f64 / p.call_count as f64
            }
        })
    }

    /// Throughput in elements/second for the given op.
    ///
    /// Computed as `total_output_elements / (total_ns / 1e9)`.
    pub fn throughput(&self, op_name: &str) -> Option<f64> {
        self.op_profiles.get(op_name).map(|p| {
            if p.total_ns == 0 {
                0.0
            } else {
                p.total_output_elements as f64 / (p.total_ns as f64 / 1e9)
            }
        })
    }

    /// Return the top `n` operations by cumulative time (descending).
    pub fn hottest_ops(&self, n: usize) -> Vec<&OpProfile> {
        let mut profiles: Vec<&OpProfile> = self.op_profiles.values().collect();
        profiles.sort_by_key(|p| std::cmp::Reverse(p.total_ns));
        profiles.truncate(n);
        profiles
    }

    /// Advance the internal tick counter by one.
    pub fn tick(&mut self) {
        self.current_tick += 1;
    }

    /// Clear all entries and aggregate profiles.
    pub fn reset(&mut self) {
        self.entries.clear();
        self.op_profiles.clear();
        self.current_tick = 0;
    }

    /// Number of entries currently stored.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Return a snapshot of summary statistics.
    pub fn stats(&self) -> ProfilerStats {
        let total_ns: u64 = self.op_profiles.values().map(|p| p.total_ns).sum();
        ProfilerStats {
            total_entries: self.entries.len(),
            unique_ops: self.op_profiles.len(),
            total_ns,
            enabled: self.enabled,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_profiler_defaults() {
        let p = TensorProfiler::new(100);
        assert!(p.is_enabled());
        assert_eq!(p.entry_count(), 0);
        let s = p.stats();
        assert_eq!(s.total_entries, 0);
        assert_eq!(s.unique_ops, 0);
        assert_eq!(s.total_ns, 0);
        assert!(s.enabled);
    }

    #[test]
    fn test_record_updates_profile() {
        let mut p = TensorProfiler::new(100);
        p.record("matmul", 1000, 64, 32);
        let prof = p.get_profile("matmul").expect("profile should exist");
        assert_eq!(prof.call_count, 1);
        assert_eq!(prof.total_ns, 1000);
        assert_eq!(prof.min_ns, 1000);
        assert_eq!(prof.max_ns, 1000);
        assert_eq!(prof.total_input_elements, 64);
        assert_eq!(prof.total_output_elements, 32);
        assert_eq!(p.entry_count(), 1);
    }

    #[test]
    fn test_disabled_skips_recording() {
        let mut p = TensorProfiler::new(100);
        p.disable();
        p.record("matmul", 500, 10, 10);
        assert_eq!(p.entry_count(), 0);
        assert!(p.get_profile("matmul").is_none());
    }

    #[test]
    fn test_avg_ns_calculation() {
        let mut p = TensorProfiler::new(100);
        p.record("add", 100, 10, 10);
        p.record("add", 200, 10, 10);
        p.record("add", 300, 10, 10);
        let avg = p.avg_ns("add").expect("avg should exist");
        assert!((avg - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_avg_ns_missing_op() {
        let p = TensorProfiler::new(100);
        assert!(p.avg_ns("nope").is_none());
    }

    #[test]
    fn test_throughput_calculation() {
        let mut p = TensorProfiler::new(100);
        // 1_000_000_000 ns = 1 second, 500 output elements => 500 elem/s
        p.record("conv", 1_000_000_000, 1000, 500);
        let tp = p.throughput("conv").expect("throughput should exist");
        assert!((tp - 500.0).abs() < 1e-6);
    }

    #[test]
    fn test_throughput_zero_time() {
        let mut p = TensorProfiler::new(100);
        p.record("noop", 0, 0, 0);
        let tp = p.throughput("noop").expect("throughput should exist");
        assert!((tp - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_throughput_missing_op() {
        let p = TensorProfiler::new(100);
        assert!(p.throughput("nope").is_none());
    }

    #[test]
    fn test_hottest_ops_ordering() {
        let mut p = TensorProfiler::new(1000);
        p.record("fast", 100, 1, 1);
        p.record("slow", 5000, 1, 1);
        p.record("mid", 2000, 1, 1);

        let hot = p.hottest_ops(3);
        assert_eq!(hot.len(), 3);
        assert_eq!(hot[0].op_name, "slow");
        assert_eq!(hot[1].op_name, "mid");
        assert_eq!(hot[2].op_name, "fast");
    }

    #[test]
    fn test_hottest_ops_fewer_than_n() {
        let mut p = TensorProfiler::new(100);
        p.record("only", 100, 1, 1);
        let hot = p.hottest_ops(10);
        assert_eq!(hot.len(), 1);
    }

    #[test]
    fn test_max_entries_eviction() {
        let mut p = TensorProfiler::new(3);
        p.record("a", 10, 1, 1);
        p.record("b", 20, 1, 1);
        p.record("c", 30, 1, 1);
        assert_eq!(p.entry_count(), 3);

        // Fourth entry evicts the oldest ("a")
        p.record("d", 40, 1, 1);
        assert_eq!(p.entry_count(), 3);

        // The aggregate profiles still include all four ops.
        assert!(p.get_profile("a").is_some());
        assert!(p.get_profile("d").is_some());
    }

    #[test]
    fn test_reset_clears_all() {
        let mut p = TensorProfiler::new(100);
        p.record("x", 100, 10, 10);
        p.tick();
        p.reset();
        assert_eq!(p.entry_count(), 0);
        assert!(p.get_profile("x").is_none());
        let s = p.stats();
        assert_eq!(s.total_entries, 0);
        assert_eq!(s.unique_ops, 0);
        assert_eq!(s.total_ns, 0);
    }

    #[test]
    fn test_multiple_ops_tracked_independently() {
        let mut p = TensorProfiler::new(100);
        p.record("matmul", 1000, 64, 32);
        p.record("relu", 200, 32, 32);
        p.record("matmul", 1500, 64, 32);

        let mm = p.get_profile("matmul").expect("matmul profile");
        assert_eq!(mm.call_count, 2);
        assert_eq!(mm.total_ns, 2500);

        let relu = p.get_profile("relu").expect("relu profile");
        assert_eq!(relu.call_count, 1);
        assert_eq!(relu.total_ns, 200);
    }

    #[test]
    fn test_min_max_tracking() {
        let mut p = TensorProfiler::new(100);
        p.record("op", 300, 1, 1);
        p.record("op", 100, 1, 1);
        p.record("op", 500, 1, 1);
        let prof = p.get_profile("op").expect("profile");
        assert_eq!(prof.min_ns, 100);
        assert_eq!(prof.max_ns, 500);
    }

    #[test]
    fn test_enable_disable_toggle() {
        let mut p = TensorProfiler::new(100);
        assert!(p.is_enabled());
        p.disable();
        assert!(!p.is_enabled());
        p.enable();
        assert!(p.is_enabled());
    }

    #[test]
    fn test_stats_accuracy() {
        let mut p = TensorProfiler::new(100);
        p.record("a", 100, 10, 5);
        p.record("b", 200, 20, 10);
        p.record("a", 300, 30, 15);

        let s = p.stats();
        assert_eq!(s.total_entries, 3);
        assert_eq!(s.unique_ops, 2);
        assert_eq!(s.total_ns, 600); // 400 (a) + 200 (b)
        assert!(s.enabled);
    }

    #[test]
    fn test_empty_profiler() {
        let p = TensorProfiler::new(100);
        assert_eq!(p.entry_count(), 0);
        assert!(p.get_profile("any").is_none());
        assert!(p.avg_ns("any").is_none());
        assert!(p.throughput("any").is_none());
        let hot = p.hottest_ops(5);
        assert!(hot.is_empty());
    }

    #[test]
    fn test_tick_increments() {
        let mut p = TensorProfiler::new(100);
        p.record("a", 10, 1, 1);
        p.tick();
        p.record("b", 20, 1, 1);

        assert_eq!(p.entries[0].tick, 0);
        assert_eq!(p.entries[1].tick, 1);
    }

    #[test]
    fn test_record_after_enable() {
        let mut p = TensorProfiler::new(100);
        p.disable();
        p.record("x", 100, 1, 1);
        assert_eq!(p.entry_count(), 0);

        p.enable();
        p.record("x", 200, 1, 1);
        assert_eq!(p.entry_count(), 1);
        let prof = p.get_profile("x").expect("profile");
        assert_eq!(prof.total_ns, 200);
    }

    #[test]
    fn test_input_elements_accumulation() {
        let mut p = TensorProfiler::new(100);
        p.record("op", 10, 100, 50);
        p.record("op", 20, 200, 100);
        let prof = p.get_profile("op").expect("profile");
        assert_eq!(prof.total_input_elements, 300);
        assert_eq!(prof.total_output_elements, 150);
    }

    #[test]
    fn test_hottest_ops_empty() {
        let p = TensorProfiler::new(100);
        assert!(p.hottest_ops(5).is_empty());
    }

    #[test]
    fn test_max_entries_zero() {
        // Edge case: max_entries = 0 means buffer stays empty.
        let mut p = TensorProfiler::new(0);
        p.record("a", 10, 1, 1);
        // Entry buffer stays empty but aggregates are updated.
        assert_eq!(p.entry_count(), 0);
        let prof = p.get_profile("a").expect("profile");
        assert_eq!(prof.call_count, 1);
    }

    #[test]
    fn test_stats_disabled() {
        let mut p = TensorProfiler::new(100);
        p.disable();
        let s = p.stats();
        assert!(!s.enabled);
    }

    #[test]
    fn test_large_duration_values() {
        let mut p = TensorProfiler::new(100);
        let big = u64::MAX / 2;
        p.record("big", big, 1, 1);
        let prof = p.get_profile("big").expect("profile");
        assert_eq!(prof.total_ns, big);
        assert_eq!(prof.min_ns, big);
        assert_eq!(prof.max_ns, big);
    }

    #[test]
    fn test_eviction_preserves_order() {
        let mut p = TensorProfiler::new(2);
        p.record("a", 10, 1, 1);
        p.record("b", 20, 1, 1);
        p.record("c", 30, 1, 1); // evicts "a"

        assert_eq!(p.entry_count(), 2);
        assert_eq!(p.entries[0].op_name, "b");
        assert_eq!(p.entries[1].op_name, "c");
    }

    #[test]
    fn test_single_entry_avg_ns() {
        let mut p = TensorProfiler::new(100);
        p.record("solo", 42, 1, 1);
        let avg = p.avg_ns("solo").expect("avg");
        assert!((avg - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_throughput_multiple_records() {
        let mut p = TensorProfiler::new(100);
        // 2 records: total_output = 200, total_ns = 2_000_000_000 (2s)
        // throughput = 200 / 2.0 = 100 elem/s
        p.record("op", 1_000_000_000, 100, 100);
        p.record("op", 1_000_000_000, 100, 100);
        let tp = p.throughput("op").expect("throughput");
        assert!((tp - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_reset_then_record() {
        let mut p = TensorProfiler::new(100);
        p.record("a", 100, 1, 1);
        p.reset();
        p.record("b", 200, 2, 2);
        assert_eq!(p.entry_count(), 1);
        assert!(p.get_profile("a").is_none());
        let prof = p.get_profile("b").expect("profile");
        assert_eq!(prof.call_count, 1);
    }

    #[test]
    fn test_op_name_preserves_string() {
        let mut p = TensorProfiler::new(100);
        let name = "my_custom_op_v2.1";
        p.record(name, 10, 1, 1);
        let prof = p.get_profile(name).expect("profile");
        assert_eq!(prof.op_name, name);
    }
}
