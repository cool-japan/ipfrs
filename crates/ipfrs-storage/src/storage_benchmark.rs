//! Storage performance benchmarking utilities
//!
//! Provides comprehensive tools for measuring storage read/write/delete
//! throughput and latency across all operation types, including:
//!
//! - Configurable benchmark runs with warmup support
//! - Per-operation latency sampling (microsecond resolution)
//! - Statistical analysis: p50/p95/p99 percentiles, min/max
//! - Throughput computation in MB/s
//! - Deterministic pseudo-random block generation (xorshift64)
//! - Human-readable result formatting
//! - Aggregate statistics across multiple runs

use serde::{Deserialize, Serialize};
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// The type of storage operation being benchmarked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkOp {
    /// Write / put operation
    Write,
    /// Read / get operation
    Read,
    /// Delete / remove operation
    Delete,
    /// Mixed workload exercising all operation types
    Mixed,
}

impl fmt::Display for BenchmarkOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BenchmarkOp::Write => write!(f, "Write"),
            BenchmarkOp::Read => write!(f, "Read"),
            BenchmarkOp::Delete => write!(f, "Delete"),
            BenchmarkOp::Mixed => write!(f, "Mixed"),
        }
    }
}

/// Configuration controlling a single benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Operation type to benchmark
    pub op: BenchmarkOp,
    /// Size of each data block in bytes
    pub block_size: usize,
    /// Number of measured operations (after warmup)
    pub num_operations: usize,
    /// Number of warmup operations (excluded from measurements)
    pub warmup_ops: usize,
    /// PRNG seed for reproducible block generation
    pub seed: u64,
}

impl BenchmarkConfig {
    /// Create a default write-benchmark configuration.
    ///
    /// - 4 KiB blocks
    /// - 1 000 measured operations
    /// - 100 warmup operations
    pub fn default_write() -> Self {
        Self {
            op: BenchmarkOp::Write,
            block_size: 4096,
            num_operations: 1_000,
            warmup_ops: 100,
            seed: 0xdeadbeef_cafebabe,
        }
    }

    /// Create a default read-benchmark configuration.
    pub fn default_read() -> Self {
        Self {
            op: BenchmarkOp::Read,
            block_size: 4096,
            num_operations: 1_000,
            warmup_ops: 100,
            seed: 0xcafebabe_deadbeef,
        }
    }

    /// Create a default mixed-workload configuration.
    pub fn default_mixed() -> Self {
        Self {
            op: BenchmarkOp::Mixed,
            block_size: 65536,
            num_operations: 500,
            warmup_ops: 50,
            seed: 0x1234567890abcdef,
        }
    }
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self::default_write()
    }
}

/// A single latency measurement for one storage operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencySample {
    /// The operation that was timed
    pub op: BenchmarkOp,
    /// Round-trip latency in microseconds
    pub latency_us: u64,
    /// Number of bytes involved in the operation
    pub bytes: usize,
    /// Whether the operation completed successfully
    pub success: bool,
}

impl LatencySample {
    /// Construct a new sample.
    pub fn new(op: BenchmarkOp, latency_us: u64, bytes: usize, success: bool) -> Self {
        Self {
            op,
            latency_us,
            bytes,
            success,
        }
    }
}

/// Aggregated result of a completed benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// The configuration used to produce this result
    pub config: BenchmarkConfig,
    /// Total operations attempted (successful + failed)
    pub total_ops: usize,
    /// Operations that completed without error
    pub successful_ops: usize,
    /// Operations that returned an error
    pub failed_ops: usize,
    /// Total bytes transferred across all successful operations
    pub total_bytes: u64,
    /// Wall-clock duration of the benchmark in microseconds
    pub duration_us: u64,
    /// Aggregate throughput in megabytes per second
    pub throughput_mbps: f64,
    /// 50th-percentile (median) latency in microseconds
    pub latency_p50_us: u64,
    /// 95th-percentile latency in microseconds
    pub latency_p95_us: u64,
    /// 99th-percentile latency in microseconds
    pub latency_p99_us: u64,
    /// Minimum observed latency in microseconds
    pub latency_min_us: u64,
    /// Maximum observed latency in microseconds
    pub latency_max_us: u64,
}

/// Running aggregate statistics across multiple benchmark runs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchmarkStats {
    /// Number of benchmark runs completed
    pub runs: u64,
    /// Cumulative operation count across all runs
    pub total_ops: u64,
    /// Cumulative bytes transferred across all runs
    pub total_bytes: u64,
}

impl BenchmarkStats {
    /// Incorporate a completed [`BenchmarkResult`] into these stats.
    pub fn record(&mut self, result: &BenchmarkResult) {
        self.runs = self.runs.saturating_add(1);
        self.total_ops = self.total_ops.saturating_add(result.total_ops as u64);
        self.total_bytes = self.total_bytes.saturating_add(result.total_bytes);
    }

    /// Average operations per run (returns 0 when no runs have been recorded).
    pub fn avg_ops_per_run(&self) -> f64 {
        if self.runs == 0 {
            0.0
        } else {
            self.total_ops as f64 / self.runs as f64
        }
    }

    /// Average bytes per run.
    pub fn avg_bytes_per_run(&self) -> f64 {
        if self.runs == 0 {
            0.0
        } else {
            self.total_bytes as f64 / self.runs as f64
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StorageBenchmark
// ─────────────────────────────────────────────────────────────────────────────

/// Storage performance benchmarking engine.
///
/// Collects [`LatencySample`]s during a benchmark run, then computes
/// statistical summaries via [`StorageBenchmark::compute_result`].
///
/// # Example
///
/// ```rust
/// use ipfrs_storage::{BenchmarkConfig, BenchmarkOp, StorageBenchmark};
///
/// let config = BenchmarkConfig::default_write();
/// let mut bench = StorageBenchmark::new(config);
///
/// // Simulate recording some operations
/// bench.record_op(BenchmarkOp::Write, 120, 4096, true);
/// bench.record_op(BenchmarkOp::Write, 95, 4096, true);
///
/// let result = bench.compute_result(500_000); // 0.5 s in µs
/// println!("{}", StorageBenchmark::format_result(&result));
/// ```
pub struct StorageBenchmark {
    config: BenchmarkConfig,
    samples: Vec<LatencySample>,
    /// Internal xorshift64 PRNG state (seeded from `config.seed`)
    rng_state: u64,
}

impl StorageBenchmark {
    /// Create a new benchmark engine from the given configuration.
    ///
    /// The PRNG is seeded from `config.seed`; if the seed is zero the engine
    /// falls back to a fixed non-zero seed to satisfy xorshift64 requirements.
    pub fn new(config: BenchmarkConfig) -> Self {
        // xorshift64 must never have state == 0
        let rng_state = if config.seed == 0 {
            0x9e3779b97f4a7c15
        } else {
            config.seed
        };
        Self {
            config,
            samples: Vec::new(),
            rng_state,
        }
    }

    // ── Sample management ────────────────────────────────────────────────────

    /// Push a pre-built [`LatencySample`] onto the collection.
    pub fn add_sample(&mut self, sample: LatencySample) {
        self.samples.push(sample);
    }

    /// Record a single operation result, constructing a [`LatencySample`]
    /// and adding it to the internal collection.
    pub fn record_op(&mut self, op: BenchmarkOp, latency_us: u64, bytes: usize, success: bool) {
        self.add_sample(LatencySample::new(op, latency_us, bytes, success));
    }

    /// Return the number of samples currently held.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Fraction of recorded operations that succeeded (0.0 when no samples).
    pub fn success_rate(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let successes = self.samples.iter().filter(|s| s.success).count();
        successes as f64 / self.samples.len() as f64
    }

    /// Clear all recorded samples and reset the PRNG to the original seed.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.rng_state = if self.config.seed == 0 {
            0x9e3779b97f4a7c15
        } else {
            self.config.seed
        };
    }

    // ── Statistical computation ──────────────────────────────────────────────

    /// Compute a [`BenchmarkResult`] from all samples collected so far.
    ///
    /// `duration_us` is the wall-clock time of the entire run in microseconds.
    /// When `duration_us` is zero, throughput is reported as 0.0 to avoid
    /// division-by-zero.
    pub fn compute_result(&self, duration_us: u64) -> BenchmarkResult {
        let total_ops = self.samples.len();
        let successful_ops = self.samples.iter().filter(|s| s.success).count();
        let failed_ops = total_ops - successful_ops;

        let total_bytes: u64 = self
            .samples
            .iter()
            .filter(|s| s.success)
            .map(|s| s.bytes as u64)
            .fold(0u64, |acc, b| acc.saturating_add(b));

        let throughput_mbps = Self::compute_throughput(total_bytes, duration_us);

        // Collect latencies for all samples (regardless of success) for stats
        let mut latencies: Vec<u64> = self.samples.iter().map(|s| s.latency_us).collect();

        let (latency_min_us, latency_max_us) = if latencies.is_empty() {
            (0, 0)
        } else {
            let min = latencies.iter().copied().min().unwrap_or(0);
            let max = latencies.iter().copied().max().unwrap_or(0);
            (min, max)
        };

        let latency_p50_us = Self::compute_percentile(&mut latencies, 50.0);
        let latency_p95_us = Self::compute_percentile(&mut latencies, 95.0);
        let latency_p99_us = Self::compute_percentile(&mut latencies, 99.0);

        BenchmarkResult {
            config: self.config.clone(),
            total_ops,
            successful_ops,
            failed_ops,
            total_bytes,
            duration_us,
            throughput_mbps,
            latency_p50_us,
            latency_p95_us,
            latency_p99_us,
            latency_min_us,
            latency_max_us,
        }
    }

    /// Compute a percentile value from a mutable slice of latency samples.
    ///
    /// The slice is sorted in place. Returns 0 for an empty slice.
    /// `percentile` must be in the range [0.0, 100.0]; values outside that
    /// range are clamped.
    pub fn compute_percentile(samples: &mut [u64], percentile: f64) -> u64 {
        if samples.is_empty() {
            return 0;
        }
        samples.sort_unstable();

        let pct = percentile.clamp(0.0, 100.0);
        // Nearest-rank method (1-based)
        let rank = ((pct / 100.0) * samples.len() as f64).ceil() as usize;
        let idx = rank.saturating_sub(1).min(samples.len() - 1);
        samples[idx]
    }

    /// Compute throughput in megabytes per second.
    ///
    /// Returns `0.0` when `duration_us` is zero.
    pub fn compute_throughput(bytes: u64, duration_us: u64) -> f64 {
        if duration_us == 0 {
            return 0.0;
        }
        // bytes / (duration_us × 1e-6) / 1e6  ==  bytes / duration_us
        bytes as f64 / duration_us as f64
    }

    // ── Block generation ─────────────────────────────────────────────────────

    /// Generate a pseudo-random block of `size` bytes using the internal
    /// xorshift64 PRNG.
    ///
    /// Each call advances the PRNG state so successive calls produce
    /// different (but deterministic) data.
    pub fn generate_block(&mut self, size: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(size);
        let mut state = self.rng_state;

        let full_words = size / 8;
        let remainder = size % 8;

        for _ in 0..full_words {
            state = xorshift64(state);
            out.extend_from_slice(&state.to_le_bytes());
        }

        if remainder > 0 {
            state = xorshift64(state);
            let word_bytes = state.to_le_bytes();
            out.extend_from_slice(&word_bytes[..remainder]);
        }

        self.rng_state = state;
        out
    }

    // ── Formatting ───────────────────────────────────────────────────────────

    /// Format a [`BenchmarkResult`] as a human-readable summary string.
    pub fn format_result(result: &BenchmarkResult) -> String {
        let success_pct = if result.total_ops == 0 {
            0.0
        } else {
            100.0 * result.successful_ops as f64 / result.total_ops as f64
        };

        let duration_ms = result.duration_us as f64 / 1_000.0;
        let block_kb = result.config.block_size as f64 / 1024.0;
        let total_mb = result.total_bytes as f64 / (1024.0 * 1024.0);

        format!(
            "=== Benchmark Result ===\n\
             Operation  : {op}\n\
             Block size : {block_kb:.1} KiB\n\
             Operations : {total_ops} total / {successful_ops} ok / {failed_ops} failed ({success_pct:.1}%)\n\
             Data       : {total_mb:.2} MiB in {duration_ms:.2} ms\n\
             Throughput : {throughput_mbps:.3} MB/s\n\
             Latency    : min={min}µs  p50={p50}µs  p95={p95}µs  p99={p99}µs  max={max}µs",
            op              = result.config.op,
            block_kb        = block_kb,
            total_ops       = result.total_ops,
            successful_ops  = result.successful_ops,
            failed_ops      = result.failed_ops,
            success_pct     = success_pct,
            total_mb        = total_mb,
            duration_ms     = duration_ms,
            throughput_mbps = result.throughput_mbps,
            min             = result.latency_min_us,
            p50             = result.latency_p50_us,
            p95             = result.latency_p95_us,
            p99             = result.latency_p99_us,
            max             = result.latency_max_us,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// xorshift64 PRNG
// ─────────────────────────────────────────────────────────────────────────────

/// Advance one xorshift64 step and return the new state.
///
/// The algorithm is from Marsaglia 2003; period is 2⁶⁴ − 1.
/// Panics if `state == 0` (invalid for xorshift).
#[inline]
fn xorshift64(mut state: u64) -> u64 {
    debug_assert_ne!(state, 0, "xorshift64 state must not be zero");
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper constructors ──────────────────────────────────────────────────

    fn default_bench() -> StorageBenchmark {
        StorageBenchmark::new(BenchmarkConfig::default_write())
    }

    fn bench_with_seed(seed: u64) -> StorageBenchmark {
        let mut cfg = BenchmarkConfig::default_write();
        cfg.seed = seed;
        StorageBenchmark::new(cfg)
    }

    fn make_sample(latency: u64, bytes: usize, success: bool) -> LatencySample {
        LatencySample::new(BenchmarkOp::Write, latency, bytes, success)
    }

    // ── 1. add_sample / sample_count ────────────────────────────────────────

    #[test]
    fn test_add_sample_increments_count() {
        let mut bench = default_bench();
        assert_eq!(bench.sample_count(), 0);
        bench.add_sample(make_sample(100, 4096, true));
        assert_eq!(bench.sample_count(), 1);
        bench.add_sample(make_sample(200, 4096, false));
        assert_eq!(bench.sample_count(), 2);
    }

    // ── 2. record_op convenience wrapper ────────────────────────────────────

    #[test]
    fn test_record_op_stores_sample() {
        let mut bench = default_bench();
        bench.record_op(BenchmarkOp::Read, 50, 8192, true);
        assert_eq!(bench.sample_count(), 1);
        let s = &bench.samples[0];
        assert_eq!(s.latency_us, 50);
        assert_eq!(s.bytes, 8192);
        assert!(s.success);
    }

    // ── 3. success_rate with no samples ─────────────────────────────────────

    #[test]
    fn test_success_rate_empty() {
        let bench = default_bench();
        assert_eq!(bench.success_rate(), 0.0);
    }

    // ── 4. success_rate all success ──────────────────────────────────────────

    #[test]
    fn test_success_rate_all_success() {
        let mut bench = default_bench();
        for _ in 0..10 {
            bench.record_op(BenchmarkOp::Write, 100, 4096, true);
        }
        let rate = bench.success_rate();
        assert!((rate - 1.0).abs() < f64::EPSILON);
    }

    // ── 5. success_rate all failure ──────────────────────────────────────────

    #[test]
    fn test_success_rate_all_failure() {
        let mut bench = default_bench();
        for _ in 0..5 {
            bench.record_op(BenchmarkOp::Write, 300, 4096, false);
        }
        assert_eq!(bench.success_rate(), 0.0);
    }

    // ── 6. success_rate mixed ────────────────────────────────────────────────

    #[test]
    fn test_success_rate_mixed() {
        let mut bench = default_bench();
        bench.record_op(BenchmarkOp::Write, 100, 4096, true);
        bench.record_op(BenchmarkOp::Write, 200, 4096, false);
        let rate = bench.success_rate();
        assert!((rate - 0.5).abs() < f64::EPSILON);
    }

    // ── 7. reset clears samples and restores PRNG ───────────────────────────

    #[test]
    fn test_reset_clears_samples() {
        let mut bench = default_bench();
        bench.record_op(BenchmarkOp::Write, 100, 4096, true);
        bench.record_op(BenchmarkOp::Write, 200, 4096, true);
        bench.reset();
        assert_eq!(bench.sample_count(), 0);
    }

    #[test]
    fn test_reset_restores_prng() {
        let mut bench = bench_with_seed(42);
        let block_a = bench.generate_block(16);
        bench.reset();
        let block_b = bench.generate_block(16);
        // After reset, PRNG starts from the same seed → same output
        assert_eq!(block_a, block_b);
    }

    // ── 8. generate_block produces correct size ──────────────────────────────

    #[test]
    fn test_generate_block_exact_size() {
        let mut bench = bench_with_seed(1);
        for size in [0, 1, 7, 8, 9, 15, 16, 63, 64, 100, 4096, 65536] {
            let block = bench.generate_block(size);
            assert_eq!(block.len(), size, "expected {size} bytes");
        }
    }

    // ── 9. generate_block is deterministic ───────────────────────────────────

    #[test]
    fn test_generate_block_deterministic() {
        let mut a = bench_with_seed(999);
        let mut b = bench_with_seed(999);
        assert_eq!(a.generate_block(64), b.generate_block(64));
    }

    // ── 10. generate_block advances PRNG (successive blocks differ) ──────────

    #[test]
    fn test_generate_block_advances_prng() {
        let mut bench = bench_with_seed(7);
        let b1 = bench.generate_block(8);
        let b2 = bench.generate_block(8);
        assert_ne!(b1, b2);
    }

    // ── 11. compute_percentile empty slice ───────────────────────────────────

    #[test]
    fn test_percentile_empty() {
        let mut v: Vec<u64> = Vec::new();
        assert_eq!(StorageBenchmark::compute_percentile(&mut v, 50.0), 0);
    }

    // ── 12. compute_percentile p50 on known data ─────────────────────────────

    #[test]
    fn test_percentile_p50_known() {
        // sorted: [1, 2, 3, 4, 5] → median = 3
        let mut v = vec![5, 1, 3, 2, 4];
        assert_eq!(StorageBenchmark::compute_percentile(&mut v, 50.0), 3);
    }

    // ── 13. compute_percentile p95 ───────────────────────────────────────────

    #[test]
    fn test_percentile_p95() {
        // 20 values 1..=20; p95 → ceil(0.95*20)=19th → value 19
        let mut v: Vec<u64> = (1..=20).collect();
        fastrand::shuffle(&mut v);
        assert_eq!(StorageBenchmark::compute_percentile(&mut v, 95.0), 19);
    }

    // ── 14. compute_percentile p99 ───────────────────────────────────────────

    #[test]
    fn test_percentile_p99() {
        // 100 values 1..=100; p99 → ceil(0.99*100)=99th → value 99
        let mut v: Vec<u64> = (1..=100).collect();
        fastrand::shuffle(&mut v);
        assert_eq!(StorageBenchmark::compute_percentile(&mut v, 99.0), 99);
    }

    // ── 15. compute_percentile single element ────────────────────────────────

    #[test]
    fn test_percentile_single() {
        let mut v = vec![42u64];
        assert_eq!(StorageBenchmark::compute_percentile(&mut v, 50.0), 42);
        assert_eq!(StorageBenchmark::compute_percentile(&mut v, 99.0), 42);
    }

    // ── 16. compute_throughput formula ───────────────────────────────────────

    #[test]
    fn test_throughput_formula() {
        // 1 MiB in 1 second → 1.0 MB/s
        // bytes = 1_048_576, duration_us = 1_000_000
        // result = 1_048_576 / 1_000_000 = 1.048576 MB/s
        let mbps = StorageBenchmark::compute_throughput(1_048_576, 1_000_000);
        assert!((mbps - 1.048576).abs() < 1e-6, "got {mbps}");
    }

    // ── 17. compute_throughput zero duration ─────────────────────────────────

    #[test]
    fn test_throughput_zero_duration() {
        assert_eq!(StorageBenchmark::compute_throughput(1_000_000, 0), 0.0);
    }

    // ── 18. compute_result with known latency data ───────────────────────────

    #[test]
    fn test_compute_result_known_data() {
        let mut bench = default_bench();
        // 5 successful ops, each 4096 bytes, latencies 10–50 µs
        for i in 1u64..=5 {
            bench.record_op(BenchmarkOp::Write, i * 10, 4096, true);
        }
        // latencies: [10, 20, 30, 40, 50]
        let result = bench.compute_result(1_000_000);

        assert_eq!(result.total_ops, 5);
        assert_eq!(result.successful_ops, 5);
        assert_eq!(result.failed_ops, 0);
        assert_eq!(result.total_bytes, 5 * 4096);
        assert_eq!(result.latency_min_us, 10);
        assert_eq!(result.latency_max_us, 50);
        // p50 of [10,20,30,40,50]: ceil(0.5*5)=3rd → 30
        assert_eq!(result.latency_p50_us, 30);
    }

    // ── 19. compute_result all-failure case ──────────────────────────────────

    #[test]
    fn test_compute_result_all_failures() {
        let mut bench = default_bench();
        for _ in 0..10 {
            bench.record_op(BenchmarkOp::Write, 500, 4096, false);
        }
        let result = bench.compute_result(500_000);
        assert_eq!(result.successful_ops, 0);
        assert_eq!(result.failed_ops, 10);
        assert_eq!(result.total_bytes, 0);
        // throughput should be 0 because no bytes were transferred
        assert_eq!(result.throughput_mbps, 0.0);
    }

    // ── 20. compute_result zero duration ────────────────────────────────────

    #[test]
    fn test_compute_result_zero_duration() {
        let mut bench = default_bench();
        bench.record_op(BenchmarkOp::Write, 100, 4096, true);
        let result = bench.compute_result(0);
        assert_eq!(result.throughput_mbps, 0.0);
    }

    // ── 21. compute_result empty sample set ──────────────────────────────────

    #[test]
    fn test_compute_result_empty() {
        let bench = default_bench();
        let result = bench.compute_result(1_000_000);
        assert_eq!(result.total_ops, 0);
        assert_eq!(result.successful_ops, 0);
        assert_eq!(result.total_bytes, 0);
        assert_eq!(result.latency_min_us, 0);
        assert_eq!(result.latency_max_us, 0);
    }

    // ── 22. mixed op results tracked correctly ───────────────────────────────

    #[test]
    fn test_mixed_op_results() {
        let mut cfg = BenchmarkConfig::default_mixed();
        cfg.seed = 1;
        let mut bench = StorageBenchmark::new(cfg);

        bench.record_op(BenchmarkOp::Write, 80, 4096, true);
        bench.record_op(BenchmarkOp::Read, 40, 4096, true);
        bench.record_op(BenchmarkOp::Delete, 20, 0, true);
        bench.record_op(BenchmarkOp::Write, 200, 4096, false);

        let result = bench.compute_result(1_000_000);
        assert_eq!(result.total_ops, 4);
        assert_eq!(result.successful_ops, 3);
        assert_eq!(result.failed_ops, 1);
        // bytes from successful ops only: 4096 + 4096 + 0 = 8192
        assert_eq!(result.total_bytes, 8192);
    }

    // ── 23. large sample set (1000 ops) ─────────────────────────────────────

    #[test]
    fn test_large_sample_set() {
        let mut bench = bench_with_seed(12345);
        let mut state = 12345u64;
        for _ in 0..1000 {
            state = xorshift64(state);
            let latency = (state % 10_000) + 1; // 1..=10000 µs
            bench.record_op(BenchmarkOp::Write, latency, 4096, true);
        }
        let result = bench.compute_result(5_000_000);

        assert_eq!(result.total_ops, 1000);
        assert_eq!(result.successful_ops, 1000);
        assert!(result.latency_p50_us > 0);
        assert!(result.latency_p95_us >= result.latency_p50_us);
        assert!(result.latency_p99_us >= result.latency_p95_us);
        assert!(result.latency_max_us >= result.latency_p99_us);
        assert!(result.latency_min_us <= result.latency_p50_us);
    }

    // ── 24. warmup ops exclusion concept ─────────────────────────────────────

    #[test]
    fn test_warmup_ops_not_in_samples() {
        // Concept: callers perform warmup_ops iterations WITHOUT calling
        // record_op, then record only the measured ops.  We verify that
        // the result only reflects the explicitly recorded samples.
        let mut cfg = BenchmarkConfig::default_write();
        cfg.warmup_ops = 10;
        cfg.num_operations = 5;
        let mut bench = StorageBenchmark::new(cfg);

        // Only record the measured ops (warmup is the caller's responsibility)
        for _ in 0..5 {
            bench.record_op(BenchmarkOp::Write, 100, 4096, true);
        }

        let result = bench.compute_result(500_000);
        // Must equal num_operations, not num_operations + warmup_ops
        assert_eq!(result.total_ops, 5);
    }

    // ── 25. format_result contains key fields ────────────────────────────────

    #[test]
    fn test_format_result_contains_key_info() {
        let mut bench = default_bench();
        for i in 1u64..=4 {
            bench.record_op(BenchmarkOp::Write, i * 25, 4096, true);
        }
        let result = bench.compute_result(1_000_000);
        let s = StorageBenchmark::format_result(&result);

        assert!(s.contains("Write"), "op type missing");
        assert!(s.contains("Throughput"), "throughput label missing");
        assert!(s.contains("Latency"), "latency label missing");
        assert!(s.contains("p50"), "p50 label missing");
        assert!(s.contains("p95"), "p95 label missing");
        assert!(s.contains("p99"), "p99 label missing");
        assert!(s.contains("Operations"), "operations label missing");
    }

    // ── 26. BenchmarkStats.record accumulates correctly ──────────────────────

    #[test]
    fn test_benchmark_stats_record() {
        let mut stats = BenchmarkStats::default();
        assert_eq!(stats.runs, 0);

        let mut bench = default_bench();
        bench.record_op(BenchmarkOp::Write, 100, 4096, true);
        bench.record_op(BenchmarkOp::Write, 200, 4096, false);
        let r1 = bench.compute_result(1_000_000);
        stats.record(&r1);

        assert_eq!(stats.runs, 1);
        assert_eq!(stats.total_ops, 2);
        // only successful bytes: 4096
        assert_eq!(stats.total_bytes, 4096);
    }

    // ── 27. BenchmarkStats avg helpers ───────────────────────────────────────

    #[test]
    fn test_benchmark_stats_averages() {
        let mut stats = BenchmarkStats::default();

        let mut bench = default_bench();
        bench.record_op(BenchmarkOp::Write, 100, 4096, true);
        let r = bench.compute_result(1_000_000);
        stats.record(&r);

        bench.reset();
        bench.record_op(BenchmarkOp::Write, 100, 4096, true);
        bench.record_op(BenchmarkOp::Write, 100, 4096, true);
        let r2 = bench.compute_result(1_000_000);
        stats.record(&r2);

        // Two runs: 1 op and 2 ops → avg 1.5
        assert!((stats.avg_ops_per_run() - 1.5).abs() < f64::EPSILON);
    }

    // ── 28. BenchmarkStats avg with zero runs ────────────────────────────────

    #[test]
    fn test_benchmark_stats_avg_zero_runs() {
        let stats = BenchmarkStats::default();
        assert_eq!(stats.avg_ops_per_run(), 0.0);
        assert_eq!(stats.avg_bytes_per_run(), 0.0);
    }

    // ── 29. generate_block zero bytes ────────────────────────────────────────

    #[test]
    fn test_generate_block_zero_bytes() {
        let mut bench = bench_with_seed(1);
        let block = bench.generate_block(0);
        assert!(block.is_empty());
    }

    // ── 30. BenchmarkOp Display ──────────────────────────────────────────────

    #[test]
    fn test_benchmark_op_display() {
        assert_eq!(format!("{}", BenchmarkOp::Write), "Write");
        assert_eq!(format!("{}", BenchmarkOp::Read), "Read");
        assert_eq!(format!("{}", BenchmarkOp::Delete), "Delete");
        assert_eq!(format!("{}", BenchmarkOp::Mixed), "Mixed");
    }
}
