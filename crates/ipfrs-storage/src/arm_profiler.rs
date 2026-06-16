//! ARM Performance Profiler
//!
//! Provides profiling utilities for ARM devices (Raspberry Pi, Jetson, etc.)
//! with NEON SIMD detection and performance monitoring.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// ARM architecture feature detection
#[derive(Debug, Clone)]
pub struct ArmFeatures {
    /// NEON SIMD support detected
    pub has_neon: bool,
    /// AArch64 architecture
    pub is_aarch64: bool,
    /// ARMv7 architecture
    pub is_armv7: bool,
}

impl ArmFeatures {
    /// Detect ARM features at runtime
    pub fn detect() -> Self {
        let is_aarch64 = cfg!(target_arch = "aarch64");
        let is_armv7 = cfg!(target_arch = "arm");

        // NEON is standard on AArch64, optional on ARMv7
        let has_neon = if is_aarch64 {
            true
        } else if is_armv7 {
            // On ARMv7, NEON is optional - check CPU features
            #[cfg(target_arch = "arm")]
            {
                // Try to detect NEON through various methods
                // Note: This is a simplified check
                std::arch::is_arm_feature_detected!("neon")
            }
            #[cfg(not(target_arch = "arm"))]
            {
                false
            }
        } else {
            false
        };

        Self {
            has_neon,
            is_aarch64,
            is_armv7,
        }
    }

    /// Check if running on any ARM architecture
    pub fn is_arm(&self) -> bool {
        self.is_aarch64 || self.is_armv7
    }
}

/// Performance counter for ARM profiling
#[derive(Debug, Clone)]
pub struct ArmPerfCounter {
    name: String,
    count: Arc<AtomicU64>,
    total_time_ns: Arc<AtomicU64>,
}

impl ArmPerfCounter {
    /// Create a new performance counter
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            count: Arc::new(AtomicU64::new(0)),
            total_time_ns: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Start timing an operation
    pub fn start(&self) -> ArmPerfTimer {
        ArmPerfTimer {
            counter: self.clone(),
            start: Instant::now(),
        }
    }

    /// Get total operation count
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get total time spent
    pub fn total_time(&self) -> Duration {
        Duration::from_nanos(self.total_time_ns.load(Ordering::Relaxed))
    }

    /// Get average time per operation
    pub fn avg_time(&self) -> Duration {
        let count = self.count();
        Duration::from_nanos(
            self.total_time_ns
                .load(Ordering::Relaxed)
                .checked_div(count)
                .unwrap_or(0),
        )
    }

    /// Reset counter
    pub fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
        self.total_time_ns.store(0, Ordering::Relaxed);
    }

    /// Get counter name
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// RAII timer for performance measurement
pub struct ArmPerfTimer {
    counter: ArmPerfCounter,
    start: Instant,
}

impl Drop for ArmPerfTimer {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_nanos() as u64;
        self.counter.count.fetch_add(1, Ordering::Relaxed);
        self.counter
            .total_time_ns
            .fetch_add(elapsed, Ordering::Relaxed);
    }
}

/// ARM profiling report
#[derive(Debug, Clone)]
pub struct ArmPerfReport {
    /// ARM features detected
    pub features: ArmFeatures,
    /// Performance counters
    pub counters: Vec<(String, u64, Duration, Duration)>, // (name, count, total, avg)
}

impl ArmPerfReport {
    /// Create a profiling report from counters
    pub fn from_counters(counters: &[ArmPerfCounter]) -> Self {
        let features = ArmFeatures::detect();
        let counters = counters
            .iter()
            .map(|c| {
                (
                    c.name().to_string(),
                    c.count(),
                    c.total_time(),
                    c.avg_time(),
                )
            })
            .collect();

        Self { features, counters }
    }

    /// Print report to stdout
    pub fn print(&self) {
        println!("=== ARM Performance Report ===");
        println!(
            "Architecture: {}",
            if self.features.is_aarch64 {
                "AArch64"
            } else if self.features.is_armv7 {
                "ARMv7"
            } else {
                "x86_64 (not ARM)"
            }
        );
        println!("NEON support: {}", self.features.has_neon);
        println!("\nPerformance Counters:");

        for (name, count, total, avg) in &self.counters {
            println!("  {name}: {count} ops, total: {total:?}, avg: {avg:?}");
        }
    }
}

/// ARM-optimized hash computation using NEON when available
#[cfg(target_arch = "aarch64")]
pub mod neon_hash {
    use std::arch::aarch64::*;

    /// Compute hash using NEON SIMD instructions (AArch64)
    ///
    /// This is a simplified example - real implementations would use
    /// more sophisticated hash algorithms optimized for NEON.
    ///
    /// # Safety
    ///
    /// Caller must ensure the target CPU supports NEON (AArch64 SIMD) instructions.
    /// This function is only valid on AArch64 targets with NEON support enabled.
    #[target_feature(enable = "neon")]
    pub unsafe fn hash_block_neon(data: &[u8]) -> u64 {
        let mut hash = 0xcbf29ce484222325u64; // FNV offset basis
        const FNV_PRIME: u64 = 0x100000001b3;

        // Process 16 bytes at a time with NEON
        let chunks = data.chunks_exact(16);
        let remainder = chunks.remainder();

        for chunk in chunks {
            // Load 16 bytes into NEON register
            let v = vld1q_u8(chunk.as_ptr());

            // Extract bytes and update hash
            // Note: This is a simple implementation - production code
            // would use more efficient NEON operations
            let bytes: [u8; 16] = std::mem::transmute(v);
            for &byte in &bytes {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        }

        // Process remaining bytes
        for &byte in remainder {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }

        hash
    }
}

/// Fallback hash computation for non-ARM or when NEON is not available
pub fn hash_block_fallback(data: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64; // FNV offset basis
    const FNV_PRIME: u64 = 0x100000001b3;

    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    hash
}

/// Hash a block using the best available method (NEON or fallback)
pub fn hash_block(data: &[u8]) -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        // Use NEON on AArch64
        unsafe { neon_hash::hash_block_neon(data) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // Fallback for non-ARM
        hash_block_fallback(data)
    }
}

/// Power profile for low-power operation tuning
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerProfile {
    /// Maximum performance, no power saving
    Performance,
    /// Balanced mode with moderate batching
    Balanced,
    /// Low power mode with aggressive batching and delays
    LowPower,
    /// Custom profile with specific parameters
    Custom {
        batch_size: usize,
        batch_delay_ms: u64,
    },
}

impl PowerProfile {
    /// Get batch size for this profile
    pub fn batch_size(&self) -> usize {
        match self {
            PowerProfile::Performance => 1,
            PowerProfile::Balanced => 10,
            PowerProfile::LowPower => 50,
            PowerProfile::Custom { batch_size, .. } => *batch_size,
        }
    }

    /// Get batch delay in milliseconds
    pub fn batch_delay_ms(&self) -> u64 {
        match self {
            PowerProfile::Performance => 0,
            PowerProfile::Balanced => 10,
            PowerProfile::LowPower => 100,
            PowerProfile::Custom { batch_delay_ms, .. } => *batch_delay_ms,
        }
    }

    /// Get batch delay as Duration
    pub fn batch_delay(&self) -> Duration {
        Duration::from_millis(self.batch_delay_ms())
    }
}

/// Low-power operation batcher
///
/// Batches operations to reduce CPU wake-ups and save power.
/// Particularly useful on battery-powered ARM devices.
pub struct LowPowerBatcher<T> {
    profile: PowerProfile,
    buffer: Arc<std::sync::Mutex<Vec<T>>>,
}

impl<T> LowPowerBatcher<T> {
    /// Create a new batcher with the given power profile
    pub fn new(profile: PowerProfile) -> Self {
        Self {
            profile,
            buffer: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    /// Add an item to the batch
    ///
    /// Returns the current batch if it's ready to be processed
    pub fn push(&self, item: T) -> Option<Vec<T>> {
        let mut buffer = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
        buffer.push(item);

        if buffer.len() >= self.profile.batch_size() {
            Some(std::mem::take(&mut *buffer))
        } else {
            None
        }
    }

    /// Flush the current batch (returns all pending items)
    pub fn flush(&self) -> Vec<T> {
        let mut buffer = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *buffer)
    }

    /// Get the current power profile
    pub fn profile(&self) -> PowerProfile {
        self.profile
    }

    /// Get the number of pending items
    pub fn pending(&self) -> usize {
        self.buffer.lock().unwrap_or_else(|e| e.into_inner()).len()
    }
}

/// Power statistics tracker
#[derive(Debug, Clone, Default)]
pub struct PowerStats {
    /// Number of CPU wake-ups (batch flushes)
    pub wakeups: u64,
    /// Number of operations batched
    pub operations: u64,
    /// Total time spent in batched delays
    pub delay_time: Duration,
}

impl PowerStats {
    /// Create a new power stats tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a batch operation
    pub fn record_batch(&mut self, ops: usize, delay: Duration) {
        self.wakeups += 1;
        self.operations += ops as u64;
        self.delay_time += delay;
    }

    /// Get average operations per wake-up
    pub fn avg_ops_per_wakeup(&self) -> f64 {
        if self.wakeups == 0 {
            0.0
        } else {
            self.operations as f64 / self.wakeups as f64
        }
    }

    /// Get power saving ratio (higher is better)
    ///
    /// This estimates how much we've reduced wake-ups compared to
    /// processing each operation individually.
    pub fn power_saving_ratio(&self) -> f64 {
        if self.operations == 0 {
            1.0
        } else {
            self.wakeups as f64 / self.operations as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arm_features() {
        let _features = ArmFeatures::detect();

        // Should detect correctly based on compile target
        #[cfg(target_arch = "aarch64")]
        {
            assert!(_features.is_aarch64);
            assert!(_features.has_neon);
        }

        #[cfg(target_arch = "arm")]
        {
            assert!(_features.is_armv7);
        }

        // On non-ARM, just verify we can detect features
        #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
        {
            assert!(!_features.is_arm());
        }
    }

    #[test]
    fn test_perf_counter() {
        let counter = ArmPerfCounter::new("test_op");

        {
            let _timer = counter.start();
            std::thread::sleep(Duration::from_millis(10));
        }

        assert_eq!(counter.count(), 1);
        assert!(counter.total_time() >= Duration::from_millis(10));
        assert!(counter.avg_time() >= Duration::from_millis(10));
    }

    #[test]
    fn test_hash_block() {
        let data = b"hello world";
        let hash1 = hash_block(data);

        // Both implementations should produce consistent results
        #[cfg(not(target_arch = "aarch64"))]
        {
            let hash2 = hash_block_fallback(data);
            assert_eq!(hash1, hash2);
        }

        // Hash should be deterministic
        assert_eq!(hash1, hash_block(data));
    }

    #[test]
    fn test_perf_report() {
        let counter1 = ArmPerfCounter::new("op1");
        let counter2 = ArmPerfCounter::new("op2");

        {
            let _t = counter1.start();
            std::thread::sleep(Duration::from_millis(1));
        }

        {
            let _t = counter2.start();
            std::thread::sleep(Duration::from_millis(1));
        }

        let report = ArmPerfReport::from_counters(&[counter1, counter2]);
        assert_eq!(report.counters.len(), 2);
    }

    #[test]
    fn test_power_profile() {
        let perf = PowerProfile::Performance;
        assert_eq!(perf.batch_size(), 1);
        assert_eq!(perf.batch_delay_ms(), 0);

        let balanced = PowerProfile::Balanced;
        assert_eq!(balanced.batch_size(), 10);
        assert_eq!(balanced.batch_delay_ms(), 10);

        let low = PowerProfile::LowPower;
        assert_eq!(low.batch_size(), 50);
        assert_eq!(low.batch_delay_ms(), 100);

        let custom = PowerProfile::Custom {
            batch_size: 20,
            batch_delay_ms: 30,
        };
        assert_eq!(custom.batch_size(), 20);
        assert_eq!(custom.batch_delay_ms(), 30);
    }

    #[test]
    fn test_low_power_batcher() {
        let batcher: LowPowerBatcher<i32> = LowPowerBatcher::new(PowerProfile::Custom {
            batch_size: 3,
            batch_delay_ms: 0,
        });

        assert_eq!(batcher.pending(), 0);

        // First two pushes shouldn't trigger batch
        assert!(batcher.push(1).is_none());
        assert_eq!(batcher.pending(), 1);

        assert!(batcher.push(2).is_none());
        assert_eq!(batcher.pending(), 2);

        // Third push should trigger batch
        let batch = batcher.push(3);
        assert!(batch.is_some());
        let batch = batch.unwrap();
        assert_eq!(batch, vec![1, 2, 3]);
        assert_eq!(batcher.pending(), 0);

        // Test flush
        batcher.push(4);
        batcher.push(5);
        let flushed = batcher.flush();
        assert_eq!(flushed, vec![4, 5]);
        assert_eq!(batcher.pending(), 0);
    }

    #[test]
    fn test_power_stats() {
        let mut stats = PowerStats::new();
        assert_eq!(stats.wakeups, 0);
        assert_eq!(stats.operations, 0);

        stats.record_batch(10, Duration::from_millis(5));
        assert_eq!(stats.wakeups, 1);
        assert_eq!(stats.operations, 10);
        assert_eq!(stats.avg_ops_per_wakeup(), 10.0);

        stats.record_batch(5, Duration::from_millis(5));
        assert_eq!(stats.wakeups, 2);
        assert_eq!(stats.operations, 15);
        assert_eq!(stats.avg_ops_per_wakeup(), 7.5);

        // Power saving ratio: 2 wakeups / 15 operations ≈ 0.133
        let ratio = stats.power_saving_ratio();
        assert!(ratio > 0.0 && ratio < 1.0);
    }
}
