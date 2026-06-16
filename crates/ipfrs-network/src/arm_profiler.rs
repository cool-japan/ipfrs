//! ARM Performance Profiling for Network Operations
//!
//! This module provides performance profiling utilities specifically designed
//! for ARM devices including Raspberry Pi, Jetson, and other embedded platforms.
//!
//! ## Features
//!
//! - CPU usage tracking
//! - Memory usage monitoring
//! - Network throughput measurement
//! - Latency profiling
//! - Battery/power consumption estimation
//! - Thermal monitoring (on supported devices)
//!
//! ## Use Cases
//!
//! - Performance optimization for ARM devices
//! - Identifying bottlenecks on resource-constrained devices
//! - Regression testing across ARM platforms
//! - Power consumption analysis

use parking_lot::RwLock;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Errors that can occur during profiling
#[derive(Debug, Error)]
pub enum ProfilerError {
    #[error("Profiler not started")]
    NotStarted,

    #[error("System information unavailable")]
    SystemInfoUnavailable,

    #[error("Insufficient samples for analysis")]
    InsufficientSamples,
}

/// ARM device profile
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArmDevice {
    /// Raspberry Pi (various models)
    RaspberryPi,
    /// NVIDIA Jetson (Nano, TX2, Xavier, etc.)
    Jetson,
    /// Generic ARM device
    Generic,
    /// Unknown device
    Unknown,
}

impl ArmDevice {
    /// Detect ARM device type from system information
    pub fn detect() -> Self {
        // In a real implementation, read /proc/cpuinfo or device tree
        #[cfg(target_arch = "aarch64")]
        {
            Self::Generic
        }
        #[cfg(target_arch = "arm")]
        {
            Self::RaspberryPi
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
        {
            Self::Unknown
        }
    }

    /// Get recommended configuration for this device
    pub fn recommended_config(&self) -> ProfilerConfig {
        match self {
            ArmDevice::RaspberryPi => ProfilerConfig::raspberry_pi(),
            ArmDevice::Jetson => ProfilerConfig::jetson(),
            ArmDevice::Generic => ProfilerConfig::default(),
            ArmDevice::Unknown => ProfilerConfig::default(),
        }
    }
}

/// Performance profiling configuration
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Enable CPU usage tracking
    pub track_cpu: bool,

    /// Enable memory usage tracking
    pub track_memory: bool,

    /// Enable network throughput tracking
    pub track_throughput: bool,

    /// Enable latency profiling
    pub track_latency: bool,

    /// Sample interval for metrics
    pub sample_interval: Duration,

    /// Maximum number of samples to keep
    pub max_samples: usize,

    /// Enable thermal monitoring (if supported)
    pub track_thermal: bool,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            track_cpu: true,
            track_memory: true,
            track_throughput: true,
            track_latency: true,
            sample_interval: Duration::from_secs(1),
            max_samples: 1000,
            track_thermal: false,
        }
    }
}

impl ProfilerConfig {
    /// Configuration optimized for Raspberry Pi
    pub fn raspberry_pi() -> Self {
        Self {
            track_cpu: true,
            track_memory: true,
            track_throughput: true,
            track_latency: true,
            sample_interval: Duration::from_secs(2),
            max_samples: 500,
            track_thermal: true, // RPi has temp sensor
        }
    }

    /// Configuration optimized for NVIDIA Jetson
    pub fn jetson() -> Self {
        Self {
            track_cpu: true,
            track_memory: true,
            track_throughput: true,
            track_latency: true,
            sample_interval: Duration::from_millis(500),
            max_samples: 2000, // Jetson has more resources
            track_thermal: true,
        }
    }
}

/// Performance sample
#[derive(Debug, Clone)]
pub struct PerformanceSample {
    /// Timestamp of the sample
    pub timestamp: Instant,
    /// CPU usage percentage (0.0-100.0)
    pub cpu_usage: Option<f64>,
    /// Memory usage in bytes
    pub memory_usage: Option<u64>,
    /// Network throughput (bytes/sec)
    pub throughput: Option<u64>,
    /// Average latency in microseconds
    pub latency_us: Option<u64>,
    /// Temperature in Celsius (if available)
    pub temperature: Option<f32>,
}

/// Performance statistics
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// Average CPU usage
    pub avg_cpu: f64,
    /// Peak CPU usage
    pub peak_cpu: f64,
    /// Average memory usage (bytes)
    pub avg_memory: u64,
    /// Peak memory usage (bytes)
    pub peak_memory: u64,
    /// Average throughput (bytes/sec)
    pub avg_throughput: u64,
    /// Peak throughput (bytes/sec)
    pub peak_throughput: u64,
    /// Average latency (microseconds)
    pub avg_latency: u64,
    /// 95th percentile latency (microseconds)
    pub p95_latency: u64,
    /// 99th percentile latency (microseconds)
    pub p99_latency: u64,
    /// Average temperature (Celsius)
    pub avg_temperature: Option<f32>,
    /// Peak temperature (Celsius)
    pub peak_temperature: Option<f32>,
    /// Number of samples
    pub sample_count: usize,
    /// Profiling duration
    pub duration: Duration,
}

/// ARM performance profiler
pub struct ArmProfiler {
    /// Configuration
    config: ProfilerConfig,
    /// Device type
    device: ArmDevice,
    /// Performance samples
    samples: Arc<RwLock<VecDeque<PerformanceSample>>>,
    /// Start time
    start_time: Option<Instant>,
    /// Last sample time
    last_sample: Arc<RwLock<Option<Instant>>>,
}

impl ArmProfiler {
    /// Create a new ARM profiler
    pub fn new(config: ProfilerConfig) -> Self {
        let device = ArmDevice::detect();
        Self {
            config,
            device,
            samples: Arc::new(RwLock::new(VecDeque::new())),
            start_time: None,
            last_sample: Arc::new(RwLock::new(None)),
        }
    }

    /// Create with auto-detected device configuration
    pub fn auto_detect() -> Self {
        let device = ArmDevice::detect();
        let config = device.recommended_config();
        Self::new(config)
    }

    /// Start profiling
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
        *self.last_sample.write() = Some(Instant::now());
    }

    /// Stop profiling
    pub fn stop(&mut self) {
        self.start_time = None;
    }

    /// Record a performance sample
    pub fn record_sample(&self, sample: PerformanceSample) {
        let mut samples = self.samples.write();

        // Add new sample
        samples.push_back(sample);

        // Limit sample count
        while samples.len() > self.config.max_samples {
            samples.pop_front();
        }

        // Update last sample time
        *self.last_sample.write() = Some(Instant::now());
    }

    /// Record CPU usage
    pub fn record_cpu(&self, cpu_usage: f64) {
        if !self.config.track_cpu {
            return;
        }

        let sample = PerformanceSample {
            timestamp: Instant::now(),
            cpu_usage: Some(cpu_usage),
            memory_usage: None,
            throughput: None,
            latency_us: None,
            temperature: None,
        };

        self.record_sample(sample);
    }

    /// Record memory usage
    pub fn record_memory(&self, memory_bytes: u64) {
        if !self.config.track_memory {
            return;
        }

        let sample = PerformanceSample {
            timestamp: Instant::now(),
            cpu_usage: None,
            memory_usage: Some(memory_bytes),
            throughput: None,
            latency_us: None,
            temperature: None,
        };

        self.record_sample(sample);
    }

    /// Record network throughput
    pub fn record_throughput(&self, bytes_per_sec: u64) {
        if !self.config.track_throughput {
            return;
        }

        let sample = PerformanceSample {
            timestamp: Instant::now(),
            cpu_usage: None,
            memory_usage: None,
            throughput: Some(bytes_per_sec),
            latency_us: None,
            temperature: None,
        };

        self.record_sample(sample);
    }

    /// Record latency
    pub fn record_latency(&self, latency: Duration) {
        if !self.config.track_latency {
            return;
        }

        let sample = PerformanceSample {
            timestamp: Instant::now(),
            cpu_usage: None,
            memory_usage: None,
            throughput: None,
            latency_us: Some(latency.as_micros() as u64),
            temperature: None,
        };

        self.record_sample(sample);
    }

    /// Get performance statistics
    pub fn stats(&self) -> Result<PerformanceStats, ProfilerError> {
        let samples = self.samples.read();

        if samples.is_empty() {
            return Err(ProfilerError::InsufficientSamples);
        }

        let duration = self
            .start_time
            .map(|start| start.elapsed())
            .unwrap_or_default();

        // Calculate CPU stats
        let cpu_values: Vec<f64> = samples.iter().filter_map(|s| s.cpu_usage).collect();

        let avg_cpu = if !cpu_values.is_empty() {
            cpu_values.iter().sum::<f64>() / cpu_values.len() as f64
        } else {
            0.0
        };

        let peak_cpu = cpu_values.iter().cloned().fold(0.0f64, |a, b| a.max(b));

        // Calculate memory stats
        let memory_values: Vec<u64> = samples.iter().filter_map(|s| s.memory_usage).collect();

        let avg_memory = if !memory_values.is_empty() {
            memory_values.iter().sum::<u64>() / memory_values.len() as u64
        } else {
            0
        };

        let peak_memory = memory_values.iter().cloned().max().unwrap_or(0);

        // Calculate throughput stats
        let throughput_values: Vec<u64> = samples.iter().filter_map(|s| s.throughput).collect();

        let avg_throughput = if !throughput_values.is_empty() {
            throughput_values.iter().sum::<u64>() / throughput_values.len() as u64
        } else {
            0
        };

        let peak_throughput = throughput_values.iter().cloned().max().unwrap_or(0);

        // Calculate latency stats
        let mut latency_values: Vec<u64> = samples.iter().filter_map(|s| s.latency_us).collect();

        latency_values.sort_unstable();

        let avg_latency = if !latency_values.is_empty() {
            latency_values.iter().sum::<u64>() / latency_values.len() as u64
        } else {
            0
        };

        let p95_latency = if !latency_values.is_empty() {
            let idx = (latency_values.len() as f64 * 0.95) as usize;
            latency_values.get(idx).cloned().unwrap_or(0)
        } else {
            0
        };

        let p99_latency = if !latency_values.is_empty() {
            let idx = (latency_values.len() as f64 * 0.99) as usize;
            latency_values.get(idx).cloned().unwrap_or(0)
        } else {
            0
        };

        // Calculate temperature stats
        let temp_values: Vec<f32> = samples.iter().filter_map(|s| s.temperature).collect();

        let avg_temperature = if !temp_values.is_empty() {
            Some(temp_values.iter().sum::<f32>() / temp_values.len() as f32)
        } else {
            None
        };

        let peak_temperature = if !temp_values.is_empty() {
            Some(temp_values.iter().cloned().fold(0.0f32, |a, b| a.max(b)))
        } else {
            None
        };

        Ok(PerformanceStats {
            avg_cpu,
            peak_cpu,
            avg_memory,
            peak_memory,
            avg_throughput,
            peak_throughput,
            avg_latency,
            p95_latency,
            p99_latency,
            avg_temperature,
            peak_temperature,
            sample_count: samples.len(),
            duration,
        })
    }

    /// Get the detected device type
    pub fn device(&self) -> &ArmDevice {
        &self.device
    }

    /// Get the configuration
    pub fn config(&self) -> &ProfilerConfig {
        &self.config
    }

    /// Clear all samples
    pub fn clear(&self) {
        self.samples.write().clear();
    }

    /// Get sample count
    pub fn sample_count(&self) -> usize {
        self.samples.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let config = ProfilerConfig::default();
        let profiler = ArmProfiler::new(config);

        assert!(profiler.sample_count() == 0);
    }

    #[test]
    fn test_auto_detect() {
        let profiler = ArmProfiler::auto_detect();
        assert!(profiler.sample_count() == 0);
    }

    #[test]
    fn test_record_cpu() {
        let profiler = ArmProfiler::auto_detect();
        profiler.record_cpu(50.0);

        assert_eq!(profiler.sample_count(), 1);
    }

    #[test]
    fn test_record_memory() {
        let profiler = ArmProfiler::auto_detect();
        profiler.record_memory(1024 * 1024);

        assert_eq!(profiler.sample_count(), 1);
    }

    #[test]
    fn test_record_throughput() {
        let profiler = ArmProfiler::auto_detect();
        profiler.record_throughput(1000000);

        assert_eq!(profiler.sample_count(), 1);
    }

    #[test]
    fn test_record_latency() {
        let profiler = ArmProfiler::auto_detect();
        profiler.record_latency(Duration::from_millis(10));

        assert_eq!(profiler.sample_count(), 1);
    }

    #[test]
    fn test_stats_calculation() {
        let profiler = ArmProfiler::auto_detect();

        // Record some samples
        profiler.record_cpu(30.0);
        profiler.record_cpu(50.0);
        profiler.record_cpu(70.0);

        profiler.record_memory(1024);
        profiler.record_memory(2048);
        profiler.record_memory(3072);

        let stats = profiler
            .stats()
            .expect("test: stats should succeed with recorded samples");

        assert_eq!(stats.avg_cpu, 50.0);
        assert_eq!(stats.peak_cpu, 70.0);
        assert_eq!(stats.avg_memory, 2048);
        assert_eq!(stats.peak_memory, 3072);
    }

    #[test]
    fn test_latency_percentiles() {
        let profiler = ArmProfiler::auto_detect();

        // Record latencies
        for i in 1..=100 {
            profiler.record_latency(Duration::from_micros(i * 10));
        }

        let stats = profiler
            .stats()
            .expect("test: stats should succeed with 100 latency samples");

        assert!(stats.avg_latency > 0);
        assert!(stats.p95_latency > stats.avg_latency);
        assert!(stats.p99_latency > stats.p95_latency);
    }

    #[test]
    fn test_max_samples_limit() {
        let config = ProfilerConfig {
            max_samples: 10,
            ..Default::default()
        };

        let profiler = ArmProfiler::new(config);

        // Record more samples than the limit
        for i in 0..20 {
            profiler.record_cpu(i as f64);
        }

        // Should only keep the last 10 samples
        assert_eq!(profiler.sample_count(), 10);
    }

    #[test]
    fn test_clear_samples() {
        let profiler = ArmProfiler::auto_detect();

        profiler.record_cpu(50.0);
        profiler.record_cpu(60.0);
        assert_eq!(profiler.sample_count(), 2);

        profiler.clear();
        assert_eq!(profiler.sample_count(), 0);
    }

    #[test]
    fn test_device_configs() {
        let rpi_config = ProfilerConfig::raspberry_pi();
        let jetson_config = ProfilerConfig::jetson();

        assert!(rpi_config.sample_interval > jetson_config.sample_interval);
        assert!(rpi_config.max_samples < jetson_config.max_samples);
    }

    #[test]
    fn test_insufficient_samples_error() {
        let profiler = ArmProfiler::auto_detect();
        let result = profiler.stats();

        assert!(matches!(result, Err(ProfilerError::InsufficientSamples)));
    }
}
