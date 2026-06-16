//! Heterogeneous Device Support
//!
//! This module provides device capability detection and adaptive resource management
//! for running tensor operations across diverse hardware (edge to cloud).

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DeviceError {
    #[error("Failed to detect device capabilities: {0}")]
    DetectionFailed(String),

    #[error("Unsupported device type: {0}")]
    UnsupportedDevice(String),

    #[error("Insufficient resources: {0}")]
    InsufficientResources(String),
}

/// Device type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    /// Edge device (IoT, mobile)
    Edge,
    /// Consumer device (laptop, desktop)
    Consumer,
    /// Server-class device
    Server,
    /// GPU-accelerated device
    GpuAccelerated,
    /// Cloud instance
    Cloud,
}

/// Device architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceArch {
    X86_64,
    Aarch64,
    Arm,
    Riscv,
    Other,
}

/// Memory tier information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    /// Total system memory in bytes
    pub total_bytes: u64,
    /// Available memory in bytes
    pub available_bytes: u64,
    /// Memory pressure (0.0 = no pressure, 1.0 = critical)
    pub pressure: f32,
}

impl MemoryInfo {
    /// Check if device has sufficient memory for operation
    pub fn has_capacity(&self, required_bytes: u64) -> bool {
        self.available_bytes >= required_bytes
    }

    /// Get memory utilization percentage
    pub fn utilization(&self) -> f32 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        ((self.total_bytes - self.available_bytes) as f32 / self.total_bytes as f32) * 100.0
    }
}

/// CPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    /// Number of logical cores
    pub logical_cores: usize,
    /// Number of physical cores
    pub physical_cores: usize,
    /// CPU architecture
    pub arch: DeviceArch,
    /// CPU frequency in MHz (if available)
    pub frequency_mhz: Option<u32>,
}

impl CpuInfo {
    /// Get recommended thread count for parallel operations
    pub fn recommended_threads(&self) -> usize {
        // Use 80% of logical cores to leave room for system
        (self.logical_cores as f32 * 0.8).ceil() as usize
    }
}

/// Device capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// Device type
    pub device_type: DeviceType,
    /// CPU information
    pub cpu: CpuInfo,
    /// Memory information
    pub memory: MemoryInfo,
    /// Has GPU acceleration
    pub has_gpu: bool,
    /// Has fast storage (SSD)
    pub has_fast_storage: bool,
    /// Network bandwidth estimate (Mbps)
    pub network_bandwidth_mbps: Option<u32>,
}

impl DeviceCapabilities {
    /// Detect device capabilities
    pub fn detect() -> Result<Self, DeviceError> {
        let cpu = Self::detect_cpu()?;
        let memory = Self::detect_memory()?;
        let device_type = Self::classify_device(&cpu, &memory);

        Ok(DeviceCapabilities {
            device_type,
            cpu,
            memory,
            has_gpu: Self::detect_gpu(),
            has_fast_storage: Self::detect_fast_storage(),
            network_bandwidth_mbps: None, // Would need network probing
        })
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_cpu() -> Result<CpuInfo, DeviceError> {
        let logical_cores = num_cpus::get();
        let physical_cores = num_cpus::get_physical();

        Ok(CpuInfo {
            logical_cores,
            physical_cores,
            arch: DeviceArch::X86_64,
            frequency_mhz: None,
        })
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_cpu() -> Result<CpuInfo, DeviceError> {
        let logical_cores = num_cpus::get();
        let physical_cores = num_cpus::get_physical();

        Ok(CpuInfo {
            logical_cores,
            physical_cores,
            arch: DeviceArch::Aarch64,
            frequency_mhz: None,
        })
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn detect_cpu() -> Result<CpuInfo, DeviceError> {
        let logical_cores = num_cpus::get();
        let physical_cores = num_cpus::get_physical();

        Ok(CpuInfo {
            logical_cores,
            physical_cores,
            arch: DeviceArch::Other,
            frequency_mhz: None,
        })
    }

    #[cfg(target_os = "linux")]
    fn detect_memory() -> Result<MemoryInfo, DeviceError> {
        use std::fs;

        let meminfo = fs::read_to_string("/proc/meminfo")
            .map_err(|e| DeviceError::DetectionFailed(format!("Failed to read meminfo: {}", e)))?;

        let mut total_kb = 0u64;
        let mut available_kb = 0u64;

        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                total_kb = Self::parse_meminfo_line(line)?;
            } else if line.starts_with("MemAvailable:") {
                available_kb = Self::parse_meminfo_line(line)?;
            }
        }

        let total_bytes = total_kb * 1024;
        let available_bytes = available_kb * 1024;
        let pressure = if total_bytes > 0 {
            1.0 - (available_bytes as f32 / total_bytes as f32)
        } else {
            0.0
        };

        Ok(MemoryInfo {
            total_bytes,
            available_bytes,
            pressure,
        })
    }

    #[cfg(not(target_os = "linux"))]
    fn detect_memory() -> Result<MemoryInfo, DeviceError> {
        // Fallback for non-Linux systems
        // Use sysinfo crate or platform-specific APIs
        Ok(MemoryInfo {
            total_bytes: 8 * 1024 * 1024 * 1024,     // Default 8GB
            available_bytes: 4 * 1024 * 1024 * 1024, // Default 4GB available
            pressure: 0.5,
        })
    }

    #[cfg(target_os = "linux")]
    fn parse_meminfo_line(line: &str) -> Result<u64, DeviceError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            parts[1].parse().map_err(|e| {
                DeviceError::DetectionFailed(format!("Failed to parse meminfo: {}", e))
            })
        } else {
            Err(DeviceError::DetectionFailed(
                "Invalid meminfo format".to_string(),
            ))
        }
    }

    fn detect_gpu() -> bool {
        // Simple heuristic: check for common GPU driver files
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new("/dev/dri").exists()
                || std::path::Path::new("/dev/nvidia0").exists()
        }

        #[cfg(not(target_os = "linux"))]
        false
    }

    fn detect_fast_storage() -> bool {
        // Heuristic: assume SSD if rotational is 0 on Linux
        #[cfg(target_os = "linux")]
        {
            if let Ok(contents) = std::fs::read_to_string("/sys/block/sda/queue/rotational") {
                contents.trim() == "0"
            } else {
                false
            }
        }

        #[cfg(not(target_os = "linux"))]
        false
    }

    fn classify_device(cpu: &CpuInfo, memory: &MemoryInfo) -> DeviceType {
        let total_gb = memory.total_bytes / (1024 * 1024 * 1024);

        match (cpu.logical_cores, total_gb) {
            (cores, gb) if cores >= 16 && gb >= 32 => DeviceType::Server,
            (cores, gb) if cores >= 8 && gb >= 16 => DeviceType::Consumer,
            (cores, gb) if cores <= 4 || gb <= 4 => DeviceType::Edge,
            _ => DeviceType::Consumer,
        }
    }

    /// Calculate optimal batch size based on available memory and model size
    pub fn optimal_batch_size(&self, model_size_bytes: u64, item_size_bytes: u64) -> usize {
        // Reserve 20% of available memory for overhead
        let usable_memory = (self.memory.available_bytes as f32 * 0.8) as u64;

        // Account for model size
        let memory_for_batch = usable_memory.saturating_sub(model_size_bytes);

        if memory_for_batch == 0 || item_size_bytes == 0 {
            return 1;
        }

        // Calculate batch size
        let batch_size = (memory_for_batch / item_size_bytes) as usize;

        // Clamp to reasonable range
        batch_size.clamp(1, 1024)
    }

    /// Get recommended worker count for parallel processing
    pub fn recommended_workers(&self) -> usize {
        match self.device_type {
            DeviceType::Edge => 1.max(self.cpu.logical_cores / 2),
            DeviceType::Consumer => self.cpu.logical_cores,
            DeviceType::Server | DeviceType::Cloud => self.cpu.logical_cores * 2,
            DeviceType::GpuAccelerated => self.cpu.logical_cores,
        }
    }
}

/// Adaptive batch size calculator
pub struct AdaptiveBatchSizer {
    capabilities: Arc<DeviceCapabilities>,
    min_batch_size: usize,
    max_batch_size: usize,
    target_memory_utilization: f32,
}

impl AdaptiveBatchSizer {
    /// Create a new adaptive batch sizer
    pub fn new(capabilities: Arc<DeviceCapabilities>) -> Self {
        Self {
            capabilities,
            min_batch_size: 1,
            max_batch_size: 1024,
            target_memory_utilization: 0.7, // Target 70% memory utilization
        }
    }

    /// Set minimum batch size
    pub fn with_min_batch_size(mut self, size: usize) -> Self {
        self.min_batch_size = size;
        self
    }

    /// Set maximum batch size
    pub fn with_max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    /// Set target memory utilization (0.0-1.0)
    pub fn with_target_utilization(mut self, utilization: f32) -> Self {
        self.target_memory_utilization = utilization.clamp(0.1, 0.9);
        self
    }

    /// Calculate adaptive batch size
    pub fn calculate(&self, item_size_bytes: u64, model_size_bytes: u64) -> usize {
        let available = (self.capabilities.memory.available_bytes as f32
            * self.target_memory_utilization) as u64;
        let memory_for_batch = available.saturating_sub(model_size_bytes);

        if memory_for_batch == 0 || item_size_bytes == 0 {
            return self.min_batch_size;
        }

        let batch_size = (memory_for_batch / item_size_bytes) as usize;
        batch_size.clamp(self.min_batch_size, self.max_batch_size)
    }

    /// Adjust batch size based on current memory pressure
    pub fn adjust_for_pressure(&self, current_batch_size: usize) -> usize {
        let pressure = self.capabilities.memory.pressure;

        if pressure > 0.9 {
            // Critical pressure: halve batch size
            (current_batch_size / 2).max(self.min_batch_size)
        } else if pressure > 0.7 {
            // High pressure: reduce by 25%
            ((current_batch_size as f32 * 0.75) as usize).max(self.min_batch_size)
        } else if pressure < 0.3 && current_batch_size < self.max_batch_size {
            // Low pressure: increase by 25%
            ((current_batch_size as f32 * 1.25) as usize).min(self.max_batch_size)
        } else {
            current_batch_size
        }
    }
}

/// Device profiler for performance optimization
pub struct DeviceProfiler {
    capabilities: Arc<DeviceCapabilities>,
}

impl DeviceProfiler {
    /// Create a new device profiler
    pub fn new(capabilities: Arc<DeviceCapabilities>) -> Self {
        Self { capabilities }
    }

    /// Profile memory bandwidth (GB/s)
    pub fn profile_memory_bandwidth(&self) -> f64 {
        use std::time::Instant;

        // Allocate test buffer (10 MB)
        let size = 10 * 1024 * 1024;
        let mut buffer = vec![0u8; size];

        // Sequential write test
        let start = Instant::now();
        for (i, item) in buffer.iter_mut().enumerate().take(size) {
            *item = (i & 0xFF) as u8;
        }
        let write_duration = start.elapsed();

        // Sequential read test
        let start = Instant::now();
        let mut _sum: u64 = 0;
        for &byte in &buffer {
            _sum += byte as u64;
        }
        let read_duration = start.elapsed();

        // Calculate bandwidth (GB/s)
        let write_bandwidth = (size as f64) / write_duration.as_secs_f64() / 1e9;
        let read_bandwidth = (size as f64) / read_duration.as_secs_f64() / 1e9;

        // Return average
        (write_bandwidth + read_bandwidth) / 2.0
    }

    /// Profile compute throughput (FLOPS)
    pub fn profile_compute_throughput(&self) -> f64 {
        use std::time::Instant;

        // Simple FP32 FLOPS test
        let iterations = 10_000_000;
        let mut result = 1.0f32;

        let start = Instant::now();
        for i in 0..iterations {
            result = result * 1.0001 + (i as f32) * 0.0001;
        }
        let duration = start.elapsed();

        // Calculate FLOPS (2 operations per iteration: multiply and add)
        let flops = (iterations * 2) as f64 / duration.as_secs_f64();

        // Prevent optimization from removing computation
        if result < 0.0 {
            println!("Unexpected result: {}", result);
        }

        flops
    }

    /// Get device performance tier
    pub fn performance_tier(&self) -> DevicePerformanceTier {
        let memory_gb = self.capabilities.memory.total_bytes / (1024 * 1024 * 1024);
        let cores = self.capabilities.cpu.logical_cores;

        match (cores, memory_gb) {
            (c, m) if c >= 32 && m >= 64 => DevicePerformanceTier::High,
            (c, m) if c >= 8 && m >= 16 => DevicePerformanceTier::Medium,
            _ => DevicePerformanceTier::Low,
        }
    }
}

/// Device performance tier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DevicePerformanceTier {
    Low,
    Medium,
    High,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_detection() {
        let caps = DeviceCapabilities::detect();
        assert!(caps.is_ok());

        let caps = caps.expect("test: should succeed");
        assert!(caps.cpu.logical_cores > 0);
        assert!(caps.memory.total_bytes > 0);
    }

    #[test]
    fn test_memory_info() {
        let mem = MemoryInfo {
            total_bytes: 8 * 1024 * 1024 * 1024,
            available_bytes: 4 * 1024 * 1024 * 1024,
            pressure: 0.5,
        };

        assert!(mem.has_capacity(1024 * 1024 * 1024));
        assert!(!mem.has_capacity(5 * 1024 * 1024 * 1024));
        assert_eq!(mem.utilization(), 50.0);
    }

    #[test]
    fn test_cpu_info() {
        let cpu = CpuInfo {
            logical_cores: 8,
            physical_cores: 4,
            arch: DeviceArch::X86_64,
            frequency_mhz: Some(3000),
        };

        assert_eq!(cpu.recommended_threads(), 7); // 80% of 8 = 6.4, ceil to 7
    }

    #[test]
    fn test_optimal_batch_size() {
        let caps = DeviceCapabilities {
            device_type: DeviceType::Consumer,
            cpu: CpuInfo {
                logical_cores: 8,
                physical_cores: 4,
                arch: DeviceArch::X86_64,
                frequency_mhz: Some(3000),
            },
            memory: MemoryInfo {
                total_bytes: 16 * 1024 * 1024 * 1024,
                available_bytes: 8 * 1024 * 1024 * 1024,
                pressure: 0.5,
            },
            has_gpu: false,
            has_fast_storage: true,
            network_bandwidth_mbps: Some(1000),
        };

        let model_size = 1024 * 1024 * 1024; // 1GB model
        let item_size = 1024 * 1024; // 1MB per item

        let batch_size = caps.optimal_batch_size(model_size, item_size);
        assert!(batch_size > 0);
        assert!(batch_size <= 1024);
    }

    #[test]
    fn test_adaptive_batch_sizer() {
        let caps = Arc::new(DeviceCapabilities {
            device_type: DeviceType::Consumer,
            cpu: CpuInfo {
                logical_cores: 8,
                physical_cores: 4,
                arch: DeviceArch::X86_64,
                frequency_mhz: Some(3000),
            },
            memory: MemoryInfo {
                total_bytes: 16 * 1024 * 1024 * 1024,
                available_bytes: 8 * 1024 * 1024 * 1024,
                pressure: 0.5,
            },
            has_gpu: false,
            has_fast_storage: true,
            network_bandwidth_mbps: Some(1000),
        });

        let sizer = AdaptiveBatchSizer::new(caps)
            .with_min_batch_size(4)
            .with_max_batch_size(256);

        let batch_size = sizer.calculate(1024 * 1024, 512 * 1024 * 1024);
        assert!(batch_size >= 4);
        assert!(batch_size <= 256);
    }

    #[test]
    fn test_pressure_adjustment() {
        let caps_low_pressure = Arc::new(DeviceCapabilities {
            device_type: DeviceType::Consumer,
            cpu: CpuInfo {
                logical_cores: 8,
                physical_cores: 4,
                arch: DeviceArch::X86_64,
                frequency_mhz: Some(3000),
            },
            memory: MemoryInfo {
                total_bytes: 16 * 1024 * 1024 * 1024,
                available_bytes: 12 * 1024 * 1024 * 1024,
                pressure: 0.25,
            },
            has_gpu: false,
            has_fast_storage: true,
            network_bandwidth_mbps: Some(1000),
        });

        let sizer = AdaptiveBatchSizer::new(caps_low_pressure)
            .with_min_batch_size(4)
            .with_max_batch_size(256);

        let adjusted = sizer.adjust_for_pressure(32);
        assert!(adjusted >= 32); // Should increase under low pressure

        let caps_high_pressure = Arc::new(DeviceCapabilities {
            device_type: DeviceType::Consumer,
            cpu: CpuInfo {
                logical_cores: 8,
                physical_cores: 4,
                arch: DeviceArch::X86_64,
                frequency_mhz: Some(3000),
            },
            memory: MemoryInfo {
                total_bytes: 16 * 1024 * 1024 * 1024,
                available_bytes: 1024 * 1024 * 1024,
                pressure: 0.95,
            },
            has_gpu: false,
            has_fast_storage: true,
            network_bandwidth_mbps: Some(1000),
        });

        let sizer = AdaptiveBatchSizer::new(caps_high_pressure)
            .with_min_batch_size(4)
            .with_max_batch_size(256);

        let adjusted = sizer.adjust_for_pressure(32);
        assert!(adjusted < 32); // Should decrease under high pressure
    }

    #[test]
    fn test_device_profiler() {
        let caps = Arc::new(DeviceCapabilities::detect().expect("test: should succeed"));
        let profiler = DeviceProfiler::new(caps);

        let bandwidth = profiler.profile_memory_bandwidth();
        assert!(bandwidth > 0.0);

        let throughput = profiler.profile_compute_throughput();
        assert!(throughput > 0.0);

        let tier = profiler.performance_tier();
        assert!(matches!(
            tier,
            DevicePerformanceTier::Low
                | DevicePerformanceTier::Medium
                | DevicePerformanceTier::High
        ));
    }
}
