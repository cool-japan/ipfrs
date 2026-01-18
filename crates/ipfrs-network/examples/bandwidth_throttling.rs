//! Bandwidth throttling example
//!
//! This example demonstrates how to use bandwidth throttling for:
//! - Mobile/cellular networks
//! - IoT/edge devices
//! - Low-power operation
//! - Custom throttle configurations

use ipfrs_network::{BandwidthThrottle, ThrottleConfig, ThrottleError, TrafficDirection};
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Bandwidth Throttling Example ===\n");

    // 1. Mobile network configuration
    println!("1. Mobile Network Configuration:");
    let mobile_throttle = BandwidthThrottle::new(ThrottleConfig::mobile())?;

    println!("   Upload limit: 1 MB/s, Download limit: 5 MB/s");
    println!("   Burst size: 2 MB");

    // Simulate uploading 500 KB
    match mobile_throttle.check_and_consume(TrafficDirection::Upload, 500_000) {
        Ok(()) => println!("   ✓ Upload of 500 KB allowed"),
        Err(e) => println!("   ✗ Upload throttled: {}", e),
    }

    // Check available bandwidth
    if let Some(available) = mobile_throttle.available_bandwidth(TrafficDirection::Upload) {
        println!("   Available upload bandwidth: {} KB\n", available / 1000);
    }

    // 2. IoT device configuration
    println!("2. IoT Device Configuration:");
    let iot_throttle = BandwidthThrottle::new(ThrottleConfig::iot())?;

    println!("   Upload limit: 128 KB/s, Download limit: 512 KB/s");
    println!("   Burst size: 256 KB");

    // Simulate downloading 400 KB
    match iot_throttle.check_and_consume(TrafficDirection::Download, 400_000) {
        Ok(()) => println!("   ✓ Download of 400 KB allowed"),
        Err(e) => println!("   ✗ Download throttled: {}", e),
    }

    if let Some(available) = iot_throttle.available_bandwidth(TrafficDirection::Download) {
        println!("   Available download bandwidth: {} KB\n", available / 1000);
    }

    // 3. Low-power mode configuration
    println!("3. Low-Power Mode Configuration:");
    let low_power_throttle = BandwidthThrottle::new(ThrottleConfig::low_power())?;

    println!("   Upload limit: 64 KB/s, Download limit: 256 KB/s");
    println!("   Very conservative for battery saving");

    // Simulate small upload (within limits)
    match low_power_throttle.check_and_consume(TrafficDirection::Upload, 50_000) {
        Ok(()) => println!("   ✓ Upload of 50 KB allowed"),
        Err(e) => println!("   ✗ Upload throttled: {}", e),
    }

    println!();

    // 4. Custom configuration
    println!("4. Custom Configuration:");
    let custom_config = ThrottleConfig {
        enabled: true,
        max_upload_bytes_per_sec: Some(256_000), // 256 KB/s
        max_download_bytes_per_sec: Some(1_000_000), // 1 MB/s
        burst_size_bytes: Some(512_000),         // 512 KB burst
        refill_interval: Duration::from_millis(100),
    };

    let custom_throttle = BandwidthThrottle::new(custom_config)?;
    println!("   Custom limits: 256 KB/s upload, 1 MB/s download");

    match custom_throttle.check_and_consume(TrafficDirection::Upload, 200_000) {
        Ok(()) => println!("   ✓ Upload of 200 KB allowed"),
        Err(e) => println!("   ✗ Upload throttled: {}", e),
    }

    println!();

    // 5. Demonstrating rate limiting
    println!("5. Rate Limiting Demonstration:");
    let demo_config = ThrottleConfig {
        enabled: true,
        max_upload_bytes_per_sec: Some(100_000), // 100 KB/s
        burst_size_bytes: Some(150_000),         // 150 KB burst
        ..Default::default()
    };

    let demo_throttle = BandwidthThrottle::new(demo_config)?;

    println!("   Limit: 100 KB/s, Burst: 150 KB");

    // First upload uses burst
    match demo_throttle.check_and_consume(TrafficDirection::Upload, 150_000) {
        Ok(()) => println!("   ✓ First upload (150 KB) used burst capacity"),
        Err(e) => println!("   ✗ First upload failed: {}", e),
    }

    // Second upload should be throttled (burst exhausted)
    match demo_throttle.check_and_consume(TrafficDirection::Upload, 50_000) {
        Ok(()) => println!("   ✓ Second upload (50 KB) allowed"),
        Err(ThrottleError::RateLimitExceeded(retry_after)) => {
            println!(
                "   ✗ Second upload throttled, retry after: {:?}",
                retry_after
            );
        }
        Err(e) => println!("   ✗ Second upload failed: {}", e),
    }

    println!();

    // 6. Independent upload/download limits
    println!("6. Independent Upload/Download Limits:");
    let independent_config = ThrottleConfig {
        enabled: true,
        max_upload_bytes_per_sec: Some(100_000),   // 100 KB/s
        max_download_bytes_per_sec: Some(500_000), // 500 KB/s
        burst_size_bytes: Some(200_000),
        ..Default::default()
    };

    let independent_throttle = BandwidthThrottle::new(independent_config)?;

    // Exhaust upload bandwidth
    let _ = independent_throttle.check_and_consume(TrafficDirection::Upload, 200_000);
    println!("   Upload bandwidth exhausted");

    // Download should still work
    match independent_throttle.check_and_consume(TrafficDirection::Download, 150_000) {
        Ok(()) => println!("   ✓ Download (150 KB) still allowed despite upload exhaustion"),
        Err(e) => println!("   ✗ Download failed: {}", e),
    }

    println!();

    // 7. Dynamic configuration updates
    println!("7. Dynamic Configuration Updates:");
    let mut dynamic_config = ThrottleConfig::iot();
    let mut dynamic_throttle = BandwidthThrottle::new(dynamic_config.clone())?;

    println!("   Initial: IoT configuration (128 KB/s upload)");

    // Upgrade to mobile configuration
    dynamic_config = ThrottleConfig::mobile();
    dynamic_throttle.update_config(dynamic_config)?;

    println!("   Updated: Mobile configuration (1 MB/s upload)");

    if let Some(available) = dynamic_throttle.available_bandwidth(TrafficDirection::Upload) {
        println!("   New available upload bandwidth: {} KB", available / 1000);
    }

    println!();

    // 8. Disabled throttling
    println!("8. Disabled Throttling:");
    let disabled_throttle = BandwidthThrottle::new(ThrottleConfig::default())?;

    match disabled_throttle.check_and_consume(TrafficDirection::Upload, 1_000_000) {
        Ok(()) => println!("   Throttling is disabled, but returned Ok"),
        Err(ThrottleError::Disabled) => println!("   ✓ Throttling disabled - no limits applied"),
        Err(e) => println!("   Error: {}", e),
    }

    println!("\n=== Example Complete ===");

    Ok(())
}
