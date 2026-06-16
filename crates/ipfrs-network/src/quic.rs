//! QUIC transport utilities and configuration
//!
//! This module provides utilities for working with QUIC transport in IPFRS network.
//! While the actual QUIC transport is provided via libp2p-quic, this module offers
//! additional configuration, monitoring, and utility functions.
//!
//! ## Features
//!
//! - **Configuration**: QUIC transport configuration with sensible defaults
//! - **Connection Monitoring**: Track QUIC connection states and metrics
//! - **Performance Tuning**: Congestion control and flow control settings
//! - **Security**: TLS configuration and certificate management
//! - **Statistics**: Detailed QUIC protocol statistics
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_network::quic::{QuicConfig, QuicStats, CongestionControl};
//!
//! // Create QUIC configuration
//! let config = QuicConfig::default()
//!     .with_max_idle_timeout(30_000)
//!     .with_keep_alive(15_000)
//!     .with_congestion_control(CongestionControl::Cubic);
//!
//! assert_eq!(config.max_idle_timeout_ms, 30_000);
//! ```

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// QUIC configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuicConfig {
    /// Maximum idle timeout in milliseconds (0 = no timeout)
    pub max_idle_timeout_ms: u64,

    /// Keep-alive interval in milliseconds (0 = disabled)
    pub keep_alive_interval_ms: u64,

    /// Maximum concurrent bidirectional streams
    pub max_concurrent_bidi_streams: u64,

    /// Maximum concurrent unidirectional streams
    pub max_concurrent_uni_streams: u64,

    /// Initial maximum data (connection-level flow control)
    pub initial_max_data: u64,

    /// Maximum data per stream
    pub max_stream_data: u64,

    /// Maximum UDP payload size
    pub max_udp_payload_size: u16,

    /// Congestion control algorithm
    pub congestion_control: CongestionControl,

    /// Enable 0-RTT (faster reconnections)
    pub enable_0rtt: bool,

    /// Enable datagram support
    pub enable_datagrams: bool,

    /// Datagram receive buffer size
    pub datagram_recv_buffer_size: usize,

    /// Datagram send buffer size
    pub datagram_send_buffer_size: usize,
}

impl Default for QuicConfig {
    fn default() -> Self {
        Self {
            max_idle_timeout_ms: 60_000,    // 60 seconds
            keep_alive_interval_ms: 15_000, // 15 seconds
            max_concurrent_bidi_streams: 100,
            max_concurrent_uni_streams: 100,
            initial_max_data: 10_000_000, // 10 MB
            max_stream_data: 1_000_000,   // 1 MB
            max_udp_payload_size: 1452,   // Standard MTU minus headers
            congestion_control: CongestionControl::Cubic,
            enable_0rtt: true,
            enable_datagrams: true,
            datagram_recv_buffer_size: 65536,
            datagram_send_buffer_size: 65536,
        }
    }
}

impl QuicConfig {
    /// Create configuration optimized for low latency
    pub fn low_latency() -> Self {
        Self {
            max_idle_timeout_ms: 30_000,
            keep_alive_interval_ms: 10_000,
            max_concurrent_bidi_streams: 50,
            max_concurrent_uni_streams: 50,
            initial_max_data: 5_000_000,
            max_stream_data: 500_000,
            max_udp_payload_size: 1200,
            congestion_control: CongestionControl::Bbr,
            enable_0rtt: true,
            enable_datagrams: true,
            datagram_recv_buffer_size: 32768,
            datagram_send_buffer_size: 32768,
        }
    }

    /// Create configuration optimized for high throughput
    pub fn high_throughput() -> Self {
        Self {
            max_idle_timeout_ms: 120_000,
            keep_alive_interval_ms: 30_000,
            max_concurrent_bidi_streams: 500,
            max_concurrent_uni_streams: 500,
            initial_max_data: 50_000_000,
            max_stream_data: 10_000_000,
            max_udp_payload_size: 1452,
            congestion_control: CongestionControl::Cubic,
            enable_0rtt: true,
            enable_datagrams: true,
            datagram_recv_buffer_size: 262144, // 256 KB
            datagram_send_buffer_size: 262144,
        }
    }

    /// Create configuration optimized for mobile/unreliable networks
    pub fn mobile() -> Self {
        Self {
            max_idle_timeout_ms: 90_000,
            keep_alive_interval_ms: 20_000,
            max_concurrent_bidi_streams: 30,
            max_concurrent_uni_streams: 30,
            initial_max_data: 2_000_000,
            max_stream_data: 200_000,
            max_udp_payload_size: 1200,
            congestion_control: CongestionControl::Bbr,
            enable_0rtt: true,
            enable_datagrams: false, // Disable to reduce overhead
            datagram_recv_buffer_size: 16384,
            datagram_send_buffer_size: 16384,
        }
    }

    /// Builder pattern: set max idle timeout
    pub fn with_max_idle_timeout(mut self, timeout_ms: u64) -> Self {
        self.max_idle_timeout_ms = timeout_ms;
        self
    }

    /// Builder pattern: set keep-alive interval
    pub fn with_keep_alive(mut self, interval_ms: u64) -> Self {
        self.keep_alive_interval_ms = interval_ms;
        self
    }

    /// Builder pattern: set congestion control
    pub fn with_congestion_control(mut self, cc: CongestionControl) -> Self {
        self.congestion_control = cc;
        self
    }

    /// Builder pattern: enable/disable 0-RTT
    pub fn with_0rtt(mut self, enable: bool) -> Self {
        self.enable_0rtt = enable;
        self
    }

    /// Builder pattern: enable/disable datagrams
    pub fn with_datagrams(mut self, enable: bool) -> Self {
        self.enable_datagrams = enable;
        self
    }
}

/// Congestion control algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CongestionControl {
    /// CUBIC congestion control (default, good for high-bandwidth links)
    Cubic,
    /// BBR congestion control (better for varying network conditions)
    Bbr,
    /// NewReno congestion control (conservative, compatible)
    NewReno,
}

/// QUIC connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuicConnectionState {
    /// Handshake in progress
    Handshaking,
    /// Connection established
    Established,
    /// Connection closing
    Closing,
    /// Connection closed
    Closed,
    /// Connection failed
    Failed,
}

/// Information about a QUIC connection
#[derive(Debug, Clone)]
pub struct QuicConnectionInfo {
    /// Remote socket address
    pub remote_addr: SocketAddr,
    /// Connection state
    pub state: QuicConnectionState,
    /// Time when connection was established
    pub established_at: Option<Instant>,
    /// Round-trip time (RTT)
    pub rtt: Option<Duration>,
    /// Congestion window size
    pub congestion_window: u64,
    /// Bytes sent
    pub bytes_sent: u64,
    /// Bytes received
    pub bytes_received: u64,
    /// Active bidirectional streams
    pub active_bidi_streams: u64,
    /// Active unidirectional streams
    pub active_uni_streams: u64,
    /// Lost packets
    pub lost_packets: u64,
    /// Connection migration count
    pub migration_count: u32,
}

/// QUIC statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuicStats {
    /// Total connections established
    pub connections_established: u64,
    /// Total connections closed
    pub connections_closed: u64,
    /// Total connections failed
    pub connections_failed: u64,
    /// Currently active connections
    pub active_connections: u64,
    /// Total bytes sent across all connections
    pub total_bytes_sent: u64,
    /// Total bytes received across all connections
    pub total_bytes_received: u64,
    /// Total packets lost
    pub total_packets_lost: u64,
    /// Total connection migrations
    pub total_migrations: u64,
    /// Total 0-RTT connections
    pub zero_rtt_connections: u64,
    /// Average RTT (milliseconds)
    pub avg_rtt_ms: f64,
}

/// QUIC connection monitor
#[derive(Debug)]
pub struct QuicMonitor {
    /// Configuration
    config: QuicConfig,
    /// Active connections
    connections: Arc<RwLock<HashMap<SocketAddr, QuicConnectionInfo>>>,
    /// Statistics
    stats: Arc<RwLock<QuicStats>>,
}

impl QuicMonitor {
    /// Create a new QUIC monitor
    pub fn new(config: QuicConfig) -> Self {
        Self {
            config,
            connections: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(QuicStats::default())),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &QuicConfig {
        &self.config
    }

    /// Record connection established
    pub fn record_connection_established(&self, remote_addr: SocketAddr, used_0rtt: bool) {
        let mut connections = self.connections.write();
        connections.insert(
            remote_addr,
            QuicConnectionInfo {
                remote_addr,
                state: QuicConnectionState::Established,
                established_at: Some(Instant::now()),
                rtt: None,
                congestion_window: self.config.initial_max_data,
                bytes_sent: 0,
                bytes_received: 0,
                active_bidi_streams: 0,
                active_uni_streams: 0,
                lost_packets: 0,
                migration_count: 0,
            },
        );

        let mut stats = self.stats.write();
        stats.connections_established += 1;
        stats.active_connections += 1;
        if used_0rtt {
            stats.zero_rtt_connections += 1;
        }
    }

    /// Record connection closed
    pub fn record_connection_closed(&self, remote_addr: &SocketAddr) {
        let mut connections = self.connections.write();
        if let Some(info) = connections.remove(remote_addr) {
            let mut stats = self.stats.write();
            stats.connections_closed += 1;
            stats.active_connections = stats.active_connections.saturating_sub(1);
            stats.total_bytes_sent += info.bytes_sent;
            stats.total_bytes_received += info.bytes_received;
            stats.total_packets_lost += info.lost_packets;
            stats.total_migrations += info.migration_count as u64;
        }
    }

    /// Record connection failed
    pub fn record_connection_failed(&self, remote_addr: &SocketAddr) {
        let mut connections = self.connections.write();
        if connections.remove(remote_addr).is_some() {
            let mut stats = self.stats.write();
            stats.connections_failed += 1;
            stats.active_connections = stats.active_connections.saturating_sub(1);
        }
    }

    /// Update connection RTT
    pub fn update_rtt(&self, remote_addr: &SocketAddr, rtt: Duration) {
        let mut connections = self.connections.write();
        if let Some(info) = connections.get_mut(remote_addr) {
            info.rtt = Some(rtt);
        }

        // Update average RTT
        let mut stats = self.stats.write();
        let new_rtt_ms = rtt.as_millis() as f64;
        if stats.avg_rtt_ms == 0.0 {
            stats.avg_rtt_ms = new_rtt_ms;
        } else {
            // Exponential moving average
            stats.avg_rtt_ms = stats.avg_rtt_ms * 0.9 + new_rtt_ms * 0.1;
        }
    }

    /// Update connection bytes
    pub fn update_bytes(&self, remote_addr: &SocketAddr, sent: u64, received: u64) {
        let mut connections = self.connections.write();
        if let Some(info) = connections.get_mut(remote_addr) {
            info.bytes_sent = sent;
            info.bytes_received = received;
        }
    }

    /// Update stream counts
    pub fn update_streams(&self, remote_addr: &SocketAddr, bidi: u64, uni: u64) {
        let mut connections = self.connections.write();
        if let Some(info) = connections.get_mut(remote_addr) {
            info.active_bidi_streams = bidi;
            info.active_uni_streams = uni;
        }
    }

    /// Record connection migration
    pub fn record_migration(&self, remote_addr: &SocketAddr) {
        let mut connections = self.connections.write();
        if let Some(info) = connections.get_mut(remote_addr) {
            info.migration_count += 1;
        }
    }

    /// Get connection info
    pub fn get_connection(&self, remote_addr: &SocketAddr) -> Option<QuicConnectionInfo> {
        self.connections.read().get(remote_addr).cloned()
    }

    /// Get all active connections
    pub fn get_active_connections(&self) -> Vec<QuicConnectionInfo> {
        self.connections.read().values().cloned().collect()
    }

    /// Get statistics
    pub fn stats(&self) -> QuicStats {
        self.stats.read().clone()
    }

    /// Get number of active connections
    pub fn active_connection_count(&self) -> usize {
        self.connections.read().len()
    }

    /// Clear all statistics
    pub fn reset_stats(&self) {
        *self.stats.write() = QuicStats::default();
    }
}

impl Default for QuicMonitor {
    fn default() -> Self {
        Self::new(QuicConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[test]
    fn test_quic_config_default() {
        let config = QuicConfig::default();
        assert_eq!(config.max_idle_timeout_ms, 60_000);
        assert_eq!(config.keep_alive_interval_ms, 15_000);
        assert!(config.enable_0rtt);
        assert!(config.enable_datagrams);
    }

    #[test]
    fn test_quic_config_low_latency() {
        let config = QuicConfig::low_latency();
        assert_eq!(config.congestion_control, CongestionControl::Bbr);
        assert_eq!(config.max_idle_timeout_ms, 30_000);
        assert!(config.max_concurrent_bidi_streams < 100);
    }

    #[test]
    fn test_quic_config_high_throughput() {
        let config = QuicConfig::high_throughput();
        assert_eq!(config.congestion_control, CongestionControl::Cubic);
        assert!(config.max_concurrent_bidi_streams >= 500);
        assert!(config.initial_max_data >= 50_000_000);
    }

    #[test]
    fn test_quic_config_mobile() {
        let config = QuicConfig::mobile();
        assert_eq!(config.congestion_control, CongestionControl::Bbr);
        assert!(!config.enable_datagrams); // Disabled for mobile
        assert!(config.max_udp_payload_size <= 1200);
    }

    #[test]
    fn test_quic_config_builder() {
        let config = QuicConfig::default()
            .with_max_idle_timeout(30_000)
            .with_keep_alive(10_000)
            .with_congestion_control(CongestionControl::Bbr)
            .with_0rtt(false)
            .with_datagrams(false);

        assert_eq!(config.max_idle_timeout_ms, 30_000);
        assert_eq!(config.keep_alive_interval_ms, 10_000);
        assert_eq!(config.congestion_control, CongestionControl::Bbr);
        assert!(!config.enable_0rtt);
        assert!(!config.enable_datagrams);
    }

    #[test]
    fn test_quic_monitor_new() {
        let monitor = QuicMonitor::default();
        assert_eq!(monitor.active_connection_count(), 0);
        assert_eq!(monitor.stats().active_connections, 0);
    }

    #[test]
    fn test_quic_monitor_connection_lifecycle() {
        let monitor = QuicMonitor::default();
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);

        // Establish connection
        monitor.record_connection_established(addr, false);
        assert_eq!(monitor.active_connection_count(), 1);
        assert_eq!(monitor.stats().connections_established, 1);
        assert_eq!(monitor.stats().active_connections, 1);

        // Close connection
        monitor.record_connection_closed(&addr);
        assert_eq!(monitor.active_connection_count(), 0);
        assert_eq!(monitor.stats().connections_closed, 1);
        assert_eq!(monitor.stats().active_connections, 0);
    }

    #[test]
    fn test_quic_monitor_0rtt() {
        let monitor = QuicMonitor::default();
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);

        monitor.record_connection_established(addr, true);
        assert_eq!(monitor.stats().zero_rtt_connections, 1);
    }

    #[test]
    fn test_quic_monitor_failed_connection() {
        let monitor = QuicMonitor::default();
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);

        monitor.record_connection_established(addr, false);
        monitor.record_connection_failed(&addr);

        assert_eq!(monitor.active_connection_count(), 0);
        assert_eq!(monitor.stats().connections_failed, 1);
        assert_eq!(monitor.stats().active_connections, 0);
    }

    #[test]
    fn test_quic_monitor_rtt_update() {
        let monitor = QuicMonitor::default();
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);

        monitor.record_connection_established(addr, false);
        monitor.update_rtt(&addr, Duration::from_millis(50));

        let info = monitor
            .get_connection(&addr)
            .expect("test: connection should exist after RTT update");
        assert_eq!(info.rtt, Some(Duration::from_millis(50)));
        assert_eq!(monitor.stats().avg_rtt_ms, 50.0);
    }

    #[test]
    fn test_quic_monitor_bytes_update() {
        let monitor = QuicMonitor::default();
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);

        monitor.record_connection_established(addr, false);
        monitor.update_bytes(&addr, 1000, 2000);

        let info = monitor
            .get_connection(&addr)
            .expect("test: connection should exist after bytes update");
        assert_eq!(info.bytes_sent, 1000);
        assert_eq!(info.bytes_received, 2000);
    }

    #[test]
    fn test_quic_monitor_streams_update() {
        let monitor = QuicMonitor::default();
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);

        monitor.record_connection_established(addr, false);
        monitor.update_streams(&addr, 5, 3);

        let info = monitor
            .get_connection(&addr)
            .expect("test: connection should exist after update_streams");
        assert_eq!(info.active_bidi_streams, 5);
        assert_eq!(info.active_uni_streams, 3);
    }

    #[test]
    fn test_quic_monitor_migration() {
        let monitor = QuicMonitor::default();
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);

        monitor.record_connection_established(addr, false);
        monitor.record_migration(&addr);
        monitor.record_migration(&addr);

        let info = monitor
            .get_connection(&addr)
            .expect("test: connection should exist after record_migration");
        assert_eq!(info.migration_count, 2);
    }

    #[test]
    fn test_quic_monitor_get_active_connections() {
        let monitor = QuicMonitor::default();
        let addr1 = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let addr2 = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8081);

        monitor.record_connection_established(addr1, false);
        monitor.record_connection_established(addr2, true);

        let connections = monitor.get_active_connections();
        assert_eq!(connections.len(), 2);
    }

    #[test]
    fn test_quic_monitor_reset_stats() {
        let monitor = QuicMonitor::default();
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);

        monitor.record_connection_established(addr, false);
        monitor.reset_stats();

        let stats = monitor.stats();
        assert_eq!(stats.connections_established, 0);
        assert_eq!(stats.active_connections, 0);
    }

    #[test]
    fn test_quic_monitor_avg_rtt_calculation() {
        let monitor = QuicMonitor::default();
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);

        monitor.record_connection_established(addr, false);

        // First RTT
        monitor.update_rtt(&addr, Duration::from_millis(100));
        assert_eq!(monitor.stats().avg_rtt_ms, 100.0);

        // Second RTT (exponential moving average)
        monitor.update_rtt(&addr, Duration::from_millis(50));
        let avg = monitor.stats().avg_rtt_ms;
        assert!(avg > 50.0 && avg < 100.0);
    }

    #[test]
    fn test_congestion_control_variants() {
        let cubic = CongestionControl::Cubic;
        let bbr = CongestionControl::Bbr;
        let newreno = CongestionControl::NewReno;

        assert_ne!(cubic, bbr);
        assert_ne!(bbr, newreno);
        assert_ne!(cubic, newreno);
    }

    #[test]
    fn test_connection_state_variants() {
        let states = [
            QuicConnectionState::Handshaking,
            QuicConnectionState::Established,
            QuicConnectionState::Closing,
            QuicConnectionState::Closed,
            QuicConnectionState::Failed,
        ];

        for (i, state1) in states.iter().enumerate() {
            for (j, state2) in states.iter().enumerate() {
                if i == j {
                    assert_eq!(state1, state2);
                } else {
                    assert_ne!(state1, state2);
                }
            }
        }
    }
}
