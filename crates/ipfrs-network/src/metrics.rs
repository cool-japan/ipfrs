//! Network metrics collection and reporting
//!
//! This module provides comprehensive network metrics tracking:
//! - Connection metrics (established, failed, duration)
//! - Bandwidth metrics (bytes sent/received)
//! - DHT metrics (queries, providers)
//! - Protocol metrics (messages sent/received by type)
//! - Prometheus export support

use parking_lot::RwLock;
use prometheus::{Encoder, Opts, Registry, TextEncoder};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Network metrics collector
pub struct NetworkMetrics {
    /// Connection metrics
    connections: ConnectionMetrics,
    /// Bandwidth metrics
    bandwidth: BandwidthMetrics,
    /// DHT metrics
    dht: DhtMetrics,
    /// Protocol metrics
    protocols: ProtocolMetrics,
    /// Start time for uptime calculation
    start_time: Instant,
}

impl NetworkMetrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            connections: ConnectionMetrics::new(),
            bandwidth: BandwidthMetrics::new(),
            dht: DhtMetrics::new(),
            protocols: ProtocolMetrics::new(),
            start_time: Instant::now(),
        }
    }

    /// Get connection metrics
    pub fn connections(&self) -> &ConnectionMetrics {
        &self.connections
    }

    /// Get bandwidth metrics
    pub fn bandwidth(&self) -> &BandwidthMetrics {
        &self.bandwidth
    }

    /// Get DHT metrics
    pub fn dht(&self) -> &DhtMetrics {
        &self.dht
    }

    /// Get protocol metrics
    pub fn protocols(&self) -> &ProtocolMetrics {
        &self.protocols
    }

    /// Get uptime duration
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get a complete metrics snapshot
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            uptime_secs: self.start_time.elapsed().as_secs(),
            connections: self.connections.snapshot(),
            bandwidth: self.bandwidth.snapshot(),
            dht: self.dht.snapshot(),
        }
    }

    /// Create and populate a Prometheus registry with all metrics
    pub fn create_prometheus_registry(&self) -> Result<Registry, prometheus::Error> {
        let registry = Registry::new();

        // Connection metrics
        let connections_established = prometheus::IntCounterVec::new(
            Opts::new(
                "ipfrs_connections_established_total",
                "Total number of connections established",
            ),
            &["direction"],
        )?;
        connections_established
            .with_label_values(&["inbound"])
            .inc_by(self.connections.total_inbound());
        connections_established
            .with_label_values(&["outbound"])
            .inc_by(self.connections.total_outbound());
        registry.register(Box::new(connections_established))?;

        let connections_failed = prometheus::IntCounter::new(
            "ipfrs_connections_failed_total",
            "Total number of failed connection attempts",
        )?;
        connections_failed.inc_by(self.connections.total_failed());
        registry.register(Box::new(connections_failed))?;

        let connections_active = prometheus::IntGauge::new(
            "ipfrs_connections_active",
            "Number of currently active connections",
        )?;
        connections_active.set(self.connections.active() as i64);
        registry.register(Box::new(connections_active))?;

        // Bandwidth metrics
        let bytes_sent = prometheus::IntCounter::new(
            "ipfrs_bytes_sent_total",
            "Total bytes sent over the network",
        )?;
        bytes_sent.inc_by(self.bandwidth.total_sent());
        registry.register(Box::new(bytes_sent))?;

        let bytes_received = prometheus::IntCounter::new(
            "ipfrs_bytes_received_total",
            "Total bytes received from the network",
        )?;
        bytes_received.inc_by(self.bandwidth.total_received());
        registry.register(Box::new(bytes_received))?;

        // DHT metrics
        let dht_queries = prometheus::IntCounterVec::new(
            Opts::new("ipfrs_dht_queries_total", "Total DHT queries by status"),
            &["status"],
        )?;
        let dht_snapshot = self.dht.snapshot();
        dht_queries
            .with_label_values(&["success"])
            .inc_by(dht_snapshot.queries_successful);
        dht_queries
            .with_label_values(&["failed"])
            .inc_by(dht_snapshot.queries_failed);
        registry.register(Box::new(dht_queries))?;

        let providers_published = prometheus::IntCounter::new(
            "ipfrs_dht_providers_published_total",
            "Total provider records published to DHT",
        )?;
        providers_published.inc_by(dht_snapshot.providers_published);
        registry.register(Box::new(providers_published))?;

        let providers_found = prometheus::IntCounter::new(
            "ipfrs_dht_providers_found_total",
            "Total providers found via DHT queries",
        )?;
        providers_found.inc_by(dht_snapshot.providers_found);
        registry.register(Box::new(providers_found))?;

        let routing_table_size = prometheus::IntGauge::new(
            "ipfrs_dht_routing_table_size",
            "Current DHT routing table size",
        )?;
        routing_table_size.set(dht_snapshot.routing_table_size as i64);
        registry.register(Box::new(routing_table_size))?;

        // Uptime
        let uptime = prometheus::IntGauge::new("ipfrs_uptime_seconds", "Node uptime in seconds")?;
        uptime.set(self.uptime().as_secs() as i64);
        registry.register(Box::new(uptime))?;

        Ok(registry)
    }

    /// Export metrics in Prometheus text format
    pub fn export_prometheus(&self) -> Result<String, prometheus::Error> {
        let registry = self.create_prometheus_registry()?;
        let encoder = TextEncoder::new();
        let metric_families = registry.gather();

        let mut buffer = Vec::new();
        encoder
            .encode(&metric_families, &mut buffer)
            .map_err(|e| prometheus::Error::Msg(e.to_string()))?;

        String::from_utf8(buffer).map_err(|e| prometheus::Error::Msg(e.to_string()))
    }
}

impl Default for NetworkMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Connection metrics
pub struct ConnectionMetrics {
    /// Total connections established
    connections_established: AtomicU64,
    /// Total connections failed
    connections_failed: AtomicU64,
    /// Currently active connections
    active_connections: AtomicU64,
    /// Total inbound connections
    inbound_connections: AtomicU64,
    /// Total outbound connections
    outbound_connections: AtomicU64,
    /// Connection durations for averaging
    connection_durations: RwLock<Vec<Duration>>,
}

impl ConnectionMetrics {
    fn new() -> Self {
        Self {
            connections_established: AtomicU64::new(0),
            connections_failed: AtomicU64::new(0),
            active_connections: AtomicU64::new(0),
            inbound_connections: AtomicU64::new(0),
            outbound_connections: AtomicU64::new(0),
            connection_durations: RwLock::new(Vec::new()),
        }
    }

    /// Record a connection established
    pub fn connection_established(&self, inbound: bool) {
        self.connections_established.fetch_add(1, Ordering::Relaxed);
        self.active_connections.fetch_add(1, Ordering::Relaxed);
        if inbound {
            self.inbound_connections.fetch_add(1, Ordering::Relaxed);
        } else {
            self.outbound_connections.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record a connection closed
    pub fn connection_closed(&self, duration: Duration) {
        self.active_connections.fetch_sub(1, Ordering::Relaxed);

        let mut durations = self.connection_durations.write();
        // Keep last 1000 durations for averaging
        if durations.len() >= 1000 {
            durations.remove(0);
        }
        durations.push(duration);
    }

    /// Record a connection failure
    pub fn connection_failed(&self) {
        self.connections_failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Get total connections established
    pub fn total_established(&self) -> u64 {
        self.connections_established.load(Ordering::Relaxed)
    }

    /// Get total connections failed
    pub fn total_failed(&self) -> u64 {
        self.connections_failed.load(Ordering::Relaxed)
    }

    /// Get active connection count
    pub fn active(&self) -> u64 {
        self.active_connections.load(Ordering::Relaxed)
    }

    /// Get inbound connection count
    pub fn total_inbound(&self) -> u64 {
        self.inbound_connections.load(Ordering::Relaxed)
    }

    /// Get outbound connection count
    pub fn total_outbound(&self) -> u64 {
        self.outbound_connections.load(Ordering::Relaxed)
    }

    /// Get average connection duration
    pub fn avg_duration(&self) -> Option<Duration> {
        let durations = self.connection_durations.read();
        if durations.is_empty() {
            None
        } else {
            let total: Duration = durations.iter().sum();
            Some(total / durations.len() as u32)
        }
    }

    /// Get snapshot
    pub fn snapshot(&self) -> ConnectionMetricsSnapshot {
        ConnectionMetricsSnapshot {
            total_established: self.total_established(),
            total_failed: self.total_failed(),
            active: self.active(),
            total_inbound: self.total_inbound(),
            total_outbound: self.total_outbound(),
            avg_duration_ms: self.avg_duration().map(|d| d.as_millis() as u64),
        }
    }
}

/// Bandwidth metrics
pub struct BandwidthMetrics {
    /// Total bytes sent
    bytes_sent: AtomicU64,
    /// Total bytes received
    bytes_received: AtomicU64,
    /// Per-protocol bandwidth
    protocol_bandwidth: RwLock<HashMap<String, (u64, u64)>>,
}

impl BandwidthMetrics {
    fn new() -> Self {
        Self {
            bytes_sent: AtomicU64::new(0),
            bytes_received: AtomicU64::new(0),
            protocol_bandwidth: RwLock::new(HashMap::new()),
        }
    }

    /// Record bytes sent
    pub fn record_sent(&self, bytes: u64) {
        self.bytes_sent.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record bytes received
    pub fn record_received(&self, bytes: u64) {
        self.bytes_received.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record protocol-specific bandwidth
    pub fn record_protocol_traffic(&self, protocol: &str, sent: u64, received: u64) {
        let mut bandwidth = self.protocol_bandwidth.write();
        let entry = bandwidth.entry(protocol.to_string()).or_insert((0, 0));
        entry.0 += sent;
        entry.1 += received;
    }

    /// Get total bytes sent
    pub fn total_sent(&self) -> u64 {
        self.bytes_sent.load(Ordering::Relaxed)
    }

    /// Get total bytes received
    pub fn total_received(&self) -> u64 {
        self.bytes_received.load(Ordering::Relaxed)
    }

    /// Get snapshot
    pub fn snapshot(&self) -> BandwidthMetricsSnapshot {
        BandwidthMetricsSnapshot {
            total_sent: self.total_sent(),
            total_received: self.total_received(),
        }
    }
}

/// DHT metrics
pub struct DhtMetrics {
    /// Total DHT queries made
    queries_made: AtomicU64,
    /// Successful DHT queries
    queries_successful: AtomicU64,
    /// Failed DHT queries
    queries_failed: AtomicU64,
    /// Provider records published
    providers_published: AtomicU64,
    /// Provider queries made
    provider_queries: AtomicU64,
    /// Providers found
    providers_found: AtomicU64,
    /// Routing table size
    routing_table_size: AtomicU64,
}

impl DhtMetrics {
    fn new() -> Self {
        Self {
            queries_made: AtomicU64::new(0),
            queries_successful: AtomicU64::new(0),
            queries_failed: AtomicU64::new(0),
            providers_published: AtomicU64::new(0),
            provider_queries: AtomicU64::new(0),
            providers_found: AtomicU64::new(0),
            routing_table_size: AtomicU64::new(0),
        }
    }

    /// Record a DHT query
    pub fn query_made(&self) {
        self.queries_made.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a successful query
    pub fn query_successful(&self) {
        self.queries_successful.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a failed query
    pub fn query_failed(&self) {
        self.queries_failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record provider published
    pub fn provider_published(&self) {
        self.providers_published.fetch_add(1, Ordering::Relaxed);
    }

    /// Record provider query
    pub fn provider_query(&self) {
        self.provider_queries.fetch_add(1, Ordering::Relaxed);
    }

    /// Record providers found
    pub fn providers_found(&self, count: u64) {
        self.providers_found.fetch_add(count, Ordering::Relaxed);
    }

    /// Update routing table size
    pub fn set_routing_table_size(&self, size: u64) {
        self.routing_table_size.store(size, Ordering::Relaxed);
    }

    /// Get snapshot
    pub fn snapshot(&self) -> DhtMetricsSnapshot {
        DhtMetricsSnapshot {
            queries_made: self.queries_made.load(Ordering::Relaxed),
            queries_successful: self.queries_successful.load(Ordering::Relaxed),
            queries_failed: self.queries_failed.load(Ordering::Relaxed),
            providers_published: self.providers_published.load(Ordering::Relaxed),
            provider_queries: self.provider_queries.load(Ordering::Relaxed),
            providers_found: self.providers_found.load(Ordering::Relaxed),
            routing_table_size: self.routing_table_size.load(Ordering::Relaxed),
        }
    }
}

/// Protocol metrics
pub struct ProtocolMetrics {
    /// Messages per protocol
    messages: RwLock<HashMap<String, ProtocolStats>>,
}

#[derive(Default, Clone)]
struct ProtocolStats {
    messages_sent: u64,
    messages_received: u64,
    bytes_sent: u64,
    bytes_received: u64,
}

impl ProtocolMetrics {
    fn new() -> Self {
        Self {
            messages: RwLock::new(HashMap::new()),
        }
    }

    /// Record message sent
    pub fn message_sent(&self, protocol: &str, bytes: u64) {
        let mut messages = self.messages.write();
        let stats = messages.entry(protocol.to_string()).or_default();
        stats.messages_sent += 1;
        stats.bytes_sent += bytes;
    }

    /// Record message received
    pub fn message_received(&self, protocol: &str, bytes: u64) {
        let mut messages = self.messages.write();
        let stats = messages.entry(protocol.to_string()).or_default();
        stats.messages_received += 1;
        stats.bytes_received += bytes;
    }

    /// Get protocol stats
    pub fn get_stats(&self, protocol: &str) -> Option<(u64, u64, u64, u64)> {
        let messages = self.messages.read();
        messages.get(protocol).map(|s| {
            (
                s.messages_sent,
                s.messages_received,
                s.bytes_sent,
                s.bytes_received,
            )
        })
    }
}

/// Complete metrics snapshot for serialization
#[derive(Debug, Clone, Serialize)]
pub struct MetricsSnapshot {
    /// Uptime in seconds
    pub uptime_secs: u64,
    /// Connection metrics
    pub connections: ConnectionMetricsSnapshot,
    /// Bandwidth metrics
    pub bandwidth: BandwidthMetricsSnapshot,
    /// DHT metrics
    pub dht: DhtMetricsSnapshot,
}

/// Connection metrics snapshot
#[derive(Debug, Clone, Serialize)]
pub struct ConnectionMetricsSnapshot {
    /// Total connections established
    pub total_established: u64,
    /// Total connections failed
    pub total_failed: u64,
    /// Currently active connections
    pub active: u64,
    /// Total inbound connections
    pub total_inbound: u64,
    /// Total outbound connections
    pub total_outbound: u64,
    /// Average connection duration in milliseconds
    pub avg_duration_ms: Option<u64>,
}

/// Bandwidth metrics snapshot
#[derive(Debug, Clone, Serialize)]
pub struct BandwidthMetricsSnapshot {
    /// Total bytes sent
    pub total_sent: u64,
    /// Total bytes received
    pub total_received: u64,
}

/// DHT metrics snapshot
#[derive(Debug, Clone, Serialize)]
pub struct DhtMetricsSnapshot {
    /// Total queries made
    pub queries_made: u64,
    /// Successful queries
    pub queries_successful: u64,
    /// Failed queries
    pub queries_failed: u64,
    /// Providers published
    pub providers_published: u64,
    /// Provider queries made
    pub provider_queries: u64,
    /// Total providers found
    pub providers_found: u64,
    /// Current routing table size
    pub routing_table_size: u64,
}

/// Thread-safe metrics handle
pub type SharedMetrics = Arc<NetworkMetrics>;

/// Create a new shared metrics instance
pub fn new_shared_metrics() -> SharedMetrics {
    Arc::new(NetworkMetrics::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_metrics() {
        let metrics = NetworkMetrics::new();

        metrics.connections.connection_established(true);
        metrics.connections.connection_established(false);
        assert_eq!(metrics.connections.active(), 2);
        assert_eq!(metrics.connections.total_established(), 2);
        assert_eq!(metrics.connections.total_inbound(), 1);
        assert_eq!(metrics.connections.total_outbound(), 1);

        metrics
            .connections
            .connection_closed(Duration::from_secs(10));
        assert_eq!(metrics.connections.active(), 1);

        metrics.connections.connection_failed();
        assert_eq!(metrics.connections.total_failed(), 1);
    }

    #[test]
    fn test_bandwidth_metrics() {
        let metrics = NetworkMetrics::new();

        metrics.bandwidth.record_sent(1000);
        metrics.bandwidth.record_received(2000);

        assert_eq!(metrics.bandwidth.total_sent(), 1000);
        assert_eq!(metrics.bandwidth.total_received(), 2000);
    }

    #[test]
    fn test_dht_metrics() {
        let metrics = NetworkMetrics::new();

        metrics.dht.query_made();
        metrics.dht.query_successful();
        metrics.dht.query_made();
        metrics.dht.query_failed();
        metrics.dht.providers_found(5);

        let snapshot = metrics.dht.snapshot();
        assert_eq!(snapshot.queries_made, 2);
        assert_eq!(snapshot.queries_successful, 1);
        assert_eq!(snapshot.queries_failed, 1);
        assert_eq!(snapshot.providers_found, 5);
    }

    #[test]
    fn test_protocol_metrics() {
        let metrics = NetworkMetrics::new();

        metrics.protocols.message_sent("/ipfs/kad/1.0.0", 100);
        metrics.protocols.message_received("/ipfs/kad/1.0.0", 200);

        let stats = metrics.protocols.get_stats("/ipfs/kad/1.0.0");
        assert!(stats.is_some());
        let (sent, received, bytes_sent, bytes_received) =
            stats.expect("test: protocol stats should be present after recording messages");
        assert_eq!(sent, 1);
        assert_eq!(received, 1);
        assert_eq!(bytes_sent, 100);
        assert_eq!(bytes_received, 200);
    }

    #[test]
    fn test_metrics_snapshot() {
        let metrics = NetworkMetrics::new();

        metrics.connections.connection_established(true);
        metrics.bandwidth.record_sent(100);
        metrics.dht.query_made();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.connections.active, 1);
        assert_eq!(snapshot.bandwidth.total_sent, 100);
        assert_eq!(snapshot.dht.queries_made, 1);
    }
}
