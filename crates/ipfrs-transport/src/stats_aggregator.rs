//! Statistics aggregation and analysis utilities
//!
//! This module provides tools to aggregate and analyze performance statistics
//! from multiple transport components for comprehensive monitoring.

use crate::{
    BitswapStats, ContentRoutingStats, EdgeStats, MulticastStats, NatTraversalStats,
    PartitionStats, PeerManagerStats, PrefetchStats, QuicPoolStats, RecoveryStats, SessionStats,
    TensorSwapStats, ThrottleStats, TransportStats,
};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Aggregated statistics from all transport components
#[derive(Debug, Clone)]
pub struct AggregatedStats {
    /// Timestamp when stats were collected
    pub timestamp: SystemTime,
    /// Duration of the monitoring period
    pub period: Duration,
    /// Peer manager statistics
    pub peer_stats: Option<PeerManagerStats>,
    /// Session statistics (aggregated from multiple sessions)
    pub session_stats: Option<AggregatedSessionStats>,
    /// Bitswap statistics
    pub bitswap_stats: Option<BitswapStats>,
    /// TensorSwap statistics
    pub tensorswap_stats: Option<TensorSwapStats>,
    /// QUIC pool statistics
    pub quic_stats: Option<QuicPoolStats>,
    /// Transport statistics (aggregated from multiple transports)
    pub transport_stats: Option<AggregatedTransportStats>,
    /// Prefetch statistics
    pub prefetch_stats: Option<PrefetchStats>,
    /// Content routing statistics
    pub content_routing_stats: Option<ContentRoutingStats>,
    /// CDN edge statistics
    pub edge_stats: Option<EdgeStats>,
    /// Multicast statistics
    pub multicast_stats: Option<MulticastStats>,
    /// NAT traversal statistics
    pub nat_stats: Option<NatTraversalStats>,
    /// Partition detection statistics
    pub partition_stats: Option<PartitionStats>,
    /// Recovery statistics
    pub recovery_stats: Option<RecoveryStats>,
    /// Throttle statistics
    pub throttle_stats: Option<ThrottleStats>,
    /// Overall performance metrics
    pub performance: PerformanceMetrics,
}

/// Aggregated session statistics from multiple sessions
#[derive(Debug, Clone)]
pub struct AggregatedSessionStats {
    /// Total number of sessions
    pub total_sessions: usize,
    /// Active sessions
    pub active_sessions: usize,
    /// Completed sessions
    pub completed_sessions: usize,
    /// Failed sessions
    pub failed_sessions: usize,
    /// Total blocks across all sessions
    pub total_blocks: usize,
    /// Total received blocks
    pub total_received: usize,
    /// Total bytes transferred
    pub total_bytes: u64,
    /// Average session completion time
    pub avg_completion_time: Duration,
    /// Average throughput across all sessions (bytes/sec)
    pub avg_throughput: u64,
}

/// Aggregated transport statistics from multiple transports
#[derive(Debug, Clone)]
pub struct AggregatedTransportStats {
    /// Total number of connections
    pub total_connections: usize,
    /// Active connections
    pub active_connections: usize,
    /// Total messages sent
    pub messages_sent: u64,
    /// Total messages received
    pub messages_received: u64,
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Transport type distribution
    pub transport_types: HashMap<String, usize>,
    /// Average latency across all transports
    pub avg_latency: Option<Duration>,
}

/// Overall performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total throughput (bytes/sec)
    pub total_throughput: u64,
    /// Average latency
    pub avg_latency: Option<Duration>,
    /// Request success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
    /// Peer utilization (0.0 to 1.0)
    pub peer_utilization: f64,
    /// Network efficiency score (0.0 to 1.0)
    pub efficiency_score: f64,
}

/// Time series data point for trend analysis
#[derive(Debug, Clone)]
pub struct DataPoint {
    /// Timestamp of the measurement
    pub timestamp: SystemTime,
    /// Value at this point
    pub value: f64,
}

/// Statistics collector for gathering stats over time
pub struct StatsCollector {
    /// Historical data points
    history: Vec<(SystemTime, AggregatedStats)>,
    /// Maximum history size
    max_history: usize,
}

impl StatsCollector {
    /// Create a new stats collector
    pub fn new(max_history: usize) -> Self {
        Self {
            history: Vec::with_capacity(max_history),
            max_history,
        }
    }

    /// Record a stats snapshot
    pub fn record(&mut self, stats: AggregatedStats) {
        let timestamp = stats.timestamp;
        self.history.push((timestamp, stats));

        // Remove old entries if exceeding max
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Get the latest stats
    pub fn latest(&self) -> Option<&AggregatedStats> {
        self.history.last().map(|(_, stats)| stats)
    }

    /// Get throughput trend over time
    pub fn throughput_trend(&self) -> Vec<DataPoint> {
        self.history
            .iter()
            .map(|(ts, stats)| DataPoint {
                timestamp: *ts,
                value: stats.performance.total_throughput as f64,
            })
            .collect()
    }

    /// Get latency trend over time
    pub fn latency_trend(&self) -> Vec<DataPoint> {
        self.history
            .iter()
            .filter_map(|(ts, stats)| {
                stats.performance.avg_latency.map(|lat| DataPoint {
                    timestamp: *ts,
                    value: lat.as_secs_f64(),
                })
            })
            .collect()
    }

    /// Get success rate trend over time
    pub fn success_rate_trend(&self) -> Vec<DataPoint> {
        self.history
            .iter()
            .map(|(ts, stats)| DataPoint {
                timestamp: *ts,
                value: stats.performance.success_rate,
            })
            .collect()
    }

    /// Calculate average throughput over the history
    pub fn avg_throughput(&self) -> u64 {
        if self.history.is_empty() {
            return 0;
        }

        let sum: u64 = self
            .history
            .iter()
            .map(|(_, stats)| stats.performance.total_throughput)
            .sum();
        sum / self.history.len() as u64
    }

    /// Calculate average latency over the history
    pub fn avg_latency(&self) -> Option<Duration> {
        let latencies: Vec<Duration> = self
            .history
            .iter()
            .filter_map(|(_, stats)| stats.performance.avg_latency)
            .collect();

        if latencies.is_empty() {
            return None;
        }

        let sum: Duration = latencies.iter().sum();
        Some(sum / latencies.len() as u32)
    }

    /// Detect anomalies in throughput
    pub fn detect_throughput_anomalies(&self, threshold: f64) -> Vec<(SystemTime, f64)> {
        if self.history.len() < 3 {
            return Vec::new();
        }

        let avg = self.avg_throughput() as f64;
        let mut anomalies = Vec::new();

        for (ts, stats) in &self.history {
            let value = stats.performance.total_throughput as f64;
            let deviation = (value - avg).abs() / avg;

            if deviation > threshold {
                anomalies.push((*ts, deviation));
            }
        }

        anomalies
    }

    /// Clear all history
    pub fn clear(&mut self) {
        self.history.clear();
    }

    /// Get the number of recorded snapshots
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Check if the collector is empty
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }
}

/// Builder for creating aggregated statistics
pub struct AggregatedStatsBuilder {
    timestamp: SystemTime,
    period: Duration,
    peer_stats: Option<PeerManagerStats>,
    session_stats: Vec<SessionStats>,
    bitswap_stats: Option<BitswapStats>,
    tensorswap_stats: Option<TensorSwapStats>,
    quic_stats: Option<QuicPoolStats>,
    transport_stats: Vec<TransportStats>,
    prefetch_stats: Option<PrefetchStats>,
    content_routing_stats: Option<ContentRoutingStats>,
    edge_stats: Option<EdgeStats>,
    multicast_stats: Option<MulticastStats>,
    nat_stats: Option<NatTraversalStats>,
    partition_stats: Option<PartitionStats>,
    recovery_stats: Option<RecoveryStats>,
    throttle_stats: Option<ThrottleStats>,
}

impl AggregatedStatsBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            timestamp: SystemTime::now(),
            period: Duration::from_secs(0),
            peer_stats: None,
            session_stats: Vec::new(),
            bitswap_stats: None,
            tensorswap_stats: None,
            quic_stats: None,
            transport_stats: Vec::new(),
            prefetch_stats: None,
            content_routing_stats: None,
            edge_stats: None,
            multicast_stats: None,
            nat_stats: None,
            partition_stats: None,
            recovery_stats: None,
            throttle_stats: None,
        }
    }

    /// Set monitoring period
    pub fn period(mut self, period: Duration) -> Self {
        self.period = period;
        self
    }

    /// Add peer manager stats
    pub fn peer_stats(mut self, stats: PeerManagerStats) -> Self {
        self.peer_stats = Some(stats);
        self
    }

    /// Add session stats
    pub fn add_session_stats(mut self, stats: SessionStats) -> Self {
        self.session_stats.push(stats);
        self
    }

    /// Add bitswap stats
    pub fn bitswap_stats(mut self, stats: BitswapStats) -> Self {
        self.bitswap_stats = Some(stats);
        self
    }

    /// Add tensorswap stats
    pub fn tensorswap_stats(mut self, stats: TensorSwapStats) -> Self {
        self.tensorswap_stats = Some(stats);
        self
    }

    /// Add QUIC stats
    pub fn quic_stats(mut self, stats: QuicPoolStats) -> Self {
        self.quic_stats = Some(stats);
        self
    }

    /// Add transport stats
    pub fn add_transport_stats(mut self, stats: TransportStats) -> Self {
        self.transport_stats.push(stats);
        self
    }

    /// Add prefetch stats
    pub fn prefetch_stats(mut self, stats: PrefetchStats) -> Self {
        self.prefetch_stats = Some(stats);
        self
    }

    /// Add content routing stats
    pub fn content_routing_stats(mut self, stats: ContentRoutingStats) -> Self {
        self.content_routing_stats = Some(stats);
        self
    }

    /// Add edge stats
    pub fn edge_stats(mut self, stats: EdgeStats) -> Self {
        self.edge_stats = Some(stats);
        self
    }

    /// Add multicast stats
    pub fn multicast_stats(mut self, stats: MulticastStats) -> Self {
        self.multicast_stats = Some(stats);
        self
    }

    /// Add NAT traversal stats
    pub fn nat_stats(mut self, stats: NatTraversalStats) -> Self {
        self.nat_stats = Some(stats);
        self
    }

    /// Add partition stats
    pub fn partition_stats(mut self, stats: PartitionStats) -> Self {
        self.partition_stats = Some(stats);
        self
    }

    /// Add recovery stats
    pub fn recovery_stats(mut self, stats: RecoveryStats) -> Self {
        self.recovery_stats = Some(stats);
        self
    }

    /// Add throttle stats
    pub fn throttle_stats(mut self, stats: ThrottleStats) -> Self {
        self.throttle_stats = Some(stats);
        self
    }

    /// Build the aggregated stats
    pub fn build(self) -> AggregatedStats {
        let session_stats = self.aggregate_session_stats();
        let transport_stats = self.aggregate_transport_stats();
        let performance = self.calculate_performance_metrics();

        AggregatedStats {
            timestamp: self.timestamp,
            period: self.period,
            peer_stats: self.peer_stats,
            session_stats,
            bitswap_stats: self.bitswap_stats,
            tensorswap_stats: self.tensorswap_stats,
            quic_stats: self.quic_stats,
            transport_stats,
            prefetch_stats: self.prefetch_stats,
            content_routing_stats: self.content_routing_stats,
            edge_stats: self.edge_stats,
            multicast_stats: self.multicast_stats,
            nat_stats: self.nat_stats,
            partition_stats: self.partition_stats,
            recovery_stats: self.recovery_stats,
            throttle_stats: self.throttle_stats,
            performance,
        }
    }

    fn aggregate_session_stats(&self) -> Option<AggregatedSessionStats> {
        if self.session_stats.is_empty() {
            return None;
        }

        let mut active = 0;
        let mut completed = 0;
        let mut failed = 0;
        let mut total_blocks = 0;
        let mut total_received = 0;
        let mut total_bytes = 0;
        let mut total_time = Duration::ZERO;
        let mut total_throughput = 0u64;

        for stats in &self.session_stats {
            total_blocks += stats.total_blocks;
            total_received += stats.blocks_received;
            total_bytes += stats.bytes_transferred;

            // Calculate elapsed time from start/end timestamps
            if let (Some(start), Some(end)) = (stats.started_at, stats.completed_at) {
                let elapsed = end.duration_since(start);
                total_time += elapsed;
                if elapsed.as_secs() > 0 {
                    total_throughput += stats.bytes_transferred / elapsed.as_secs();
                }
            } else if let Some(start) = stats.started_at {
                let elapsed = std::time::Instant::now().duration_since(start);
                total_time += elapsed;
                if elapsed.as_secs() > 0 {
                    total_throughput += stats.bytes_transferred / elapsed.as_secs();
                }
            }

            // Count state (simplified - would need actual state info)
            if stats.blocks_received == stats.total_blocks && stats.total_blocks > 0 {
                completed += 1;
            } else if stats.blocks_received > 0 {
                active += 1;
            }
            failed += stats.blocks_failed;
        }

        let avg_completion_time = if !self.session_stats.is_empty() {
            total_time / self.session_stats.len() as u32
        } else {
            Duration::ZERO
        };

        let avg_throughput = if !self.session_stats.is_empty() {
            total_throughput / self.session_stats.len() as u64
        } else {
            0
        };

        Some(AggregatedSessionStats {
            total_sessions: self.session_stats.len(),
            active_sessions: active,
            completed_sessions: completed,
            failed_sessions: failed,
            total_blocks,
            total_received,
            total_bytes,
            avg_completion_time,
            avg_throughput,
        })
    }

    fn aggregate_transport_stats(&self) -> Option<AggregatedTransportStats> {
        if self.transport_stats.is_empty() {
            return None;
        }

        let mut total_connections = 0;
        let mut active_connections = 0;
        let mut messages_sent = 0;
        let mut messages_received = 0;
        let mut bytes_sent = 0;
        let mut bytes_received = 0;
        let mut transport_types = HashMap::new();
        let mut latencies = Vec::new();

        for stats in &self.transport_stats {
            total_connections += stats.connections_established;
            active_connections += stats.active_connections;
            // TransportStats doesn't have messages_sent/received, using bytes as proxy
            messages_sent += stats.bytes_sent / 1024; // Rough estimate: 1 message per KB
            messages_received += stats.bytes_received / 1024;
            bytes_sent += stats.bytes_sent;
            bytes_received += stats.bytes_received;

            // Note: TransportStats doesn't have a transport_type field
            // This would need to be tracked separately if needed
            *transport_types.entry("Unknown".to_string()).or_insert(0) += 1;

            if let Some(lat) = stats.avg_rtt {
                latencies.push(lat);
            }
        }

        let avg_latency = if !latencies.is_empty() {
            let sum: Duration = latencies.iter().sum();
            Some(sum / latencies.len() as u32)
        } else {
            None
        };

        Some(AggregatedTransportStats {
            total_connections: total_connections as usize,
            active_connections,
            messages_sent,
            messages_received,
            bytes_sent,
            bytes_received,
            transport_types,
            avg_latency,
        })
    }

    fn calculate_performance_metrics(&self) -> PerformanceMetrics {
        let total_throughput = if self.period.as_secs() > 0 {
            self.session_stats
                .iter()
                .map(|s| s.bytes_transferred)
                .sum::<u64>()
                / self.period.as_secs()
        } else {
            0
        };

        let avg_latency = self
            .peer_stats
            .as_ref()
            .map(|s| Duration::from_millis(s.avg_latency_ms as u64));

        let success_rate = if let Some(bs) = &self.bitswap_stats {
            let total = bs.total_requests;
            if total > 0 {
                bs.completed_requests as f64 / total as f64
            } else {
                1.0
            }
        } else {
            1.0
        };

        let cache_hit_rate = if let Some(pr) = &self.prefetch_stats {
            pr.hit_rate
        } else {
            0.0
        };

        let peer_utilization = if let Some(pm) = &self.peer_stats {
            if pm.total_peers > 0 {
                pm.connected_peers as f64 / pm.total_peers as f64
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Efficiency score combines multiple factors
        let efficiency_score = (success_rate + cache_hit_rate + peer_utilization) / 3.0;

        PerformanceMetrics {
            total_throughput,
            avg_latency,
            success_rate,
            cache_hit_rate,
            peer_utilization,
            efficiency_score,
        }
    }
}

impl Default for AggregatedStatsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_collector_creation() {
        let collector = StatsCollector::new(100);
        assert_eq!(collector.len(), 0);
        assert!(collector.is_empty());
        assert!(collector.latest().is_none());
    }

    #[test]
    fn test_stats_collector_record() {
        let mut collector = StatsCollector::new(100);
        let stats = AggregatedStatsBuilder::new().build();

        collector.record(stats.clone());
        assert_eq!(collector.len(), 1);
        assert!(!collector.is_empty());
        assert!(collector.latest().is_some());
    }

    #[test]
    fn test_stats_collector_max_history() {
        let mut collector = StatsCollector::new(5);

        for _ in 0..10 {
            let stats = AggregatedStatsBuilder::new().build();
            collector.record(stats);
        }

        assert_eq!(collector.len(), 5);
    }

    #[test]
    fn test_aggregated_stats_builder() {
        let builder = AggregatedStatsBuilder::new();
        let stats = builder.period(Duration::from_secs(60)).build();

        assert_eq!(stats.period, Duration::from_secs(60));
        assert!(stats.peer_stats.is_none());
        assert!(stats.session_stats.is_none());
    }

    #[test]
    fn test_performance_metrics_default() {
        let builder = AggregatedStatsBuilder::new();
        let stats = builder.build();

        assert_eq!(stats.performance.total_throughput, 0);
        assert!(stats.performance.avg_latency.is_none());
        assert_eq!(stats.performance.success_rate, 1.0);
    }

    #[test]
    fn test_throughput_trend() {
        let mut collector = StatsCollector::new(100);

        for i in 0..5 {
            let mut builder = AggregatedStatsBuilder::new();
            builder.period = Duration::from_secs(1);

            let mut stats = builder.build();
            stats.performance.total_throughput = (i + 1) * 1000;

            collector.record(stats);
        }

        let trend = collector.throughput_trend();
        assert_eq!(trend.len(), 5);
        assert_eq!(trend[0].value, 1000.0);
        assert_eq!(trend[4].value, 5000.0);
    }

    #[test]
    fn test_avg_throughput() {
        let mut collector = StatsCollector::new(100);

        for _ in 0..5 {
            let mut builder = AggregatedStatsBuilder::new();
            builder.period = Duration::from_secs(1);

            let mut stats = builder.build();
            stats.performance.total_throughput = 1000;

            collector.record(stats);
        }

        assert_eq!(collector.avg_throughput(), 1000);
    }

    #[test]
    fn test_clear() {
        let mut collector = StatsCollector::new(100);
        let stats = AggregatedStatsBuilder::new().build();
        collector.record(stats);

        assert_eq!(collector.len(), 1);
        collector.clear();
        assert_eq!(collector.len(), 0);
    }

    #[test]
    fn test_detect_anomalies() {
        let mut collector = StatsCollector::new(100);

        // Add normal values
        for _ in 0..5 {
            let mut stats = AggregatedStatsBuilder::new().build();
            stats.performance.total_throughput = 1000;
            collector.record(stats);
        }

        // Add anomaly
        let mut stats = AggregatedStatsBuilder::new().build();
        stats.performance.total_throughput = 5000;
        collector.record(stats);

        let anomalies = collector.detect_throughput_anomalies(0.5); // 50% threshold
        assert!(!anomalies.is_empty());
    }
}
