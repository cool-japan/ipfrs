//! Cluster coordinator for multi-node RAFT deployments.
//!
//! Provides cluster management, node discovery, and health monitoring
//! for distributed RAFT-based block storage.
//!
//! # Example
//!
//! ```rust,ignore
//! use ipfrs_storage::cluster::{ClusterConfig, ClusterCoordinator};
//!
//! let config = ClusterConfig::default();
//! let coordinator = ClusterCoordinator::new(config);
//! coordinator.add_node(node_id, address).await?;
//! ```

use crate::raft::{NodeId, NodeState};
use dashmap::DashMap;
use ipfrs_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;

/// Cluster configuration
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    /// Heartbeat interval in milliseconds
    pub heartbeat_interval_ms: u64,
    /// Node failure threshold (missed heartbeats)
    pub failure_threshold: u32,
    /// Minimum cluster size for quorum
    pub min_cluster_size: usize,
    /// Maximum cluster size
    pub max_cluster_size: usize,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval_ms: 1000, // 1 second
            failure_threshold: 3,        // 3 missed heartbeats
            min_cluster_size: 3,         // Minimum 3 nodes for fault tolerance
            max_cluster_size: 100,       // Maximum 100 nodes
        }
    }
}

/// Node metadata and health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    /// Node identifier
    pub node_id: NodeId,
    /// Network address
    pub address: SocketAddr,
    /// Current RAFT state
    pub state: NodeState,
    /// Last heartbeat timestamp
    pub last_heartbeat: SystemTime,
    /// Node health status
    pub health: NodeHealth,
    /// Number of missed heartbeats
    pub missed_heartbeats: u32,
}

/// Node health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeHealth {
    /// Node is healthy and responsive
    Healthy,
    /// Node is degraded (slow responses)
    Degraded,
    /// Node is suspected to be down
    Suspected,
    /// Node is confirmed down
    Down,
}

/// Type alias for failover callback
type FailoverCallback = Arc<RwLock<Option<Box<dyn Fn(NodeId) + Send + Sync>>>>;

/// Cluster coordinator for managing RAFT nodes
pub struct ClusterCoordinator {
    /// Cluster configuration
    config: ClusterConfig,
    /// Registry of nodes in the cluster
    nodes: Arc<DashMap<NodeId, NodeInfo>>,
    /// Leader node (if known)
    leader: Arc<RwLock<Option<NodeId>>>,
    /// Shutdown signal
    shutdown: Arc<RwLock<bool>>,
    /// Failover callback (triggered when leader fails)
    failover_callback: FailoverCallback,
}

impl ClusterCoordinator {
    /// Create a new cluster coordinator
    pub fn new(config: ClusterConfig) -> Self {
        Self {
            config,
            nodes: Arc::new(DashMap::new()),
            leader: Arc::new(RwLock::new(None)),
            shutdown: Arc::new(RwLock::new(false)),
            failover_callback: Arc::new(RwLock::new(None)),
        }
    }

    /// Set a callback to be invoked when leader failover is triggered
    pub async fn set_failover_callback<F>(&self, callback: F)
    where
        F: Fn(NodeId) + Send + Sync + 'static,
    {
        *self.failover_callback.write().await = Some(Box::new(callback));
    }

    /// Add a node to the cluster
    #[allow(clippy::unused_async)]
    pub async fn add_node(&self, node_id: NodeId, address: SocketAddr) -> Result<()> {
        if self.nodes.len() >= self.config.max_cluster_size {
            return Err(Error::Network(format!(
                "Cluster size limit reached: {}",
                self.config.max_cluster_size
            )));
        }

        let node_info = NodeInfo {
            node_id,
            address,
            state: NodeState::Follower,
            last_heartbeat: SystemTime::now(),
            health: NodeHealth::Healthy,
            missed_heartbeats: 0,
        };

        self.nodes.insert(node_id, node_info);
        tracing::info!("Added node {} to cluster at {}", node_id.0, address);

        Ok(())
    }

    /// Remove a node from the cluster
    pub async fn remove_node(&self, node_id: NodeId) -> Result<()> {
        self.nodes.remove(&node_id);
        tracing::info!("Removed node {} from cluster", node_id.0);

        // Clear leader if it was this node
        let mut leader = self.leader.write().await;
        if *leader == Some(node_id) {
            *leader = None;
        }

        Ok(())
    }

    /// Update node state
    pub async fn update_node_state(&self, node_id: NodeId, state: NodeState) -> Result<()> {
        if let Some(mut node) = self.nodes.get_mut(&node_id) {
            node.state = state;

            // Update leader if this node became leader
            if state == NodeState::Leader {
                *self.leader.write().await = Some(node_id);
                tracing::info!("Node {} is now the leader", node_id.0);
            }

            Ok(())
        } else {
            Err(Error::Network(format!("Node {} not found", node_id.0)))
        }
    }

    /// Record heartbeat from a node
    #[allow(clippy::unused_async)]
    pub async fn heartbeat(&self, node_id: NodeId) -> Result<()> {
        if let Some(mut node) = self.nodes.get_mut(&node_id) {
            node.last_heartbeat = SystemTime::now();
            node.missed_heartbeats = 0;
            node.health = NodeHealth::Healthy;
            Ok(())
        } else {
            Err(Error::Network(format!("Node {} not found", node_id.0)))
        }
    }

    /// Start health monitoring background task
    #[allow(clippy::unused_async)]
    pub async fn start_health_monitoring(&self) {
        let nodes = self.nodes.clone();
        let config = self.config.clone();
        let shutdown = self.shutdown.clone();
        let leader = self.leader.clone();
        let failover_callback = self.failover_callback.clone();

        tokio::spawn(async move {
            let interval = Duration::from_millis(config.heartbeat_interval_ms);

            loop {
                if *shutdown.read().await {
                    break;
                }

                let mut leader_down = false;
                let mut failed_leader_id = None;

                // Check health of all nodes
                for mut entry in nodes.iter_mut() {
                    let node = entry.value_mut();

                    if let Ok(elapsed) = node.last_heartbeat.elapsed() {
                        let missed =
                            (elapsed.as_millis() / config.heartbeat_interval_ms as u128) as u32;

                        if missed > node.missed_heartbeats {
                            node.missed_heartbeats = missed;

                            // Update health status
                            let old_health = node.health;
                            node.health = if missed >= config.failure_threshold {
                                NodeHealth::Down
                            } else if missed >= config.failure_threshold / 2 {
                                NodeHealth::Suspected
                            } else if missed > 0 {
                                NodeHealth::Degraded
                            } else {
                                NodeHealth::Healthy
                            };

                            // Check if leader went down
                            if node.health == NodeHealth::Down && old_health != NodeHealth::Down {
                                tracing::warn!(
                                    "Node {} is down (missed {} heartbeats)",
                                    node.node_id.0,
                                    missed
                                );

                                // Check if this was the leader
                                let current_leader = leader.read().await;
                                if *current_leader == Some(node.node_id) {
                                    leader_down = true;
                                    failed_leader_id = Some(node.node_id);
                                }
                            }
                        }
                    }
                }

                // Trigger failover if leader is down
                if leader_down {
                    if let Some(leader_id) = failed_leader_id {
                        tracing::warn!("Leader {} has failed, triggering failover", leader_id.0);

                        // Clear current leader
                        *leader.write().await = None;

                        // Invoke failover callback if set
                        if let Some(callback) = failover_callback.read().await.as_ref() {
                            callback(leader_id);
                        }
                    }
                }

                tokio::time::sleep(interval).await;
            }
        });
    }

    /// Manually trigger failover (for testing or manual intervention)
    pub async fn trigger_failover(&self) -> Result<()> {
        let current_leader = *self.leader.read().await;

        if let Some(leader_id) = current_leader {
            tracing::info!("Manually triggering failover for leader {}", leader_id.0);

            // Clear current leader
            *self.leader.write().await = None;

            // Invoke failover callback if set
            if let Some(callback) = self.failover_callback.read().await.as_ref() {
                callback(leader_id);
            }

            Ok(())
        } else {
            Err(Error::Network("No leader to failover from".into()))
        }
    }

    /// Check if automatic re-election should be triggered
    pub async fn should_trigger_reelection(&self) -> bool {
        let current_leader = *self.leader.read().await;

        // If no leader and cluster has quorum, should trigger re-election
        current_leader.is_none() && self.has_quorum()
    }

    /// Get healthy nodes that can participate in election
    pub fn get_election_candidates(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|entry| {
                let node = entry.value();
                matches!(node.health, NodeHealth::Healthy | NodeHealth::Degraded)
            })
            .map(|entry| *entry.key())
            .collect()
    }

    /// Get current cluster size
    pub fn cluster_size(&self) -> usize {
        self.nodes.len()
    }

    /// Get number of healthy nodes
    pub fn healthy_nodes(&self) -> usize {
        self.nodes
            .iter()
            .filter(|entry| entry.value().health == NodeHealth::Healthy)
            .count()
    }

    /// Check if cluster has quorum
    pub fn has_quorum(&self) -> bool {
        let healthy = self.healthy_nodes();
        healthy >= (self.config.min_cluster_size / 2 + 1)
    }

    /// Get current leader
    pub async fn get_leader(&self) -> Option<NodeId> {
        *self.leader.read().await
    }

    /// Get all node IDs
    pub fn get_node_ids(&self) -> Vec<NodeId> {
        self.nodes.iter().map(|entry| *entry.key()).collect()
    }

    /// Get node info
    pub fn get_node_info(&self, node_id: NodeId) -> Option<NodeInfo> {
        self.nodes.get(&node_id).map(|entry| entry.value().clone())
    }

    /// Get cluster statistics
    pub fn get_cluster_stats(&self) -> ClusterStats {
        let total = self.nodes.len();
        let mut healthy = 0;
        let mut degraded = 0;
        let mut suspected = 0;
        let mut down = 0;

        for entry in self.nodes.iter() {
            match entry.value().health {
                NodeHealth::Healthy => healthy += 1,
                NodeHealth::Degraded => degraded += 1,
                NodeHealth::Suspected => suspected += 1,
                NodeHealth::Down => down += 1,
            }
        }

        ClusterStats {
            total_nodes: total,
            healthy_nodes: healthy,
            degraded_nodes: degraded,
            suspected_nodes: suspected,
            down_nodes: down,
            has_quorum: self.has_quorum(),
        }
    }

    /// Shutdown the coordinator
    pub async fn shutdown(&self) {
        *self.shutdown.write().await = true;
    }
}

/// Cluster statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStats {
    /// Total number of nodes
    pub total_nodes: usize,
    /// Number of healthy nodes
    pub healthy_nodes: usize,
    /// Number of degraded nodes
    pub degraded_nodes: usize,
    /// Number of suspected nodes
    pub suspected_nodes: usize,
    /// Number of down nodes
    pub down_nodes: usize,
    /// Whether cluster has quorum
    pub has_quorum: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cluster_add_remove_node() {
        let config = ClusterConfig::default();
        let coordinator = ClusterCoordinator::new(config);

        let node_id = NodeId(1);
        let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();

        coordinator.add_node(node_id, addr).await.unwrap();
        assert_eq!(coordinator.cluster_size(), 1);

        coordinator.remove_node(node_id).await.unwrap();
        assert_eq!(coordinator.cluster_size(), 0);
    }

    #[tokio::test]
    async fn test_cluster_size_limit() {
        let config = ClusterConfig {
            max_cluster_size: 2,
            ..Default::default()
        };
        let coordinator = ClusterCoordinator::new(config);

        coordinator
            .add_node(NodeId(1), "127.0.0.1:8001".parse().unwrap())
            .await
            .unwrap();

        coordinator
            .add_node(NodeId(2), "127.0.0.1:8002".parse().unwrap())
            .await
            .unwrap();

        // Should fail - cluster full
        let result = coordinator
            .add_node(NodeId(3), "127.0.0.1:8003".parse().unwrap())
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_heartbeat() {
        let config = ClusterConfig::default();
        let coordinator = ClusterCoordinator::new(config);

        let node_id = NodeId(1);
        coordinator
            .add_node(node_id, "127.0.0.1:8000".parse().unwrap())
            .await
            .unwrap();

        coordinator.heartbeat(node_id).await.unwrap();

        let info = coordinator.get_node_info(node_id).unwrap();
        assert_eq!(info.health, NodeHealth::Healthy);
        assert_eq!(info.missed_heartbeats, 0);
    }

    #[tokio::test]
    async fn test_leader_tracking() {
        let config = ClusterConfig::default();
        let coordinator = ClusterCoordinator::new(config);

        let node_id = NodeId(1);
        coordinator
            .add_node(node_id, "127.0.0.1:8000".parse().unwrap())
            .await
            .unwrap();

        assert_eq!(coordinator.get_leader().await, None);

        coordinator
            .update_node_state(node_id, NodeState::Leader)
            .await
            .unwrap();

        assert_eq!(coordinator.get_leader().await, Some(node_id));
    }

    #[tokio::test]
    async fn test_quorum() {
        let config = ClusterConfig {
            min_cluster_size: 3,
            ..Default::default()
        };
        let coordinator = ClusterCoordinator::new(config);

        // Add 3 nodes
        coordinator
            .add_node(NodeId(1), "127.0.0.1:8001".parse().unwrap())
            .await
            .unwrap();

        coordinator
            .add_node(NodeId(2), "127.0.0.1:8002".parse().unwrap())
            .await
            .unwrap();

        coordinator
            .add_node(NodeId(3), "127.0.0.1:8003".parse().unwrap())
            .await
            .unwrap();

        // All healthy - should have quorum
        assert!(coordinator.has_quorum());

        let stats = coordinator.get_cluster_stats();
        assert_eq!(stats.total_nodes, 3);
        assert_eq!(stats.healthy_nodes, 3);
        assert!(stats.has_quorum);
    }

    #[tokio::test]
    async fn test_cluster_stats() {
        let config = ClusterConfig::default();
        let coordinator = ClusterCoordinator::new(config);

        coordinator
            .add_node(NodeId(1), "127.0.0.1:8001".parse().unwrap())
            .await
            .unwrap();

        coordinator
            .add_node(NodeId(2), "127.0.0.1:8002".parse().unwrap())
            .await
            .unwrap();

        let stats = coordinator.get_cluster_stats();
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.healthy_nodes, 2);
    }

    #[tokio::test]
    async fn test_manual_failover() {
        let config = ClusterConfig::default();
        let coordinator = ClusterCoordinator::new(config);

        let node_id = NodeId(1);
        coordinator
            .add_node(node_id, "127.0.0.1:8000".parse().unwrap())
            .await
            .unwrap();

        // Set as leader
        coordinator
            .update_node_state(node_id, NodeState::Leader)
            .await
            .unwrap();

        assert_eq!(coordinator.get_leader().await, Some(node_id));

        // Trigger failover
        coordinator.trigger_failover().await.unwrap();

        // Leader should be cleared
        assert_eq!(coordinator.get_leader().await, None);
    }

    #[tokio::test]
    async fn test_failover_callback() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let config = ClusterConfig::default();
        let coordinator = ClusterCoordinator::new(config);

        let node_id = NodeId(1);
        coordinator
            .add_node(node_id, "127.0.0.1:8000".parse().unwrap())
            .await
            .unwrap();

        // Set callback
        let callback_triggered = Arc::new(AtomicBool::new(false));
        let callback_triggered_clone = callback_triggered.clone();

        coordinator
            .set_failover_callback(move |_| {
                callback_triggered_clone.store(true, Ordering::SeqCst);
            })
            .await;

        // Set as leader
        coordinator
            .update_node_state(node_id, NodeState::Leader)
            .await
            .unwrap();

        // Trigger failover
        coordinator.trigger_failover().await.unwrap();

        // Callback should have been triggered
        assert!(callback_triggered.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_reelection_trigger_check() {
        let config = ClusterConfig {
            min_cluster_size: 3,
            ..Default::default()
        };
        let coordinator = ClusterCoordinator::new(config);

        // Add 3 nodes
        coordinator
            .add_node(NodeId(1), "127.0.0.1:8001".parse().unwrap())
            .await
            .unwrap();

        coordinator
            .add_node(NodeId(2), "127.0.0.1:8002".parse().unwrap())
            .await
            .unwrap();

        coordinator
            .add_node(NodeId(3), "127.0.0.1:8003".parse().unwrap())
            .await
            .unwrap();

        // No leader, has quorum - should trigger re-election
        assert!(coordinator.should_trigger_reelection().await);

        // Set leader
        coordinator
            .update_node_state(NodeId(1), NodeState::Leader)
            .await
            .unwrap();

        // Has leader - should not trigger re-election
        assert!(!coordinator.should_trigger_reelection().await);
    }

    #[tokio::test]
    async fn test_election_candidates() {
        let config = ClusterConfig::default();
        let coordinator = ClusterCoordinator::new(config);

        coordinator
            .add_node(NodeId(1), "127.0.0.1:8001".parse().unwrap())
            .await
            .unwrap();

        coordinator
            .add_node(NodeId(2), "127.0.0.1:8002".parse().unwrap())
            .await
            .unwrap();

        let candidates = coordinator.get_election_candidates();
        assert_eq!(candidates.len(), 2);
        assert!(candidates.contains(&NodeId(1)));
        assert!(candidates.contains(&NodeId(2)));
    }
}
