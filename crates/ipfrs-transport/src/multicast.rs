//! Multicast block announcements
//!
//! Efficient fan-out for block availability notifications:
//! - Subscription management for interested peers
//! - Topic-based filtering
//! - Reduce announcement overhead
//! - Scalable notifications

use dashmap::DashMap;
use ipfrs_core::Cid;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, trace};

/// Serialize CID as string
fn serialize_cid<S>(cid: &Cid, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(&cid.to_string())
}

/// Deserialize CID from string
fn deserialize_cid<'de, D>(deserializer: D) -> Result<Cid, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    s.parse().map_err(serde::de::Error::custom)
}

/// Error types for multicast
#[derive(Error, Debug)]
pub enum MulticastError {
    #[error("Topic not found: {0}")]
    TopicNotFound(String),
    #[error("Subscription failed: {0}")]
    SubscriptionFailed(String),
    #[error("Already subscribed")]
    AlreadySubscribed,
    #[error("Not subscribed")]
    NotSubscribed,
}

/// Peer identifier type
pub type PeerId = String;

/// Topic for grouping related announcements
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Topic(pub String);

impl Topic {
    /// Create a new topic
    pub fn new(name: impl Into<String>) -> Self {
        Topic(name.into())
    }

    /// Topic for all block announcements
    pub fn all_blocks() -> Self {
        Topic("blocks:all".to_string())
    }

    /// Topic for specific content type
    pub fn content_type(content_type: &str) -> Self {
        Topic(format!("blocks:{}", content_type))
    }

    /// Topic for tensor blocks
    pub fn tensors() -> Self {
        Topic("blocks:tensors".to_string())
    }

    /// Topic for gradients
    pub fn gradients() -> Self {
        Topic("blocks:gradients".to_string())
    }
}

/// Block announcement message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockAnnouncement {
    /// CID of the announced block
    #[serde(serialize_with = "serialize_cid", deserialize_with = "deserialize_cid")]
    pub cid: Cid,
    /// Block size in bytes
    pub size: u64,
    /// Optional topic for filtering
    pub topic: Option<Topic>,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl BlockAnnouncement {
    /// Create a new block announcement
    pub fn new(cid: Cid, size: u64) -> Self {
        Self {
            cid,
            size,
            topic: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the topic for this announcement
    pub fn with_topic(mut self, topic: Topic) -> Self {
        self.topic = Some(topic);
        self
    }

    /// Add metadata to the announcement
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Subscription filter
#[derive(Clone)]
pub enum SubscriptionFilter {
    /// Subscribe to all announcements
    All,
    /// Subscribe to specific topic
    Topic(Topic),
    /// Subscribe to multiple topics
    Topics(Vec<Topic>),
    /// Subscribe to announcements matching a predicate
    Custom(Arc<dyn Fn(&BlockAnnouncement) -> bool + Send + Sync>),
}

impl std::fmt::Debug for SubscriptionFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::All => write!(f, "SubscriptionFilter::All"),
            Self::Topic(topic) => write!(f, "SubscriptionFilter::Topic({:?})", topic),
            Self::Topics(topics) => write!(f, "SubscriptionFilter::Topics({:?})", topics),
            Self::Custom(_) => write!(f, "SubscriptionFilter::Custom(<function>)"),
        }
    }
}

impl SubscriptionFilter {
    /// Check if an announcement matches this filter
    pub fn matches(&self, announcement: &BlockAnnouncement) -> bool {
        match self {
            SubscriptionFilter::All => true,
            SubscriptionFilter::Topic(topic) => announcement.topic.as_ref() == Some(topic),
            SubscriptionFilter::Topics(topics) => {
                if let Some(ref ann_topic) = announcement.topic {
                    topics.contains(ann_topic)
                } else {
                    false
                }
            }
            SubscriptionFilter::Custom(predicate) => predicate(announcement),
        }
    }
}

/// Subscription handle
#[derive(Debug)]
pub struct Subscription {
    peer_id: PeerId,
    filter: SubscriptionFilter,
    created_at: std::time::Instant,
}

impl Subscription {
    /// Create a new subscription
    pub fn new(peer_id: PeerId, filter: SubscriptionFilter) -> Self {
        Self {
            peer_id,
            filter,
            created_at: std::time::Instant::now(),
        }
    }

    /// Check if this subscription matches an announcement
    pub fn matches(&self, announcement: &BlockAnnouncement) -> bool {
        self.filter.matches(announcement)
    }

    /// Get the peer ID for this subscription
    pub fn peer_id(&self) -> &str {
        &self.peer_id
    }

    /// Get subscription age
    pub fn age(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }
}

/// Multicast manager configuration
#[derive(Debug, Clone)]
pub struct MulticastConfig {
    /// Maximum subscriptions per peer
    pub max_subscriptions_per_peer: usize,
    /// Maximum total subscriptions
    pub max_total_subscriptions: usize,
    /// Enable topic-based routing
    pub enable_topic_routing: bool,
    /// Announcement deduplication window (seconds)
    pub dedup_window_secs: u64,
}

impl Default for MulticastConfig {
    fn default() -> Self {
        Self {
            max_subscriptions_per_peer: 100,
            max_total_subscriptions: 10000,
            enable_topic_routing: true,
            dedup_window_secs: 60,
        }
    }
}

/// Multicast statistics
#[derive(Debug, Clone, Default)]
pub struct MulticastStats {
    /// Total announcements sent
    pub announcements_sent: u64,
    /// Total announcements received
    pub announcements_received: u64,
    /// Active subscriptions
    pub active_subscriptions: usize,
    /// Unique topics
    pub unique_topics: usize,
    /// Announcements filtered out
    pub filtered_announcements: u64,
}

/// Multicast manager for efficient block announcements
pub struct MulticastManager {
    /// Configuration
    config: MulticastConfig,
    /// Active subscriptions by peer
    subscriptions: Arc<DashMap<PeerId, Vec<Subscription>>>,
    /// Topic index for efficient routing
    topic_index: Arc<RwLock<HashMap<Topic, HashSet<PeerId>>>>,
    /// Recent announcements for deduplication
    recent_announcements: Arc<RwLock<HashMap<Cid, std::time::Instant>>>,
    /// Statistics
    stats: Arc<RwLock<MulticastStats>>,
}

impl MulticastManager {
    /// Create a new multicast manager
    pub fn new(config: MulticastConfig) -> Self {
        Self {
            config,
            subscriptions: Arc::new(DashMap::new()),
            topic_index: Arc::new(RwLock::new(HashMap::new())),
            recent_announcements: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(MulticastStats::default())),
        }
    }

    /// Subscribe a peer to announcements
    pub async fn subscribe(
        &self,
        peer_id: PeerId,
        filter: SubscriptionFilter,
    ) -> Result<(), MulticastError> {
        // Check subscription limits
        let total_subs = self
            .subscriptions
            .iter()
            .map(|r| r.value().len())
            .sum::<usize>();
        if total_subs >= self.config.max_total_subscriptions {
            return Err(MulticastError::SubscriptionFailed(
                "Max total subscriptions reached".to_string(),
            ));
        }

        let subscription = Subscription::new(peer_id.clone(), filter.clone());

        // Add subscription
        self.subscriptions
            .entry(peer_id.clone())
            .or_default()
            .push(subscription);

        // Update topic index if topic-based routing is enabled
        if self.config.enable_topic_routing {
            if let SubscriptionFilter::Topic(topic) = &filter {
                let mut index = self.topic_index.write().await;
                index
                    .entry(topic.clone())
                    .or_insert_with(HashSet::new)
                    .insert(peer_id.clone());
            } else if let SubscriptionFilter::Topics(topics) = &filter {
                let mut index = self.topic_index.write().await;
                for topic in topics {
                    index
                        .entry(topic.clone())
                        .or_insert_with(HashSet::new)
                        .insert(peer_id.clone());
                }
            }
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.active_subscriptions = self.subscriptions.iter().map(|r| r.value().len()).sum();

        debug!("Peer {} subscribed with filter: {:?}", peer_id, filter);
        Ok(())
    }

    /// Unsubscribe a peer from all announcements
    pub async fn unsubscribe(&self, peer_id: &str) -> Result<(), MulticastError> {
        // Remove from subscriptions
        if self.subscriptions.remove(peer_id).is_none() {
            return Err(MulticastError::NotSubscribed);
        }

        // Remove from topic index
        if self.config.enable_topic_routing {
            let mut index = self.topic_index.write().await;
            for peers in index.values_mut() {
                peers.remove(peer_id);
            }
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.active_subscriptions = self.subscriptions.iter().map(|r| r.value().len()).sum();

        debug!("Peer {} unsubscribed", peer_id);
        Ok(())
    }

    /// Announce a new block to subscribed peers
    pub async fn announce(&self, announcement: BlockAnnouncement) -> Vec<PeerId> {
        // Check for duplicate announcement
        {
            let mut recent = self.recent_announcements.write().await;
            let now = std::time::Instant::now();

            // Clean up old announcements
            recent.retain(|_, timestamp| {
                now.duration_since(*timestamp).as_secs() < self.config.dedup_window_secs
            });

            // Check if already announced recently
            if recent.contains_key(&announcement.cid) {
                trace!(
                    "Skipping duplicate announcement for CID: {}",
                    announcement.cid
                );
                return Vec::new();
            }

            recent.insert(announcement.cid, now);
        }

        let mut interested_peers = HashSet::new();

        // Topic-based routing for efficiency
        if self.config.enable_topic_routing {
            if let Some(ref topic) = announcement.topic {
                let index = self.topic_index.read().await;
                if let Some(peers) = index.get(topic) {
                    interested_peers.extend(peers.iter().cloned());
                }
            }
        }

        // Also check subscriptions that don't use topic routing
        for entry in self.subscriptions.iter() {
            let peer_id = entry.key();
            let subscriptions = entry.value();

            for subscription in subscriptions {
                if subscription.matches(&announcement) {
                    interested_peers.insert(peer_id.clone());
                }
            }
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.announcements_sent += interested_peers.len() as u64;

        trace!(
            "Announced CID {} to {} peers",
            announcement.cid,
            interested_peers.len()
        );

        interested_peers.into_iter().collect()
    }

    /// Get subscriptions for a peer
    pub fn get_subscriptions(&self, peer_id: &str) -> Option<Vec<PeerId>> {
        self.subscriptions
            .get(peer_id)
            .map(|subs| vec![peer_id.to_string(); subs.len()])
    }

    /// Get statistics
    pub async fn stats(&self) -> MulticastStats {
        let stats = self.stats.read().await;
        let mut result = stats.clone();
        result.unique_topics = self.topic_index.read().await.len();
        result
    }

    /// Clear all subscriptions
    pub async fn clear(&self) {
        self.subscriptions.clear();
        self.topic_index.write().await.clear();
        self.recent_announcements.write().await.clear();

        let mut stats = self.stats.write().await;
        stats.active_subscriptions = 0;
        stats.unique_topics = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cid() -> Cid {
        "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse()
            .expect("test: parse well-known CID string")
    }

    fn test_cid2() -> Cid {
        "bafybeihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku"
            .parse()
            .expect("test: parse well-known CID string")
    }

    #[test]
    fn test_topic_creation() {
        let topic = Topic::new("test");
        assert_eq!(topic.0, "test");

        let all_blocks = Topic::all_blocks();
        assert_eq!(all_blocks.0, "blocks:all");

        let tensors = Topic::tensors();
        assert_eq!(tensors.0, "blocks:tensors");
    }

    #[test]
    fn test_block_announcement() {
        let cid = test_cid();
        let announcement = BlockAnnouncement::new(cid, 1024)
            .with_topic(Topic::tensors())
            .with_metadata("dtype".to_string(), "float32".to_string());

        assert_eq!(announcement.size, 1024);
        assert_eq!(announcement.topic, Some(Topic::tensors()));
        assert_eq!(
            announcement
                .metadata
                .get("dtype")
                .expect("test: get dtype metadata entry"),
            "float32"
        );
    }

    #[tokio::test]
    async fn test_subscribe_unsubscribe() {
        let manager = MulticastManager::new(MulticastConfig::default());

        manager
            .subscribe("peer1".to_string(), SubscriptionFilter::All)
            .await
            .expect("test: subscribe peer1 with All filter");

        let stats = manager.stats().await;
        assert_eq!(stats.active_subscriptions, 1);

        manager
            .unsubscribe("peer1")
            .await
            .expect("test: unsubscribe peer1");

        let stats = manager.stats().await;
        assert_eq!(stats.active_subscriptions, 0);
    }

    #[tokio::test]
    async fn test_topic_based_announcement() {
        let manager = MulticastManager::new(MulticastConfig::default());

        manager
            .subscribe(
                "peer1".to_string(),
                SubscriptionFilter::Topic(Topic::tensors()),
            )
            .await
            .expect("test: subscribe peer1 with tensors topic filter");

        manager
            .subscribe(
                "peer2".to_string(),
                SubscriptionFilter::Topic(Topic::gradients()),
            )
            .await
            .expect("test: subscribe peer2 with gradients topic filter");

        let cid = test_cid();
        let announcement = BlockAnnouncement::new(cid, 1024).with_topic(Topic::tensors());

        let peers = manager.announce(announcement).await;
        assert_eq!(peers.len(), 1);
        assert!(peers.contains(&"peer1".to_string()));
    }

    #[tokio::test]
    async fn test_all_announcements_subscription() {
        let manager = MulticastManager::new(MulticastConfig::default());

        manager
            .subscribe("peer1".to_string(), SubscriptionFilter::All)
            .await
            .expect("test: subscribe peer1 with All filter");

        let cid = test_cid();
        let announcement = BlockAnnouncement::new(cid, 1024).with_topic(Topic::tensors());

        let peers = manager.announce(announcement).await;
        assert_eq!(peers.len(), 1);
        assert!(peers.contains(&"peer1".to_string()));
    }

    #[tokio::test]
    async fn test_multiple_topics_subscription() {
        let manager = MulticastManager::new(MulticastConfig::default());

        manager
            .subscribe(
                "peer1".to_string(),
                SubscriptionFilter::Topics(vec![Topic::tensors(), Topic::gradients()]),
            )
            .await
            .expect("test: subscribe peer1 with multiple topics filter");

        let cid1 = test_cid();
        let announcement1 = BlockAnnouncement::new(cid1, 1024).with_topic(Topic::tensors());
        let peers1 = manager.announce(announcement1).await;
        assert_eq!(peers1.len(), 1);

        let cid2 = test_cid2();
        let announcement2 = BlockAnnouncement::new(cid2, 2048).with_topic(Topic::gradients());
        let peers2 = manager.announce(announcement2).await;
        assert_eq!(peers2.len(), 1);
    }

    #[tokio::test]
    async fn test_deduplication() {
        let manager = MulticastManager::new(MulticastConfig::default());

        manager
            .subscribe("peer1".to_string(), SubscriptionFilter::All)
            .await
            .expect("test: subscribe peer1 with All filter");

        let cid = test_cid();
        let announcement = BlockAnnouncement::new(cid, 1024);

        let peers1 = manager.announce(announcement.clone()).await;
        assert_eq!(peers1.len(), 1);

        // Duplicate announcement should be filtered
        let peers2 = manager.announce(announcement).await;
        assert_eq!(peers2.len(), 0);
    }

    #[tokio::test]
    async fn test_subscription_limits() {
        let config = MulticastConfig {
            max_total_subscriptions: 2,
            ..Default::default()
        };
        let manager = MulticastManager::new(config);

        manager
            .subscribe("peer1".to_string(), SubscriptionFilter::All)
            .await
            .expect("test: subscribe peer1 within limit");

        manager
            .subscribe("peer2".to_string(), SubscriptionFilter::All)
            .await
            .expect("test: subscribe peer2 within limit");

        let result = manager
            .subscribe("peer3".to_string(), SubscriptionFilter::All)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_clear_subscriptions() {
        let manager = MulticastManager::new(MulticastConfig::default());

        manager
            .subscribe("peer1".to_string(), SubscriptionFilter::All)
            .await
            .expect("test: subscribe peer1 with All filter");

        manager
            .subscribe("peer2".to_string(), SubscriptionFilter::All)
            .await
            .expect("test: subscribe peer2 with All filter");

        let stats = manager.stats().await;
        assert_eq!(stats.active_subscriptions, 2);

        manager.clear().await;

        let stats = manager.stats().await;
        assert_eq!(stats.active_subscriptions, 0);
    }
}
