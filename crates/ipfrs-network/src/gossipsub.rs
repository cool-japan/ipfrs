//! GossipSub - Topic-based pub/sub messaging
//!
//! This module provides efficient topic-based publish/subscribe messaging
//! using the GossipSub protocol from libp2p.
//!
//! ## Features
//!
//! - **Topic Subscription**: Subscribe to topics of interest
//! - **Message Publishing**: Publish messages to topics
//! - **Mesh Formation**: Automatic peer mesh formation for topic propagation
//! - **Message Deduplication**: Seen message tracking to prevent duplicates
//! - **Peer Scoring**: Score-based peer selection for mesh quality
//! - **Content Announcements**: Broadcast new content availability
//!
//! ## Design
//!
//! GossipSub maintains a mesh of peers for each topic, ensuring:
//! - Low latency message delivery
//! - High reliability through redundancy
//! - Efficient bandwidth usage through mesh optimization
//! - Resistance to spam and malicious peers through scoring

use dashmap::DashMap;
use libp2p::PeerId;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Errors that can occur in GossipSub operations
#[derive(Error, Debug)]
pub enum GossipSubError {
    #[error("Topic not found: {0}")]
    TopicNotFound(String),

    #[error("Already subscribed to topic: {0}")]
    AlreadySubscribed(String),

    #[error("Not subscribed to topic: {0}")]
    NotSubscribed(String),

    #[error("Message too large: {size} bytes (max: {max})")]
    MessageTooLarge { size: usize, max: usize },

    #[error("Invalid topic name: {0}")]
    InvalidTopicName(String),

    #[error("Peer scoring error: {0}")]
    ScoringError(String),
}

/// GossipSub configuration
#[derive(Debug, Clone)]
pub struct GossipSubConfig {
    /// Minimum number of peers in mesh (D_low)
    pub mesh_n_low: usize,

    /// Target number of peers in mesh (D)
    pub mesh_n: usize,

    /// Maximum number of peers in mesh (D_high)
    pub mesh_n_high: usize,

    /// Number of peers to send gossip to (D_lazy)
    pub gossip_n: usize,

    /// Heartbeat interval for mesh maintenance
    pub heartbeat_interval: Duration,

    /// Maximum message size
    pub max_message_size: usize,

    /// Enable peer scoring
    pub enable_scoring: bool,

    /// Time window for message deduplication
    pub duplicate_cache_time: Duration,

    /// Maximum number of messages in duplicate cache
    pub max_duplicate_cache_size: usize,

    /// Enable message validation
    pub enable_validation: bool,
}

impl Default for GossipSubConfig {
    fn default() -> Self {
        Self {
            mesh_n_low: 4,
            mesh_n: 6,
            mesh_n_high: 12,
            gossip_n: 3,
            heartbeat_interval: Duration::from_secs(1),
            max_message_size: 1024 * 1024, // 1 MB
            enable_scoring: true,
            duplicate_cache_time: Duration::from_secs(120),
            max_duplicate_cache_size: 10000,
            enable_validation: true,
        }
    }
}

/// Topic identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TopicId(pub String);

impl TopicId {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Topic for content announcements
    pub fn content_announce() -> Self {
        Self("/ipfrs/content/announce/1.0.0".to_string())
    }

    /// Topic for peer announcements
    pub fn peer_announce() -> Self {
        Self("/ipfrs/peer/announce/1.0.0".to_string())
    }

    /// Topic for DHT events
    pub fn dht_events() -> Self {
        Self("/ipfrs/dht/events/1.0.0".to_string())
    }
}

/// GossipSub message
#[derive(Debug, Clone)]
pub struct GossipSubMessage {
    /// Message ID
    pub id: MessageId,

    /// Source peer
    pub source: PeerId,

    /// Topic this message belongs to
    pub topic: TopicId,

    /// Message payload
    pub data: Vec<u8>,

    /// Sequence number
    pub sequence: u64,

    /// Timestamp
    pub timestamp: Instant,
}

/// Message identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MessageId(pub Vec<u8>);

impl MessageId {
    /// Create a message ID from source peer and sequence number
    pub fn new(source: &PeerId, sequence: u64) -> Self {
        let mut data = source.to_bytes();
        data.extend_from_slice(&sequence.to_le_bytes());
        Self(data)
    }
}

/// Peer score for mesh quality
#[derive(Debug, Clone, Default)]
pub struct PeerScore {
    /// Topic-specific scores
    pub topic_scores: HashMap<TopicId, f64>,

    /// Overall score
    pub total_score: f64,

    /// Number of invalid messages
    pub invalid_messages: u64,

    /// Number of valid messages
    pub valid_messages: u64,

    /// Last update time
    pub last_update: Option<Instant>,
}

impl PeerScore {
    /// Calculate overall score
    pub fn calculate_total(&mut self) {
        if self.topic_scores.is_empty() {
            self.total_score = 0.0;
            return;
        }

        // Average of topic scores
        let sum: f64 = self.topic_scores.values().sum();
        self.total_score = sum / self.topic_scores.len() as f64;

        // Penalize for invalid messages
        let total_messages = self.invalid_messages + self.valid_messages;
        if total_messages > 0 {
            let invalid_ratio = self.invalid_messages as f64 / total_messages as f64;
            self.total_score *= 1.0 - invalid_ratio;
        }

        self.last_update = Some(Instant::now());
    }

    /// Update topic score
    pub fn update_topic_score(&mut self, topic: TopicId, score: f64) {
        self.topic_scores.insert(topic, score);
        self.calculate_total();
    }

    /// Record message validation result
    pub fn record_message(&mut self, valid: bool) {
        if valid {
            self.valid_messages += 1;
        } else {
            self.invalid_messages += 1;
        }
        self.calculate_total();
    }
}

/// Topic subscription information
#[derive(Debug, Clone)]
pub struct TopicSubscription {
    /// Topic ID
    pub topic: TopicId,

    /// Subscribed since
    pub subscribed_at: Instant,

    /// Mesh peers for this topic
    pub mesh_peers: HashSet<PeerId>,

    /// Number of messages received
    pub messages_received: u64,

    /// Number of messages published
    pub messages_published: u64,
}

/// GossipSub statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GossipSubStats {
    /// Total topics subscribed
    pub subscribed_topics: usize,

    /// Total messages published
    pub messages_published: u64,

    /// Total messages received
    pub messages_received: u64,

    /// Total duplicate messages seen
    pub duplicate_messages: u64,

    /// Total invalid messages
    pub invalid_messages: u64,

    /// Active mesh peers
    pub active_mesh_peers: usize,

    /// Mesh prune events
    pub mesh_prune_count: u64,

    /// Mesh graft events
    pub mesh_graft_count: u64,

    /// Messages per topic
    pub messages_per_topic: HashMap<String, u64>,
}

/// Message seen cache entry
#[derive(Debug, Clone)]
struct SeenCacheEntry {
    timestamp: Instant,
}

/// GossipSub manager
pub struct GossipSubManager {
    /// Configuration
    config: GossipSubConfig,

    /// Subscribed topics
    subscriptions: Arc<DashMap<TopicId, TopicSubscription>>,

    /// Peer scores
    peer_scores: Arc<DashMap<PeerId, PeerScore>>,

    /// Seen message cache (deduplication)
    seen_messages: Arc<DashMap<MessageId, SeenCacheEntry>>,

    /// Message sequence number counter
    sequence_counter: Arc<RwLock<u64>>,

    /// Statistics
    stats: Arc<RwLock<GossipSubStats>>,
}

impl GossipSubManager {
    /// Create a new GossipSub manager
    pub fn new(config: GossipSubConfig) -> Self {
        Self {
            config,
            subscriptions: Arc::new(DashMap::new()),
            peer_scores: Arc::new(DashMap::new()),
            seen_messages: Arc::new(DashMap::new()),
            sequence_counter: Arc::new(RwLock::new(0)),
            stats: Arc::new(RwLock::new(GossipSubStats::default())),
        }
    }

    /// Subscribe to a topic
    pub fn subscribe(&self, topic: TopicId) -> Result<(), GossipSubError> {
        if self.subscriptions.contains_key(&topic) {
            return Err(GossipSubError::AlreadySubscribed(topic.0.clone()));
        }

        let subscription = TopicSubscription {
            topic: topic.clone(),
            subscribed_at: Instant::now(),
            mesh_peers: HashSet::new(),
            messages_received: 0,
            messages_published: 0,
        };

        self.subscriptions.insert(topic.clone(), subscription);

        let mut stats = self.stats.write();
        stats.subscribed_topics = self.subscriptions.len();

        Ok(())
    }

    /// Unsubscribe from a topic
    pub fn unsubscribe(&self, topic: &TopicId) -> Result<(), GossipSubError> {
        self.subscriptions
            .remove(topic)
            .ok_or_else(|| GossipSubError::NotSubscribed(topic.0.clone()))?;

        let mut stats = self.stats.write();
        stats.subscribed_topics = self.subscriptions.len();

        Ok(())
    }

    /// Publish a message to a topic
    pub fn publish(
        &self,
        topic: TopicId,
        data: Vec<u8>,
        source: PeerId,
    ) -> Result<MessageId, GossipSubError> {
        // Check if subscribed
        if !self.subscriptions.contains_key(&topic) {
            return Err(GossipSubError::NotSubscribed(topic.0.clone()));
        }

        // Check message size
        if data.len() > self.config.max_message_size {
            return Err(GossipSubError::MessageTooLarge {
                size: data.len(),
                max: self.config.max_message_size,
            });
        }

        // Generate sequence number
        let sequence = {
            let mut counter = self.sequence_counter.write();
            *counter += 1;
            *counter
        };

        // Create message ID
        let message_id = MessageId::new(&source, sequence);

        // Update statistics
        if let Some(mut subscription) = self.subscriptions.get_mut(&topic) {
            subscription.messages_published += 1;
        }

        let mut stats = self.stats.write();
        stats.messages_published += 1;
        *stats.messages_per_topic.entry(topic.0.clone()).or_insert(0) += 1;

        Ok(message_id)
    }

    /// Handle received message
    pub fn handle_message(&self, message: GossipSubMessage) -> Result<bool, GossipSubError> {
        // Check for duplicate
        if self.is_duplicate(&message.id) {
            let mut stats = self.stats.write();
            stats.duplicate_messages += 1;
            return Ok(false); // Message already seen
        }

        // Add to seen cache
        self.add_to_seen_cache(message.id.clone());

        // Validate message if enabled
        if self.config.enable_validation && !self.validate_message(&message) {
            let mut stats = self.stats.write();
            stats.invalid_messages += 1;

            // Update peer score
            if self.config.enable_scoring {
                if let Some(mut score) = self.peer_scores.get_mut(&message.source) {
                    score.record_message(false);
                }
            }

            return Ok(false);
        }

        // Update statistics
        if let Some(mut subscription) = self.subscriptions.get_mut(&message.topic) {
            subscription.messages_received += 1;
        }

        let mut stats = self.stats.write();
        stats.messages_received += 1;

        // Update peer score
        if self.config.enable_scoring {
            self.peer_scores
                .entry(message.source)
                .or_default()
                .record_message(true);
        }

        Ok(true) // Message is new and valid
    }

    /// Check if message is a duplicate
    fn is_duplicate(&self, message_id: &MessageId) -> bool {
        if let Some(entry) = self.seen_messages.get(message_id) {
            let age = Instant::now().duration_since(entry.timestamp);
            return age < self.config.duplicate_cache_time;
        }
        false
    }

    /// Add message to seen cache
    fn add_to_seen_cache(&self, message_id: MessageId) {
        let entry = SeenCacheEntry {
            timestamp: Instant::now(),
        };

        self.seen_messages.insert(message_id, entry);

        // Cleanup old entries if cache is too large
        if self.seen_messages.len() > self.config.max_duplicate_cache_size {
            self.cleanup_seen_cache();
        }
    }

    /// Cleanup old entries from seen cache
    fn cleanup_seen_cache(&self) {
        let now = Instant::now();
        let ttl = self.config.duplicate_cache_time;

        self.seen_messages
            .retain(|_, entry| now.duration_since(entry.timestamp) < ttl);
    }

    /// Validate message
    fn validate_message(&self, _message: &GossipSubMessage) -> bool {
        // Basic validation - can be extended
        // Check if source peer is not banned, message format is correct, etc.
        true
    }

    /// Add peer to topic mesh
    pub fn add_peer_to_mesh(&self, topic: &TopicId, peer: PeerId) -> Result<(), GossipSubError> {
        let inserted = {
            let mut subscription = self
                .subscriptions
                .get_mut(topic)
                .ok_or_else(|| GossipSubError::NotSubscribed(topic.0.clone()))?;
            subscription.mesh_peers.insert(peer)
        }; // Guard dropped here before count_mesh_peers()

        if inserted {
            let mut stats = self.stats.write();
            stats.mesh_graft_count += 1;
            stats.active_mesh_peers = self.count_mesh_peers();
        }

        Ok(())
    }

    /// Remove peer from topic mesh
    pub fn remove_peer_from_mesh(
        &self,
        topic: &TopicId,
        peer: &PeerId,
    ) -> Result<(), GossipSubError> {
        let removed = {
            let mut subscription = self
                .subscriptions
                .get_mut(topic)
                .ok_or_else(|| GossipSubError::NotSubscribed(topic.0.clone()))?;
            subscription.mesh_peers.remove(peer)
        }; // Guard dropped here before count_mesh_peers()

        if removed {
            let mut stats = self.stats.write();
            stats.mesh_prune_count += 1;
            stats.active_mesh_peers = self.count_mesh_peers();
        }

        Ok(())
    }

    /// Get peers in topic mesh
    pub fn get_mesh_peers(&self, topic: &TopicId) -> Result<Vec<PeerId>, GossipSubError> {
        let subscription = self
            .subscriptions
            .get(topic)
            .ok_or_else(|| GossipSubError::NotSubscribed(topic.0.clone()))?;

        Ok(subscription.mesh_peers.iter().cloned().collect())
    }

    /// Count total mesh peers across all topics
    fn count_mesh_peers(&self) -> usize {
        self.subscriptions
            .iter()
            .map(|entry| entry.mesh_peers.len())
            .sum()
    }

    /// Get peer score
    pub fn get_peer_score(&self, peer: &PeerId) -> Option<PeerScore> {
        self.peer_scores.get(peer).map(|s| s.clone())
    }

    /// Update peer score for a topic
    pub fn update_peer_score(&self, peer: &PeerId, topic: TopicId, score: f64) {
        self.peer_scores
            .entry(*peer)
            .or_default()
            .update_topic_score(topic, score);
    }

    /// Get low-scoring peers that should be pruned
    pub fn get_peers_to_prune(&self, topic: &TopicId, threshold: f64) -> Vec<PeerId> {
        let subscription = match self.subscriptions.get(topic) {
            Some(sub) => sub,
            None => return Vec::new(),
        };

        subscription
            .mesh_peers
            .iter()
            .filter(|peer| {
                if let Some(score) = self.peer_scores.get(peer) {
                    score.total_score < threshold
                } else {
                    false
                }
            })
            .cloned()
            .collect()
    }

    /// Get statistics
    pub fn stats(&self) -> GossipSubStats {
        self.stats.read().clone()
    }

    /// List subscribed topics
    pub fn list_topics(&self) -> Vec<TopicId> {
        self.subscriptions
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Check if subscribed to a topic
    pub fn is_subscribed(&self, topic: &TopicId) -> bool {
        self.subscriptions.contains_key(topic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::identity::Keypair;

    /// Create a deterministic PeerId from an index (avoids slow random key generation)
    fn test_peer_id(index: u8) -> PeerId {
        // Use a deterministic seed based on index
        let mut seed = [0u8; 32];
        seed[0] = index;
        let keypair = Keypair::ed25519_from_bytes(seed).expect("valid seed");
        keypair.public().to_peer_id()
    }

    #[test]
    fn test_gossipsub_manager_creation() {
        let config = GossipSubConfig::default();
        let manager = GossipSubManager::new(config);
        assert_eq!(manager.list_topics().len(), 0);
    }

    #[test]
    fn test_topic_subscription() {
        let manager = GossipSubManager::new(GossipSubConfig::default());
        let topic = TopicId::content_announce();

        manager.subscribe(topic.clone()).unwrap();
        assert!(manager.is_subscribed(&topic));
        assert_eq!(manager.list_topics().len(), 1);
    }

    #[test]
    fn test_duplicate_subscription() {
        let manager = GossipSubManager::new(GossipSubConfig::default());
        let topic = TopicId::content_announce();

        manager.subscribe(topic.clone()).unwrap();
        let result = manager.subscribe(topic);
        assert!(matches!(result, Err(GossipSubError::AlreadySubscribed(_))));
    }

    #[test]
    fn test_unsubscribe() {
        let manager = GossipSubManager::new(GossipSubConfig::default());
        let topic = TopicId::content_announce();

        manager.subscribe(topic.clone()).unwrap();
        manager.unsubscribe(&topic).unwrap();
        assert!(!manager.is_subscribed(&topic));
        assert_eq!(manager.list_topics().len(), 0);
    }

    #[test]
    fn test_publish_message() {
        let manager = GossipSubManager::new(GossipSubConfig::default());
        let topic = TopicId::content_announce();
        let peer = test_peer_id(1);

        manager.subscribe(topic.clone()).unwrap();

        let data = b"Hello, GossipSub!".to_vec();
        let message_id = manager.publish(topic, data, peer).unwrap();

        let stats = manager.stats();
        assert_eq!(stats.messages_published, 1);
        assert!(!message_id.0.is_empty());
    }

    #[test]
    fn test_publish_without_subscription() {
        let manager = GossipSubManager::new(GossipSubConfig::default());
        let topic = TopicId::content_announce();
        let peer = test_peer_id(1);

        let data = b"Hello".to_vec();
        let result = manager.publish(topic, data, peer);
        assert!(matches!(result, Err(GossipSubError::NotSubscribed(_))));
    }

    #[test]
    fn test_message_too_large() {
        let config = GossipSubConfig {
            max_message_size: 100,
            ..Default::default()
        };
        let manager = GossipSubManager::new(config);
        let topic = TopicId::content_announce();
        let peer = test_peer_id(1);

        manager.subscribe(topic.clone()).unwrap();

        let data = vec![0u8; 200]; // Larger than max
        let result = manager.publish(topic, data, peer);
        assert!(matches!(
            result,
            Err(GossipSubError::MessageTooLarge { .. })
        ));
    }

    #[test]
    fn test_message_deduplication() {
        let manager = GossipSubManager::new(GossipSubConfig::default());
        let topic = TopicId::content_announce();
        let peer = test_peer_id(1);

        manager.subscribe(topic.clone()).unwrap();

        let message = GossipSubMessage {
            id: MessageId::new(&peer, 1),
            source: peer,
            topic: topic.clone(),
            data: b"Test".to_vec(),
            sequence: 1,
            timestamp: Instant::now(),
        };

        // First message should be accepted
        let result1 = manager.handle_message(message.clone()).unwrap();
        assert!(result1);

        // Duplicate should be rejected
        let result2 = manager.handle_message(message).unwrap();
        assert!(!result2);

        let stats = manager.stats();
        assert_eq!(stats.duplicate_messages, 1);
    }

    #[test]
    fn test_peer_scoring() {
        let manager = GossipSubManager::new(GossipSubConfig::default());
        let peer = test_peer_id(1);
        let topic = TopicId::content_announce();

        manager.update_peer_score(&peer, topic.clone(), 0.8);

        let score = manager.get_peer_score(&peer).unwrap();
        assert_eq!(score.topic_scores.get(&topic), Some(&0.8));
        assert!(score.total_score > 0.0);
    }

    #[test]
    fn test_mesh_management() {
        let manager = GossipSubManager::new(GossipSubConfig::default());
        let topic = TopicId::content_announce();
        let peer1 = test_peer_id(1);
        let peer2 = test_peer_id(2);

        manager.subscribe(topic.clone()).unwrap();

        // Add peers to mesh
        manager.add_peer_to_mesh(&topic, peer1).unwrap();
        manager.add_peer_to_mesh(&topic, peer2).unwrap();

        let mesh_peers = manager.get_mesh_peers(&topic).unwrap();
        assert_eq!(mesh_peers.len(), 2);
        assert!(mesh_peers.contains(&peer1));
        assert!(mesh_peers.contains(&peer2));

        // Remove peer from mesh
        manager.remove_peer_from_mesh(&topic, &peer1).unwrap();
        let mesh_peers = manager.get_mesh_peers(&topic).unwrap();
        assert_eq!(mesh_peers.len(), 1);
        assert!(!mesh_peers.contains(&peer1));
    }

    #[test]
    fn test_peers_to_prune() {
        let manager = GossipSubManager::new(GossipSubConfig::default());
        let topic = TopicId::content_announce();
        let peer1 = test_peer_id(1);
        let peer2 = test_peer_id(2);

        manager.subscribe(topic.clone()).unwrap();
        manager.add_peer_to_mesh(&topic, peer1).unwrap();
        manager.add_peer_to_mesh(&topic, peer2).unwrap();

        // Set scores
        manager.update_peer_score(&peer1, topic.clone(), 0.9);
        manager.update_peer_score(&peer2, topic.clone(), 0.3);

        // Get peers below threshold
        let to_prune = manager.get_peers_to_prune(&topic, 0.5);
        assert_eq!(to_prune.len(), 1);
        assert!(to_prune.contains(&peer2));
    }

    #[test]
    fn test_topic_ids() {
        assert_eq!(
            TopicId::content_announce().0,
            "/ipfrs/content/announce/1.0.0"
        );
        assert_eq!(TopicId::peer_announce().0, "/ipfrs/peer/announce/1.0.0");
        assert_eq!(TopicId::dht_events().0, "/ipfrs/dht/events/1.0.0");
    }

    #[test]
    fn test_message_id_generation() {
        let peer = test_peer_id(1);
        let id1 = MessageId::new(&peer, 1);
        let id2 = MessageId::new(&peer, 1);
        let id3 = MessageId::new(&peer, 2);

        assert_eq!(id1, id2); // Same peer and sequence
        assert_ne!(id1, id3); // Different sequence
    }

    #[test]
    fn test_peer_score_calculation() {
        let mut score = PeerScore::default();

        score.update_topic_score(TopicId::content_announce(), 0.8);
        score.update_topic_score(TopicId::peer_announce(), 0.6);

        assert_eq!(score.topic_scores.len(), 2);
        assert_eq!(score.total_score, 0.7); // Average: (0.8 + 0.6) / 2

        // Record invalid message
        score.record_message(false);
        assert!(score.total_score < 0.7); // Score should decrease
    }
}
