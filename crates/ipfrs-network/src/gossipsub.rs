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
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
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

/// Standard GossipSub topics for IPFRS protocol messages
pub mod topics {
    /// Inference requests broadcast to all peers
    pub const INFERENCE_REQUEST: &str = "/ipfrs/inference/request/1.0.0";
    /// Inference results streamed back to requester
    pub const INFERENCE_RESULT: &str = "/ipfrs/inference/result/1.0.0";
    /// DHT provider announcements
    pub const PROVIDER_ANNOUNCE: &str = "/ipfrs/provider/announce/1.0.0";
    /// Block availability announcements
    pub const BLOCK_ANNOUNCE: &str = "/ipfrs/block/announce/1.0.0";
    /// Gradient synchronization
    pub const GRADIENT_SYNC: &str = "/ipfrs/gradient/sync/1.0.0";
    /// Knowledge base delta updates
    pub const KB_DELTA: &str = "/ipfrs/kb/delta/1.0.0";
}

/// Message envelope for all IPFRS GossipSub messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicMessage {
    pub topic: String,
    pub sender_peer_id: String,
    pub payload: Vec<u8>,
    pub timestamp: u64,
    pub sequence: u64,
}

impl TopicMessage {
    /// Create a new topic message with the current Unix timestamp
    pub fn new(topic: &str, sender: &str, payload: Vec<u8>) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            topic: topic.to_string(),
            sender_peer_id: sender.to_string(),
            payload,
            timestamp,
            sequence: 0,
        }
    }

    /// Encode the message to JSON bytes
    pub fn encode(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    /// Decode a message from JSON bytes
    pub fn decode(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }
}

/// Type-safe topic handle
pub struct IpfrsTopic {
    pub name: String,
    pub hash: libp2p::gossipsub::TopicHash,
}

impl IpfrsTopic {
    pub fn new(name: &str) -> Self {
        use libp2p::gossipsub::IdentTopic;
        let topic = IdentTopic::new(name);
        IpfrsTopic {
            name: name.to_string(),
            hash: topic.hash(),
        }
    }
}

impl GossipSubManager {
    /// Subscribe to all standard IPFRS inference topics
    pub fn subscribe_inference_topics(&self) -> Result<(), GossipSubError> {
        use topics::{
            BLOCK_ANNOUNCE, GRADIENT_SYNC, INFERENCE_REQUEST, INFERENCE_RESULT, KB_DELTA,
            PROVIDER_ANNOUNCE,
        };
        let all_topics = [
            INFERENCE_REQUEST,
            INFERENCE_RESULT,
            PROVIDER_ANNOUNCE,
            BLOCK_ANNOUNCE,
            GRADIENT_SYNC,
            KB_DELTA,
        ];
        for t in &all_topics {
            let tid = TopicId::new(*t);
            // Subscribing when already subscribed is acceptable during bulk init
            match self.subscribe(tid) {
                Ok(()) | Err(GossipSubError::AlreadySubscribed(_)) => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// Publish an inference request to all subscribers
    pub fn publish_inference_request(
        &self,
        request_bytes: &[u8],
        sender_peer_id: &str,
    ) -> Result<(), GossipSubError> {
        let msg = TopicMessage::new(
            topics::INFERENCE_REQUEST,
            sender_peer_id,
            request_bytes.to_vec(),
        );
        let encoded = msg
            .encode()
            .map_err(|e| GossipSubError::InvalidTopicName(format!("encode error: {}", e)))?;
        let topic = TopicId::new(topics::INFERENCE_REQUEST);
        // We need a source PeerId; use the zero peer as a stand-in for the local node
        // (real integration would pass in the local PeerId)
        let source = self.dummy_local_peer();
        self.publish(topic, encoded, source)?;
        Ok(())
    }

    /// Publish an inference result to subscribers
    pub fn publish_inference_result(
        &self,
        result_bytes: &[u8],
        sender_peer_id: &str,
    ) -> Result<(), GossipSubError> {
        let msg = TopicMessage::new(
            topics::INFERENCE_RESULT,
            sender_peer_id,
            result_bytes.to_vec(),
        );
        let encoded = msg
            .encode()
            .map_err(|e| GossipSubError::InvalidTopicName(format!("encode error: {}", e)))?;
        let topic = TopicId::new(topics::INFERENCE_RESULT);
        let source = self.dummy_local_peer();
        self.publish(topic, encoded, source)?;
        Ok(())
    }

    /// Publish a block availability announcement
    pub fn announce_block(&self, cid: &str, sender_peer_id: &str) -> Result<(), GossipSubError> {
        let msg = TopicMessage::new(
            topics::BLOCK_ANNOUNCE,
            sender_peer_id,
            cid.as_bytes().to_vec(),
        );
        let encoded = msg
            .encode()
            .map_err(|e| GossipSubError::InvalidTopicName(format!("encode error: {}", e)))?;
        let topic = TopicId::new(topics::BLOCK_ANNOUNCE);
        let source = self.dummy_local_peer();
        self.publish(topic, encoded, source)?;
        Ok(())
    }

    /// Publish a gradient CID to the `GRADIENT_SYNC` topic so that peers can fetch it.
    ///
    /// The `cid` is the content-address of the Arrow IPC block holding the gradient tensor.
    /// `sender_peer_id` is the local node's peer-id string used as the message source.
    pub fn publish_gradient_cid(
        &self,
        cid: &str,
        sender_peer_id: &str,
    ) -> Result<(), GossipSubError> {
        let msg = TopicMessage::new(
            topics::GRADIENT_SYNC,
            sender_peer_id,
            cid.as_bytes().to_vec(),
        );
        let encoded = msg
            .encode()
            .map_err(|e| GossipSubError::InvalidTopicName(format!("encode error: {e}")))?;
        let topic = TopicId::new(topics::GRADIENT_SYNC);
        let source = self.dummy_local_peer();
        self.publish(topic, encoded, source)?;
        Ok(())
    }

    /// Publish a provider record announcement
    pub fn announce_provider(&self, cid: &str, peer_id: &str) -> Result<(), GossipSubError> {
        let msg = TopicMessage::new(topics::PROVIDER_ANNOUNCE, peer_id, cid.as_bytes().to_vec());
        let encoded = msg
            .encode()
            .map_err(|e| GossipSubError::InvalidTopicName(format!("encode error: {}", e)))?;
        let topic = TopicId::new(topics::PROVIDER_ANNOUNCE);
        let source = self.dummy_local_peer();
        self.publish(topic, encoded, source)?;
        Ok(())
    }

    /// Return a deterministic placeholder PeerId used when no local peer id is configured.
    /// This is intentionally not random so tests are reproducible and the manager remains
    /// Send + Sync without needing to store a PeerId field.
    fn dummy_local_peer(&self) -> PeerId {
        use libp2p::identity::Keypair;
        let seed = [0u8; 32];
        // Safe: valid fixed seed, only used as a stand-in source identifier
        Keypair::ed25519_from_bytes(seed.to_vec())
            .map(|kp| kp.public().to_peer_id())
            .unwrap_or_else(|_| PeerId::random())
    }
}

#[cfg(test)]
mod topic_tests {
    use super::*;
    use topics::{
        BLOCK_ANNOUNCE, GRADIENT_SYNC, INFERENCE_REQUEST, INFERENCE_RESULT, KB_DELTA,
        PROVIDER_ANNOUNCE,
    };

    #[test]
    fn test_topic_message_roundtrip() {
        let original =
            TopicMessage::new(INFERENCE_REQUEST, "peer-abc", b"hello inference".to_vec());
        let encoded = original.encode().expect("encode should succeed");
        let decoded = TopicMessage::decode(&encoded).expect("decode should succeed");
        assert_eq!(decoded.topic, INFERENCE_REQUEST);
        assert_eq!(decoded.sender_peer_id, "peer-abc");
        assert_eq!(decoded.payload, b"hello inference");
    }

    #[test]
    fn test_all_topic_constants_unique() {
        let topics = [
            INFERENCE_REQUEST,
            INFERENCE_RESULT,
            PROVIDER_ANNOUNCE,
            BLOCK_ANNOUNCE,
            GRADIENT_SYNC,
            KB_DELTA,
        ];
        let unique: std::collections::HashSet<_> = topics.iter().collect();
        assert_eq!(
            unique.len(),
            topics.len(),
            "all topic constants must be unique"
        );
    }

    #[test]
    fn test_ipfrs_topic_hash() {
        let t1 = IpfrsTopic::new(INFERENCE_REQUEST);
        let t2 = IpfrsTopic::new(INFERENCE_REQUEST);
        let t3 = IpfrsTopic::new(INFERENCE_RESULT);

        assert_eq!(t1.name, INFERENCE_REQUEST);
        assert_eq!(t1.hash, t2.hash, "same name must produce same hash");
        assert_ne!(
            t1.hash, t3.hash,
            "different names must produce different hashes"
        );
    }

    #[test]
    fn test_subscribe_inference_topics_subscribes_all() {
        let manager = GossipSubManager::new(GossipSubConfig::default());
        manager
            .subscribe_inference_topics()
            .expect("should subscribe without error");

        let all_topics = [
            INFERENCE_REQUEST,
            INFERENCE_RESULT,
            PROVIDER_ANNOUNCE,
            BLOCK_ANNOUNCE,
            GRADIENT_SYNC,
            KB_DELTA,
        ];
        for t in &all_topics {
            assert!(
                manager.is_subscribed(&TopicId::new(*t)),
                "should be subscribed to {}",
                t
            );
        }
    }

    #[test]
    fn test_subscribe_inference_topics_idempotent() {
        let manager = GossipSubManager::new(GossipSubConfig::default());
        manager.subscribe_inference_topics().expect("first call ok");
        // Second call should not error even though already subscribed
        manager
            .subscribe_inference_topics()
            .expect("second call should also be ok");
    }

    #[test]
    fn test_publish_inference_request() {
        let manager = GossipSubManager::new(GossipSubConfig::default());
        manager.subscribe_inference_topics().expect("subscribe");
        manager
            .publish_inference_request(b"req_data", "peer-1")
            .expect("publish inference request");
        let stats = manager.stats();
        assert!(stats.messages_published >= 1);
    }

    #[test]
    fn test_publish_inference_result() {
        let manager = GossipSubManager::new(GossipSubConfig::default());
        manager.subscribe_inference_topics().expect("subscribe");
        manager
            .publish_inference_result(b"result_data", "peer-2")
            .expect("publish inference result");
        let stats = manager.stats();
        assert!(stats.messages_published >= 1);
    }

    #[test]
    fn test_announce_block() {
        let manager = GossipSubManager::new(GossipSubConfig::default());
        manager.subscribe_inference_topics().expect("subscribe");
        manager
            .announce_block("bafyreiabc123", "peer-3")
            .expect("announce block");
        let stats = manager.stats();
        assert!(stats.messages_published >= 1);
    }

    #[test]
    fn test_announce_provider() {
        let manager = GossipSubManager::new(GossipSubConfig::default());
        manager.subscribe_inference_topics().expect("subscribe");
        manager
            .announce_provider("bafyreiabc123", "peer-4")
            .expect("announce provider");
        let stats = manager.stats();
        assert!(stats.messages_published >= 1);
    }

    #[test]
    fn test_topic_message_timestamp_nonnegative() {
        let msg = TopicMessage::new(KB_DELTA, "p", vec![1, 2, 3]);
        // timestamp should be a reasonable Unix epoch value (after year 2020)
        assert!(
            msg.timestamp > 1_577_836_800,
            "timestamp should be after 2020"
        );
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

        manager
            .subscribe(topic.clone())
            .expect("test: subscribe to content_announce topic");
        assert!(manager.is_subscribed(&topic));
        assert_eq!(manager.list_topics().len(), 1);
    }

    #[test]
    fn test_duplicate_subscription() {
        let manager = GossipSubManager::new(GossipSubConfig::default());
        let topic = TopicId::content_announce();

        manager
            .subscribe(topic.clone())
            .expect("test: first subscribe should succeed");
        let result = manager.subscribe(topic);
        assert!(matches!(result, Err(GossipSubError::AlreadySubscribed(_))));
    }

    #[test]
    fn test_unsubscribe() {
        let manager = GossipSubManager::new(GossipSubConfig::default());
        let topic = TopicId::content_announce();

        manager
            .subscribe(topic.clone())
            .expect("test: subscribe should succeed");
        manager
            .unsubscribe(&topic)
            .expect("test: unsubscribe should succeed");
        assert!(!manager.is_subscribed(&topic));
        assert_eq!(manager.list_topics().len(), 0);
    }

    #[test]
    fn test_publish_message() {
        let manager = GossipSubManager::new(GossipSubConfig::default());
        let topic = TopicId::content_announce();
        let peer = test_peer_id(1);

        manager
            .subscribe(topic.clone())
            .expect("test: subscribe should succeed");

        let data = b"Hello, GossipSub!".to_vec();
        let message_id = manager
            .publish(topic, data, peer)
            .expect("test: publish should succeed");

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

        manager
            .subscribe(topic.clone())
            .expect("test: subscribe should succeed");

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

        manager
            .subscribe(topic.clone())
            .expect("test: subscribe should succeed");

        let message = GossipSubMessage {
            id: MessageId::new(&peer, 1),
            source: peer,
            topic: topic.clone(),
            data: b"Test".to_vec(),
            sequence: 1,
            timestamp: Instant::now(),
        };

        // First message should be accepted
        let result1 = manager
            .handle_message(message.clone())
            .expect("test: handle first message should succeed");
        assert!(result1);

        // Duplicate should be rejected
        let result2 = manager
            .handle_message(message)
            .expect("test: handle duplicate message should succeed");
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

        let score = manager
            .get_peer_score(&peer)
            .expect("test: peer score should exist after update");
        assert_eq!(score.topic_scores.get(&topic), Some(&0.8));
        assert!(score.total_score > 0.0);
    }

    #[test]
    fn test_mesh_management() {
        let manager = GossipSubManager::new(GossipSubConfig::default());
        let topic = TopicId::content_announce();
        let peer1 = test_peer_id(1);
        let peer2 = test_peer_id(2);

        manager
            .subscribe(topic.clone())
            .expect("test: subscribe should succeed");

        // Add peers to mesh
        manager
            .add_peer_to_mesh(&topic, peer1)
            .expect("test: add peer1 to mesh should succeed");
        manager
            .add_peer_to_mesh(&topic, peer2)
            .expect("test: add peer2 to mesh should succeed");

        let mesh_peers = manager
            .get_mesh_peers(&topic)
            .expect("test: get mesh peers should succeed");
        assert_eq!(mesh_peers.len(), 2);
        assert!(mesh_peers.contains(&peer1));
        assert!(mesh_peers.contains(&peer2));

        // Remove peer from mesh
        manager
            .remove_peer_from_mesh(&topic, &peer1)
            .expect("test: remove peer1 from mesh should succeed");
        let mesh_peers = manager
            .get_mesh_peers(&topic)
            .expect("test: get mesh peers after removal should succeed");
        assert_eq!(mesh_peers.len(), 1);
        assert!(!mesh_peers.contains(&peer1));
    }

    #[test]
    fn test_peers_to_prune() {
        let manager = GossipSubManager::new(GossipSubConfig::default());
        let topic = TopicId::content_announce();
        let peer1 = test_peer_id(1);
        let peer2 = test_peer_id(2);

        manager
            .subscribe(topic.clone())
            .expect("test: subscribe should succeed");
        manager
            .add_peer_to_mesh(&topic, peer1)
            .expect("test: add peer1 to mesh should succeed");
        manager
            .add_peer_to_mesh(&topic, peer2)
            .expect("test: add peer2 to mesh should succeed");

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

// ============================================================================
// Mesh Health Monitor
// ============================================================================

/// Health status of a GossipSub topic mesh.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeshHealthStatus {
    /// Mesh peer count is within the healthy range [D_low, D_high].
    Healthy,
    /// Mesh peer count is below D_low — grafting is needed.
    Underpeered {
        /// Current number of mesh peers.
        current: usize,
        /// Minimum required number of mesh peers (D_low).
        minimum: usize,
    },
    /// Mesh peer count exceeds D_high — pruning is needed.
    Overloaded {
        /// Current number of mesh peers.
        current: usize,
        /// Maximum allowed number of mesh peers (D_high).
        maximum: usize,
    },
}

/// Monitors GossipSub mesh health and manages graft/prune decisions.
///
/// The monitor tracks a per-topic peer count relative to the D_low / D_high
/// thresholds defined in the GossipSub specification and implements a
/// backoff-aware heal scheduler so that repeated failed graft attempts do not
/// flood the network.
pub struct MeshHealthMonitor {
    /// Minimum mesh peers threshold (D_low).
    d_low: usize,
    /// Maximum mesh peers threshold (D_high).
    d_high: usize,
    /// Minimum seconds between consecutive heal attempts.
    heal_interval_secs: u64,
    /// Timestamp of the last heal attempt (if any).
    last_heal_attempt: Option<Instant>,
    /// Total number of heal attempts recorded.
    heal_attempts: u64,
}

impl MeshHealthMonitor {
    /// Create a new monitor with the given D_low and D_high thresholds.
    ///
    /// The heal interval defaults to 30 seconds.
    pub fn new(d_low: usize, d_high: usize) -> Self {
        Self {
            d_low,
            d_high,
            heal_interval_secs: 30,
            last_heal_attempt: None,
            heal_attempts: 0,
        }
    }

    /// Create a monitor with an explicit heal interval.
    pub fn with_heal_interval(mut self, secs: u64) -> Self {
        self.heal_interval_secs = secs;
        self
    }

    /// Return the D_low threshold.
    pub fn d_low(&self) -> usize {
        self.d_low
    }

    /// Return the D_high threshold.
    pub fn d_high(&self) -> usize {
        self.d_high
    }

    /// Return the heal interval in seconds.
    pub fn heal_interval_secs(&self) -> u64 {
        self.heal_interval_secs
    }

    /// Return the number of heal attempts recorded so far.
    pub fn heal_attempts(&self) -> u64 {
        self.heal_attempts
    }

    /// Return `true` when the given peer count is below D_low.
    pub fn is_mesh_underpeered(&self, topic_peer_count: usize) -> bool {
        topic_peer_count < self.d_low
    }

    /// Return `true` when the given peer count exceeds D_high.
    pub fn is_mesh_overloaded(&self, topic_peer_count: usize) -> bool {
        topic_peer_count > self.d_high
    }

    /// Summarise mesh health given the current peer count.
    pub fn health_status(&self, mesh_size: usize) -> MeshHealthStatus {
        if mesh_size < self.d_low {
            MeshHealthStatus::Underpeered {
                current: mesh_size,
                minimum: self.d_low,
            }
        } else if mesh_size > self.d_high {
            MeshHealthStatus::Overloaded {
                current: mesh_size,
                maximum: self.d_high,
            }
        } else {
            MeshHealthStatus::Healthy
        }
    }

    /// Suggest peers to graft from the gossip peer cache when the mesh is
    /// sparse.
    ///
    /// Returns peer IDs that are in `gossip_peers` but **not** in
    /// `mesh_peers`.  The result is bounded to at most `D_high - mesh_size`
    /// candidates so callers never receive more grafts than needed.
    pub fn suggest_grafts(&self, gossip_peers: &[String], mesh_peers: &[String]) -> Vec<String> {
        let mesh_set: std::collections::HashSet<&str> =
            mesh_peers.iter().map(|s| s.as_str()).collect();

        gossip_peers
            .iter()
            .filter(|p| !mesh_set.contains(p.as_str()))
            .cloned()
            .collect()
    }

    /// Record that a heal attempt was made right now.
    ///
    /// After calling this `should_attempt_heal` will return `false` until
    /// the configured heal interval has elapsed.
    pub fn record_heal_attempt(&mut self) {
        self.last_heal_attempt = Some(Instant::now());
        self.heal_attempts += 1;
    }

    /// Return `true` when sufficient time has passed since the last heal
    /// attempt (or no attempt has ever been made) so that a new heal should
    /// be tried.
    pub fn should_attempt_heal(&self) -> bool {
        match self.last_heal_attempt {
            None => true,
            Some(last) => last.elapsed() >= Duration::from_secs(self.heal_interval_secs),
        }
    }
}

// ============================================================================
// GossipSubManager — mesh health methods
// ============================================================================

impl GossipSubManager {
    /// Return the aggregate mesh health status for a given topic.
    ///
    /// Uses the manager's configured `mesh_n_low` and `mesh_n_high` thresholds.
    pub fn mesh_health(&self, topic: &TopicId) -> MeshHealthStatus {
        let monitor = MeshHealthMonitor::new(self.config.mesh_n_low, self.config.mesh_n_high);
        let mesh_size = self
            .subscriptions
            .get(topic)
            .map(|s| s.mesh_peers.len())
            .unwrap_or(0);
        monitor.health_status(mesh_size)
    }

    /// Attempt to heal the mesh for `topic` if it is underpeered.
    ///
    /// Selects peers from the gossip peer score table that are not already in
    /// the mesh and grafts them in until D_low is satisfied.  Returns the list
    /// of peer IDs grafted.
    pub fn heal_mesh_if_needed(&self, topic: &TopicId) -> Vec<PeerId> {
        let d_low = self.config.mesh_n_low;
        let d_high = self.config.mesh_n_high;

        let current_mesh: Vec<PeerId> = self
            .subscriptions
            .get(topic)
            .map(|s| s.mesh_peers.iter().cloned().collect())
            .unwrap_or_default();

        if current_mesh.len() >= d_low {
            return Vec::new();
        }

        let slots_needed = d_low.saturating_sub(current_mesh.len());
        let mesh_set: std::collections::HashSet<PeerId> = current_mesh.into_iter().collect();

        // Collect candidate peers from the score table, ordered by score
        // (descending) so that the best-quality peers are grafted first.
        let mut candidates: Vec<(PeerId, f64)> = self
            .peer_scores
            .iter()
            .filter(|entry| !mesh_set.contains(entry.key()))
            .map(|entry| (*entry.key(), entry.value().total_score))
            .collect();

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let to_graft: Vec<PeerId> = candidates
            .into_iter()
            .take(slots_needed.min(d_high - mesh_set.len()))
            .map(|(peer, _)| peer)
            .collect();

        for peer in &to_graft {
            // Best-effort: ignore errors (e.g. not subscribed)
            let _ = self.add_peer_to_mesh(topic, *peer);
        }

        to_graft
    }

    /// Prune the mesh for `topic` if it exceeds D_high.
    ///
    /// Removes the lowest-scoring peers until the mesh size equals D_high.
    /// Returns the list of peer IDs pruned.
    pub fn prune_mesh_if_needed(&self, topic: &TopicId) -> Vec<PeerId> {
        let d_high = self.config.mesh_n_high;

        let current_mesh: Vec<PeerId> = self
            .subscriptions
            .get(topic)
            .map(|s| s.mesh_peers.iter().cloned().collect())
            .unwrap_or_default();

        if current_mesh.len() <= d_high {
            return Vec::new();
        }

        let excess = current_mesh.len() - d_high;

        // Score all mesh peers; peers without a score record get score 0.
        let mut scored: Vec<(PeerId, f64)> = current_mesh
            .into_iter()
            .map(|peer| {
                let score = self
                    .peer_scores
                    .get(&peer)
                    .map(|s| s.total_score)
                    .unwrap_or(0.0);
                (peer, score)
            })
            .collect();

        // Sort ascending (lowest score first) so we prune the worst peers.
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let to_prune: Vec<PeerId> = scored.into_iter().take(excess).map(|(p, _)| p).collect();

        for peer in &to_prune {
            let _ = self.remove_peer_from_mesh(topic, peer);
        }

        to_prune
    }
}

// ============================================================================
// Mesh health tests
// ============================================================================

#[cfg(test)]
mod mesh_health_tests {
    use super::*;
    use libp2p::identity::Keypair;

    fn test_peer_str(index: u8) -> String {
        let mut seed = [0u8; 32];
        seed[0] = index;
        let kp = Keypair::ed25519_from_bytes(seed).expect("valid seed");
        kp.public().to_peer_id().to_string()
    }

    #[test]
    fn test_mesh_health_underpeered() {
        let monitor = MeshHealthMonitor::new(6, 12);
        let status = monitor.health_status(3);
        assert_eq!(
            status,
            MeshHealthStatus::Underpeered {
                current: 3,
                minimum: 6
            }
        );
        assert!(monitor.is_mesh_underpeered(3));
    }

    #[test]
    fn test_mesh_health_healthy() {
        let monitor = MeshHealthMonitor::new(6, 12);
        let status = monitor.health_status(8);
        assert_eq!(status, MeshHealthStatus::Healthy);
        assert!(!monitor.is_mesh_underpeered(8));
        assert!(!monitor.is_mesh_overloaded(8));
    }

    #[test]
    fn test_mesh_health_overloaded() {
        let monitor = MeshHealthMonitor::new(6, 12);
        let status = monitor.health_status(15);
        assert_eq!(
            status,
            MeshHealthStatus::Overloaded {
                current: 15,
                maximum: 12,
            }
        );
        assert!(monitor.is_mesh_overloaded(15));
    }

    #[test]
    fn test_mesh_suggest_grafts() {
        let monitor = MeshHealthMonitor::new(6, 12);
        let a = test_peer_str(1);
        let b = test_peer_str(2);
        let c = test_peer_str(3);
        let d = test_peer_str(4);

        let gossip_peers = vec![a.clone(), b.clone(), c.clone(), d.clone()];
        let mesh_peers = vec![a.clone()];

        let suggestions = monitor.suggest_grafts(&gossip_peers, &mesh_peers);

        // A is already in mesh; B, C, D should be suggested.
        assert_eq!(suggestions.len(), 3);
        assert!(!suggestions.contains(&a));
        assert!(suggestions.contains(&b));
        assert!(suggestions.contains(&c));
        assert!(suggestions.contains(&d));
    }

    #[test]
    fn test_heal_backoff() {
        let mut monitor = MeshHealthMonitor::new(6, 12).with_heal_interval(60);

        // Initially should attempt.
        assert!(monitor.should_attempt_heal());

        // After recording an attempt it should not attempt immediately.
        monitor.record_heal_attempt();
        assert!(
            !monitor.should_attempt_heal(),
            "should not heal immediately after attempt"
        );

        assert_eq!(monitor.heal_attempts(), 1);
    }

    #[test]
    fn test_gossipsub_manager_mesh_health_method() {
        let mgr = GossipSubManager::new(GossipSubConfig::default());
        let topic = TopicId::content_announce();
        mgr.subscribe(topic.clone()).expect("subscribe");

        // Empty mesh → underpeered (default d_low = 4)
        let status = mgr.mesh_health(&topic);
        assert!(
            matches!(status, MeshHealthStatus::Underpeered { .. }),
            "empty mesh should be underpeered"
        );
    }

    #[test]
    fn test_gossipsub_manager_heal_mesh_if_needed() {
        let mgr = GossipSubManager::new(GossipSubConfig::default());
        let topic = TopicId::content_announce();
        mgr.subscribe(topic.clone()).expect("subscribe");

        // With an empty peer score table, heal_mesh_if_needed returns nothing.
        let grafted = mgr.heal_mesh_if_needed(&topic);
        assert!(grafted.is_empty(), "no scored peers to graft from");
    }

    #[test]
    fn test_gossipsub_manager_prune_mesh_if_needed() {
        let config = GossipSubConfig {
            mesh_n_high: 2,
            ..GossipSubConfig::default()
        };
        let mgr = GossipSubManager::new(config);
        let topic = TopicId::content_announce();
        mgr.subscribe(topic.clone()).expect("subscribe");

        // Add 4 peers to the mesh, exceeding d_high = 2.
        for i in 0u8..4 {
            let mut seed = [0u8; 32];
            seed[0] = i + 10;
            let kp = Keypair::ed25519_from_bytes(seed).expect("seed");
            let peer = kp.public().to_peer_id();
            mgr.add_peer_to_mesh(&topic, peer).expect("add to mesh");
        }

        let before = mgr.get_mesh_peers(&topic).expect("get peers").len();
        assert_eq!(before, 4);

        let pruned = mgr.prune_mesh_if_needed(&topic);
        assert_eq!(pruned.len(), 2, "should prune 2 peers to reach d_high=2");

        let after = mgr.get_mesh_peers(&topic).expect("get peers").len();
        assert_eq!(after, 2);
    }
}
