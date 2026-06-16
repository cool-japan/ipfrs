//! WebSocket Support for Real-Time IPFRS Communication
//!
//! Provides:
//! - WebSocket upgrade handler
//! - Message routing and handling
//! - Pub/sub pattern for subscriptions
//! - Real-time event notifications (block additions, peer connections, etc.)

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::Response,
};
use futures::{sink::SinkExt, stream::StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

// ============================================================================
// WebSocket Message Types
// ============================================================================

/// WebSocket message envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum WsMessage {
    /// Subscribe to a topic
    Subscribe {
        topic: String,
        filter: Option<String>,
    },
    /// Unsubscribe from a topic
    Unsubscribe { topic: String },
    /// Event notification
    Event {
        topic: String,
        data: serde_json::Value,
    },
    /// Ping message for keepalive
    Ping,
    /// Pong response
    Pong,
    /// Error message
    Error { code: u16, message: String },
}

// ============================================================================
// Event Types
// ============================================================================

/// Real-time event types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event_type", rename_all = "snake_case")]
pub enum RealtimeEvent {
    /// Block was added to the store
    BlockAdded {
        cid: String,
        size: usize,
        timestamp: u64,
    },
    /// Block was removed from the store
    BlockRemoved { cid: String, timestamp: u64 },
    /// Peer connected
    PeerConnected {
        peer_id: String,
        address: String,
        timestamp: u64,
    },
    /// Peer disconnected
    PeerDisconnected { peer_id: String, timestamp: u64 },
    /// DHT query started
    DhtQueryStarted { query_id: String, key: String },
    /// DHT query progress
    DhtQueryProgress {
        query_id: String,
        peers_queried: usize,
        results_found: usize,
    },
    /// DHT query completed
    DhtQueryCompleted {
        query_id: String,
        success: bool,
        results: usize,
    },
}

impl RealtimeEvent {
    /// Get the topic for this event
    pub fn topic(&self) -> &str {
        match self {
            RealtimeEvent::BlockAdded { .. } | RealtimeEvent::BlockRemoved { .. } => "blocks",
            RealtimeEvent::PeerConnected { .. } | RealtimeEvent::PeerDisconnected { .. } => "peers",
            RealtimeEvent::DhtQueryStarted { .. }
            | RealtimeEvent::DhtQueryProgress { .. }
            | RealtimeEvent::DhtQueryCompleted { .. } => "dht",
        }
    }
}

// ============================================================================
// Subscription Manager
// ============================================================================

/// Manages WebSocket subscriptions and pub/sub
#[derive(Clone)]
pub struct SubscriptionManager {
    /// Topic-based broadcast channels
    topics: Arc<RwLock<HashMap<String, broadcast::Sender<RealtimeEvent>>>>,
    /// Active subscriptions per connection
    subscriptions: Arc<RwLock<HashMap<Uuid, Vec<String>>>>,
}

impl SubscriptionManager {
    /// Create a new subscription manager
    pub fn new() -> Self {
        Self {
            topics: Arc::new(RwLock::new(HashMap::new())),
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Subscribe a connection to a topic
    pub async fn subscribe(
        &self,
        connection_id: Uuid,
        topic: String,
    ) -> Result<broadcast::Receiver<RealtimeEvent>, WsError> {
        let mut topics = self.topics.write().await;

        // Get or create topic channel
        let sender = topics
            .entry(topic.clone())
            .or_insert_with(|| {
                let (tx, _rx) = broadcast::channel(100);
                info!("Created new topic channel: {}", topic);
                tx
            })
            .clone();

        // Track subscription
        let mut subs = self.subscriptions.write().await;
        subs.entry(connection_id).or_default().push(topic.clone());

        info!(
            "Connection {} subscribed to topic: {}",
            connection_id, topic
        );

        Ok(sender.subscribe())
    }

    /// Unsubscribe a connection from a topic
    pub async fn unsubscribe(&self, connection_id: Uuid, topic: &str) {
        let mut subs = self.subscriptions.write().await;
        if let Some(topics) = subs.get_mut(&connection_id) {
            topics.retain(|t| t != topic);
            info!(
                "Connection {} unsubscribed from topic: {}",
                connection_id, topic
            );
        }
    }

    /// Remove all subscriptions for a connection
    pub async fn remove_connection(&self, connection_id: Uuid) {
        let mut subs = self.subscriptions.write().await;
        subs.remove(&connection_id);
        info!(
            "Removed all subscriptions for connection: {}",
            connection_id
        );
    }

    /// Publish an event to a topic
    pub async fn publish(&self, event: RealtimeEvent) -> Result<usize, WsError> {
        let topic = event.topic().to_string();
        let topics = self.topics.read().await;

        if let Some(sender) = topics.get(&topic) {
            match sender.send(event.clone()) {
                Ok(count) => {
                    debug!(
                        "Published event to {} subscribers on topic: {}",
                        count, topic
                    );
                    Ok(count)
                }
                Err(_) => {
                    warn!("No active subscribers for topic: {}", topic);
                    Ok(0)
                }
            }
        } else {
            debug!("Topic not found: {}", topic);
            Ok(0)
        }
    }

    /// Get active subscription count
    pub async fn subscription_count(&self) -> usize {
        let subs = self.subscriptions.read().await;
        subs.len()
    }

    /// Get topic count
    pub async fn topic_count(&self) -> usize {
        let topics = self.topics.read().await;
        topics.len()
    }
}

impl Default for SubscriptionManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// WebSocket Handler
// ============================================================================

/// WebSocket handler state
#[derive(Clone)]
pub struct WsState {
    pub subscription_manager: SubscriptionManager,
}

impl WsState {
    /// Create new WebSocket state
    pub fn new() -> Self {
        Self {
            subscription_manager: SubscriptionManager::new(),
        }
    }
}

impl Default for WsState {
    fn default() -> Self {
        Self::new()
    }
}

/// WebSocket upgrade handler
///
/// GET /ws
pub async fn ws_handler(ws: WebSocketUpgrade, State(state): State<WsState>) -> Response {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

/// Handle individual WebSocket connection
#[allow(clippy::too_many_arguments)]
async fn handle_socket(socket: WebSocket, state: WsState) {
    let connection_id = Uuid::new_v4();
    info!("New WebSocket connection: {}", connection_id);

    let (sender, receiver) = socket.split();
    let sender = Arc::new(tokio::sync::Mutex::new(sender));

    // Subscriptions for this connection
    let active_subscriptions: Arc<
        tokio::sync::Mutex<HashMap<String, broadcast::Receiver<RealtimeEvent>>>,
    > = Arc::new(tokio::sync::Mutex::new(HashMap::new()));

    // Spawn task to handle outgoing events
    let sender_clone = sender.clone();
    let subs_clone = active_subscriptions.clone();
    let event_task = tokio::spawn(async move {
        loop {
            // Check all active subscriptions for events
            let mut subs = subs_clone.lock().await;
            let topics: Vec<String> = subs.keys().cloned().collect();

            for topic in topics {
                if let Some(rx) = subs.get_mut(&topic) {
                    match rx.try_recv() {
                        Ok(event) => {
                            let msg = WsMessage::Event {
                                topic: topic.clone(),
                                data: serde_json::to_value(&event).unwrap_or_default(),
                            };

                            if let Ok(json) = serde_json::to_string(&msg) {
                                let mut tx = sender_clone.lock().await;
                                if tx.send(Message::Text(json.into())).await.is_err() {
                                    return;
                                }
                            }
                        }
                        Err(broadcast::error::TryRecvError::Empty) => {}
                        Err(_) => {}
                    }
                }
            }

            drop(subs);
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
    });

    // Handle incoming messages
    let mut receiver = receiver;
    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(Message::Text(text)) => match serde_json::from_str::<WsMessage>(&text) {
                Ok(ws_msg) => match ws_msg {
                    WsMessage::Subscribe { topic, filter } => {
                        debug!(
                            "Connection {} subscribing to topic: {} (filter: {:?})",
                            connection_id, topic, filter
                        );

                        match state
                            .subscription_manager
                            .subscribe(connection_id, topic.clone())
                            .await
                        {
                            Ok(rx) => {
                                let mut subs = active_subscriptions.lock().await;
                                subs.insert(topic, rx);
                            }
                            Err(e) => {
                                error!("Failed to subscribe: {}", e);
                                let error_msg = WsMessage::Error {
                                    code: 500,
                                    message: format!("Subscription failed: {}", e),
                                };
                                if let Ok(json) = serde_json::to_string(&error_msg) {
                                    let mut tx = sender.lock().await;
                                    let _ = tx.send(Message::Text(json.into())).await;
                                }
                            }
                        }
                    }
                    WsMessage::Unsubscribe { topic } => {
                        debug!(
                            "Connection {} unsubscribing from topic: {}",
                            connection_id, topic
                        );
                        state
                            .subscription_manager
                            .unsubscribe(connection_id, &topic)
                            .await;
                        let mut subs = active_subscriptions.lock().await;
                        subs.remove(&topic);
                    }
                    WsMessage::Ping => {
                        let pong = WsMessage::Pong;
                        if let Ok(json) = serde_json::to_string(&pong) {
                            let mut tx = sender.lock().await;
                            let _ = tx.send(Message::Text(json.into())).await;
                        }
                    }
                    _ => {
                        warn!("Unexpected message type from client");
                    }
                },
                Err(e) => {
                    error!("Failed to parse message: {}", e);
                    let error_msg = WsMessage::Error {
                        code: 400,
                        message: format!("Invalid message format: {}", e),
                    };
                    if let Ok(json) = serde_json::to_string(&error_msg) {
                        let mut tx = sender.lock().await;
                        let _ = tx.send(Message::Text(json.into())).await;
                    }
                }
            },
            Ok(Message::Close(_)) => {
                info!("Connection {} closed by client", connection_id);
                break;
            }
            Err(e) => {
                error!("WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }

    // Cleanup
    event_task.abort();
    state
        .subscription_manager
        .remove_connection(connection_id)
        .await;
    info!("Connection {} disconnected", connection_id);
}

// ============================================================================
// Error Types
// ============================================================================

/// WebSocket errors
#[derive(Debug, Error)]
pub enum WsError {
    #[error("Subscription error: {0}")]
    SubscriptionError(String),

    #[error("Invalid topic: {0}")]
    InvalidTopic(String),

    #[error("Send error: {0}")]
    SendError(String),
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_subscription_manager_new() {
        let manager = SubscriptionManager::new();
        assert_eq!(manager.subscription_count().await, 0);
        assert_eq!(manager.topic_count().await, 0);
    }

    #[tokio::test]
    async fn test_subscribe_and_publish() {
        let manager = SubscriptionManager::new();
        let conn_id = Uuid::new_v4();

        // Subscribe to blocks topic
        let mut rx = manager
            .subscribe(conn_id, "blocks".to_string())
            .await
            .expect("test: subscription to blocks topic should succeed");

        // Publish an event
        let event = RealtimeEvent::BlockAdded {
            cid: "QmTest".to_string(),
            size: 1024,
            timestamp: 12345,
        };

        let count = manager
            .publish(event.clone())
            .await
            .expect("test: publish to subscribed topic should succeed");
        assert_eq!(count, 1);

        // Receive the event
        let received = rx
            .recv()
            .await
            .expect("test: event should be received from subscription channel");
        match received {
            RealtimeEvent::BlockAdded { cid, size, .. } => {
                assert_eq!(cid, "QmTest");
                assert_eq!(size, 1024);
            }
            _ => panic!("Wrong event type"),
        }
    }

    #[tokio::test]
    async fn test_unsubscribe() {
        let manager = SubscriptionManager::new();
        let conn_id = Uuid::new_v4();

        // Subscribe
        let _rx = manager
            .subscribe(conn_id, "blocks".to_string())
            .await
            .expect("test: subscription should succeed");
        assert_eq!(manager.subscription_count().await, 1);

        // Unsubscribe
        manager.unsubscribe(conn_id, "blocks").await;
        assert_eq!(manager.subscription_count().await, 1); // Connection still tracked

        // Remove connection
        manager.remove_connection(conn_id).await;
        assert_eq!(manager.subscription_count().await, 0);
    }

    #[tokio::test]
    async fn test_multiple_subscribers() {
        let manager = SubscriptionManager::new();
        let conn1 = Uuid::new_v4();
        let conn2 = Uuid::new_v4();

        // Subscribe both connections
        let mut rx1 = manager
            .subscribe(conn1, "blocks".to_string())
            .await
            .expect("test: first subscriber connection should succeed");
        let mut rx2 = manager
            .subscribe(conn2, "blocks".to_string())
            .await
            .expect("test: second subscriber connection should succeed");

        // Publish event
        let event = RealtimeEvent::BlockAdded {
            cid: "QmTest".to_string(),
            size: 2048,
            timestamp: 12345,
        };

        let count = manager
            .publish(event)
            .await
            .expect("test: publish to multiple subscribers should succeed");
        assert_eq!(count, 2); // Both subscribers receive it

        // Both should receive
        assert!(rx1.recv().await.is_ok());
        assert!(rx2.recv().await.is_ok());
    }

    #[test]
    fn test_realtime_event_topic() {
        let block_event = RealtimeEvent::BlockAdded {
            cid: "test".to_string(),
            size: 100,
            timestamp: 123,
        };
        assert_eq!(block_event.topic(), "blocks");

        let peer_event = RealtimeEvent::PeerConnected {
            peer_id: "peer1".to_string(),
            address: "addr1".to_string(),
            timestamp: 123,
        };
        assert_eq!(peer_event.topic(), "peers");

        let dht_event = RealtimeEvent::DhtQueryStarted {
            query_id: "q1".to_string(),
            key: "key1".to_string(),
        };
        assert_eq!(dht_event.topic(), "dht");
    }

    #[test]
    fn test_ws_message_serialization() {
        let subscribe = WsMessage::Subscribe {
            topic: "blocks".to_string(),
            filter: Some("cid=Qm*".to_string()),
        };

        let json = serde_json::to_string(&subscribe)
            .expect("test: WsMessage serialization to JSON should succeed");
        assert!(json.contains("subscribe"));
        assert!(json.contains("blocks"));

        let deserialized: WsMessage = serde_json::from_str(&json)
            .expect("test: WsMessage deserialization from JSON should succeed");
        match deserialized {
            WsMessage::Subscribe { topic, .. } => {
                assert_eq!(topic, "blocks");
            }
            _ => panic!("Wrong message type"),
        }
    }
}
