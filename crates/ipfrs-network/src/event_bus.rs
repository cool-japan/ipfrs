//! NetworkEventBus — synchronous in-process publish-subscribe bus for network events.
//!
//! Decouples protocol handlers from application logic by providing a simple,
//! zero-dependency pub/sub mechanism with per-subscriber event queues and
//! configurable event filtering.

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// NetworkEvent
// ---------------------------------------------------------------------------

/// All events that can be emitted by the network layer.
#[derive(Clone, Debug)]
pub enum NetworkEvent {
    /// A remote peer successfully connected.
    PeerConnected { peer_id: String, address: String },
    /// A remote peer disconnected.
    PeerDisconnected { peer_id: String, reason: String },
    /// A block was received from a remote peer.
    BlockReceived {
        from: String,
        cid: String,
        size_bytes: usize,
    },
    /// A remote peer requested a block from us.
    BlockRequested { from: String, cid: String },
    /// A DHT provider record was found for a given CID.
    DhtProviderFound { cid: String, provider: String },
    /// A gossip message was received on a topic.
    GossipMessage {
        topic: String,
        from: String,
        payload_len: usize,
    },
    /// A dial attempt to a peer failed.
    DialFailed {
        peer_id: String,
        address: String,
        error: String,
    },
}

// ---------------------------------------------------------------------------
// EventFilter
// ---------------------------------------------------------------------------

/// Determines which events a subscription will receive.
#[derive(Clone, Debug)]
pub enum EventFilter {
    /// Receive every event.
    All,
    /// Receive only peer-related events: `PeerConnected`, `PeerDisconnected`, `DialFailed`.
    PeerEvents,
    /// Receive only block-related events: `BlockReceived`, `BlockRequested`.
    BlockEvents,
    /// Receive only DHT events: `DhtProviderFound`.
    DhtEvents,
    /// Receive only gossip events: `GossipMessage`.
    GossipEvents,
}

// ---------------------------------------------------------------------------
// Subscription
// ---------------------------------------------------------------------------

/// Maximum number of events that can be buffered per subscription before
/// events are dropped.
const MAX_QUEUE_CAPACITY: usize = 200;

/// A single subscription to the `NetworkEventBus`.
#[derive(Debug)]
pub struct Subscription {
    /// Unique subscription identifier.
    pub id: u64,
    /// Filter controlling which events are delivered.
    pub filter: EventFilter,
    /// Buffered events waiting to be drained by the subscriber.
    pub queue: VecDeque<NetworkEvent>,
    /// Number of events dropped because the queue was full.
    pub dropped: u64,
}

impl Subscription {
    fn new(id: u64, filter: EventFilter) -> Self {
        Self {
            id,
            filter,
            queue: VecDeque::new(),
            dropped: 0,
        }
    }

    /// Returns `true` if this subscription should receive `event`.
    pub fn matches(&self, event: &NetworkEvent) -> bool {
        match &self.filter {
            EventFilter::All => true,
            EventFilter::PeerEvents => matches!(
                event,
                NetworkEvent::PeerConnected { .. }
                    | NetworkEvent::PeerDisconnected { .. }
                    | NetworkEvent::DialFailed { .. }
            ),
            EventFilter::BlockEvents => matches!(
                event,
                NetworkEvent::BlockReceived { .. } | NetworkEvent::BlockRequested { .. }
            ),
            EventFilter::DhtEvents => matches!(event, NetworkEvent::DhtProviderFound { .. }),
            EventFilter::GossipEvents => matches!(event, NetworkEvent::GossipMessage { .. }),
        }
    }
}

// ---------------------------------------------------------------------------
// BusStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for the `NetworkEventBus`.
#[derive(Clone, Debug, Default)]
pub struct BusStats {
    /// Total number of `publish` calls made.
    pub total_published: u64,
    /// Total number of event copies successfully enqueued across all subscriptions.
    pub total_delivered: u64,
    /// Total number of event copies dropped because a subscription queue was full.
    pub total_dropped: u64,
    /// Current number of active subscriptions.
    pub subscriber_count: usize,
}

// ---------------------------------------------------------------------------
// NetworkEventBus
// ---------------------------------------------------------------------------

/// Synchronous in-process publish-subscribe bus for `NetworkEvent`s.
///
/// # Example
///
/// ```rust
/// use ipfrs_network::event_bus::{NetworkEventBus, NetworkEvent, EventFilter};
///
/// let mut bus = NetworkEventBus::new();
/// let id = bus.subscribe(EventFilter::PeerEvents);
///
/// bus.publish(NetworkEvent::PeerConnected {
///     peer_id: "peer1".into(),
///     address: "/ip4/1.2.3.4/tcp/4001".into(),
/// });
///
/// let events = bus.drain(id);
/// assert_eq!(events.len(), 1);
/// ```
#[derive(Debug, Default)]
pub struct NetworkEventBus {
    subscriptions: HashMap<u64, Subscription>,
    next_id: u64,
    stats: BusStats,
}

impl NetworkEventBus {
    /// Create a new, empty `NetworkEventBus`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Subscribe to events matching `filter`.
    ///
    /// Returns a unique subscription id that can be used to drain events or
    /// unsubscribe later.
    pub fn subscribe(&mut self, filter: EventFilter) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.subscriptions.insert(id, Subscription::new(id, filter));
        id
    }

    /// Remove the subscription identified by `id`.
    ///
    /// Returns `true` if the subscription existed, `false` otherwise.
    pub fn unsubscribe(&mut self, id: u64) -> bool {
        self.subscriptions.remove(&id).is_some()
    }

    /// Publish an event to all matching subscribers.
    ///
    /// Each matching subscriber receives its own clone of `event`.  If a
    /// subscriber's queue is full (`>= 200` entries) the event is dropped and
    /// both the per-subscription and global drop counters are incremented.
    pub fn publish(&mut self, event: NetworkEvent) {
        self.stats.total_published += 1;

        for sub in self.subscriptions.values_mut() {
            if sub.matches(&event) {
                if sub.queue.len() < MAX_QUEUE_CAPACITY {
                    sub.queue.push_back(event.clone());
                    self.stats.total_delivered += 1;
                } else {
                    sub.dropped += 1;
                    self.stats.total_dropped += 1;
                }
            }
        }
    }

    /// Drain all buffered events for the subscription identified by `id`.
    ///
    /// Returns an empty `Vec` if the id does not exist.
    pub fn drain(&mut self, id: u64) -> Vec<NetworkEvent> {
        match self.subscriptions.get_mut(&id) {
            Some(sub) => sub.queue.drain(..).collect(),
            None => Vec::new(),
        }
    }

    /// Return the number of events currently buffered for subscription `id`.
    ///
    /// Returns `0` for unknown ids.
    pub fn peek_count(&self, id: u64) -> usize {
        self.subscriptions
            .get(&id)
            .map(|s| s.queue.len())
            .unwrap_or(0)
    }

    /// Return a snapshot of the current bus statistics.
    pub fn stats(&self) -> BusStats {
        BusStats {
            subscriber_count: self.subscriptions.len(),
            ..self.stats.clone()
        }
    }

    /// Drain all queues for all active subscriptions, discarding pending events.
    pub fn clear_all_queues(&mut self) {
        for sub in self.subscriptions.values_mut() {
            sub.queue.clear();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn peer_connected() -> NetworkEvent {
        NetworkEvent::PeerConnected {
            peer_id: "peer1".into(),
            address: "/ip4/127.0.0.1/tcp/4001".into(),
        }
    }

    fn peer_disconnected() -> NetworkEvent {
        NetworkEvent::PeerDisconnected {
            peer_id: "peer1".into(),
            reason: "timeout".into(),
        }
    }

    fn block_received() -> NetworkEvent {
        NetworkEvent::BlockReceived {
            from: "peer1".into(),
            cid: "bafy123".into(),
            size_bytes: 1024,
        }
    }

    fn block_requested() -> NetworkEvent {
        NetworkEvent::BlockRequested {
            from: "peer2".into(),
            cid: "bafy456".into(),
        }
    }

    fn dht_provider_found() -> NetworkEvent {
        NetworkEvent::DhtProviderFound {
            cid: "bafy789".into(),
            provider: "peer3".into(),
        }
    }

    fn gossip_message() -> NetworkEvent {
        NetworkEvent::GossipMessage {
            topic: "blocks".into(),
            from: "peer4".into(),
            payload_len: 256,
        }
    }

    fn dial_failed() -> NetworkEvent {
        NetworkEvent::DialFailed {
            peer_id: "peer5".into(),
            address: "/ip4/1.2.3.4/tcp/4001".into(),
            error: "connection refused".into(),
        }
    }

    // 1. new() empty state
    #[test]
    fn test_new_empty_state() {
        let bus = NetworkEventBus::new();
        let stats = bus.stats();
        assert_eq!(stats.total_published, 0);
        assert_eq!(stats.total_delivered, 0);
        assert_eq!(stats.total_dropped, 0);
        assert_eq!(stats.subscriber_count, 0);
    }

    // 2. subscribe() returns unique IDs
    #[test]
    fn test_subscribe_unique_ids() {
        let mut bus = NetworkEventBus::new();
        let id1 = bus.subscribe(EventFilter::All);
        let id2 = bus.subscribe(EventFilter::All);
        let id3 = bus.subscribe(EventFilter::PeerEvents);
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    // 3. unsubscribe() returns true/false
    #[test]
    fn test_unsubscribe_true_false() {
        let mut bus = NetworkEventBus::new();
        let id = bus.subscribe(EventFilter::All);
        assert!(bus.unsubscribe(id));
        assert!(!bus.unsubscribe(id)); // already removed
        assert!(!bus.unsubscribe(9999)); // never existed
    }

    // 4. publish() delivers to All filter
    #[test]
    fn test_publish_delivers_to_all_filter() {
        let mut bus = NetworkEventBus::new();
        let id = bus.subscribe(EventFilter::All);
        bus.publish(peer_connected());
        bus.publish(block_received());
        bus.publish(dht_provider_found());
        assert_eq!(bus.peek_count(id), 3);
    }

    // 5. publish() PeerEvents filter matches PeerConnected
    #[test]
    fn test_peer_events_filter_matches_peer_connected() {
        let mut bus = NetworkEventBus::new();
        let id = bus.subscribe(EventFilter::PeerEvents);
        bus.publish(peer_connected());
        assert_eq!(bus.peek_count(id), 1);
    }

    // 6. publish() PeerEvents filter ignores BlockReceived
    #[test]
    fn test_peer_events_filter_ignores_block_received() {
        let mut bus = NetworkEventBus::new();
        let id = bus.subscribe(EventFilter::PeerEvents);
        bus.publish(block_received());
        assert_eq!(bus.peek_count(id), 0);
    }

    // 7. publish() BlockEvents filter matches BlockReceived and BlockRequested
    #[test]
    fn test_block_events_filter_matches_block_events() {
        let mut bus = NetworkEventBus::new();
        let id = bus.subscribe(EventFilter::BlockEvents);
        bus.publish(block_received());
        bus.publish(block_requested());
        bus.publish(peer_connected()); // should be ignored
        assert_eq!(bus.peek_count(id), 2);
    }

    // 8. publish() DhtEvents filter matches DhtProviderFound only
    #[test]
    fn test_dht_events_filter_matches_dht_provider_found_only() {
        let mut bus = NetworkEventBus::new();
        let id = bus.subscribe(EventFilter::DhtEvents);
        bus.publish(dht_provider_found());
        bus.publish(block_received());
        bus.publish(peer_connected());
        assert_eq!(bus.peek_count(id), 1);
    }

    // 9. publish() GossipEvents filter matches GossipMessage only
    #[test]
    fn test_gossip_events_filter_matches_gossip_message_only() {
        let mut bus = NetworkEventBus::new();
        let id = bus.subscribe(EventFilter::GossipEvents);
        bus.publish(gossip_message());
        bus.publish(block_received());
        bus.publish(dht_provider_found());
        assert_eq!(bus.peek_count(id), 1);
    }

    // 10. drain() returns buffered events
    #[test]
    fn test_drain_returns_buffered_events() {
        let mut bus = NetworkEventBus::new();
        let id = bus.subscribe(EventFilter::All);
        bus.publish(peer_connected());
        bus.publish(block_received());
        let events = bus.drain(id);
        assert_eq!(events.len(), 2);
        // Queue is now empty
        assert_eq!(bus.peek_count(id), 0);
    }

    // 11. drain() unknown id returns empty vec
    #[test]
    fn test_drain_unknown_id_returns_empty() {
        let mut bus = NetworkEventBus::new();
        let events = bus.drain(9999);
        assert!(events.is_empty());
    }

    // 12. peek_count() accurate
    #[test]
    fn test_peek_count_accurate() {
        let mut bus = NetworkEventBus::new();
        let id = bus.subscribe(EventFilter::All);
        assert_eq!(bus.peek_count(id), 0);
        bus.publish(peer_connected());
        assert_eq!(bus.peek_count(id), 1);
        bus.publish(block_received());
        assert_eq!(bus.peek_count(id), 2);
        bus.drain(id);
        assert_eq!(bus.peek_count(id), 0);
    }

    // 13. queue bounded at 200, overflow increments dropped
    #[test]
    fn test_queue_bounded_at_200_overflow_drops() {
        let mut bus = NetworkEventBus::new();
        let id = bus.subscribe(EventFilter::All);

        for _ in 0..205 {
            bus.publish(peer_connected());
        }

        assert_eq!(bus.peek_count(id), 200);

        let sub = bus.subscriptions.get(&id).expect("subscription must exist");
        assert_eq!(sub.dropped, 5);
        assert_eq!(bus.stats().total_dropped, 5);
    }

    // 14. stats totals updated correctly
    #[test]
    fn test_stats_totals_updated_correctly() {
        let mut bus = NetworkEventBus::new();
        let id1 = bus.subscribe(EventFilter::All);
        let id2 = bus.subscribe(EventFilter::PeerEvents);

        bus.publish(peer_connected()); // delivered to both → 2 delivered
        bus.publish(block_received()); // delivered to id1 only → 1 delivered

        let stats = bus.stats();
        assert_eq!(stats.total_published, 2);
        assert_eq!(stats.total_delivered, 3);
        assert_eq!(stats.total_dropped, 0);
        assert_eq!(stats.subscriber_count, 2);

        let _ = (id1, id2); // suppress unused warnings
    }

    // 15. clear_all_queues empties all queues
    #[test]
    fn test_clear_all_queues_empties_all() {
        let mut bus = NetworkEventBus::new();
        let id1 = bus.subscribe(EventFilter::All);
        let id2 = bus.subscribe(EventFilter::BlockEvents);
        bus.publish(peer_connected());
        bus.publish(block_received());
        assert!(bus.peek_count(id1) > 0);
        assert!(bus.peek_count(id2) > 0);

        bus.clear_all_queues();

        assert_eq!(bus.peek_count(id1), 0);
        assert_eq!(bus.peek_count(id2), 0);
    }

    // 16. multiple subscribers receive same event (each gets own copy)
    #[test]
    fn test_multiple_subscribers_each_get_copy() {
        let mut bus = NetworkEventBus::new();
        let ids: Vec<u64> = (0..5).map(|_| bus.subscribe(EventFilter::All)).collect();

        bus.publish(peer_connected());

        for id in &ids {
            assert_eq!(
                bus.peek_count(*id),
                1,
                "subscriber {id} should have 1 event"
            );
        }
    }

    // 17. unsubscribed id no longer receives events
    #[test]
    fn test_unsubscribed_id_no_longer_receives_events() {
        let mut bus = NetworkEventBus::new();
        let id = bus.subscribe(EventFilter::All);
        bus.publish(peer_connected());
        assert_eq!(bus.peek_count(id), 1);

        assert!(bus.unsubscribe(id));
        bus.publish(block_received()); // should not be delivered

        // After unsubscribe peek_count returns 0 (id is gone)
        assert_eq!(bus.peek_count(id), 0);
    }

    // 18. PeerEvents filter also matches PeerDisconnected and DialFailed
    #[test]
    fn test_peer_events_filter_matches_all_peer_variants() {
        let mut bus = NetworkEventBus::new();
        let id = bus.subscribe(EventFilter::PeerEvents);
        bus.publish(peer_connected());
        bus.publish(peer_disconnected());
        bus.publish(dial_failed());
        bus.publish(gossip_message()); // should not match
        assert_eq!(bus.peek_count(id), 3);
    }

    // 19. subscriber_count reflects current subscriptions
    #[test]
    fn test_subscriber_count_reflects_current() {
        let mut bus = NetworkEventBus::new();
        assert_eq!(bus.stats().subscriber_count, 0);
        let id1 = bus.subscribe(EventFilter::All);
        assert_eq!(bus.stats().subscriber_count, 1);
        let id2 = bus.subscribe(EventFilter::BlockEvents);
        assert_eq!(bus.stats().subscriber_count, 2);
        bus.unsubscribe(id1);
        assert_eq!(bus.stats().subscriber_count, 1);
        bus.unsubscribe(id2);
        assert_eq!(bus.stats().subscriber_count, 0);
    }
}
