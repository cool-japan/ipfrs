//! NetworkEventBus — synchronous publish-subscribe event bus for network events.
//!
//! Provides topic-based routing, subscriber filtering, and event replay for late
//! subscribers. All operations are synchronous and suitable for use from both
//! synchronous and asynchronous contexts without spawning tasks.
//!
//! # Design
//!
//! - Publishers call [`NetworkEventBus::publish`] to emit events.
//! - Consumers call [`NetworkEventBus::subscribe`] to register interest in events.
//! - Subscribers may specify an [`EventFilter`] to narrow the event stream.
//! - A configurable replay buffer retains recent events so that late subscribers
//!   can catch up via [`NetworkEventBus::replay_for_subscriber`].
//! - No `unwrap()` is used anywhere in this module.

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// EventTopic
// ---------------------------------------------------------------------------

/// Categorises the kind of network event being published.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum EventTopic {
    /// A remote peer established a connection.
    PeerConnected,
    /// A remote peer disconnected or was dropped.
    PeerDisconnected,
    /// A content block was received from a peer.
    BlockReceived,
    /// A content block was requested by a peer.
    BlockRequested,
    /// A DHT lookup was initiated or completed.
    DhtLookup,
    /// A gossip-sub message arrived on some topic.
    GossipMessage,
    /// An application-defined event with a free-form tag.
    Custom(String),
}

impl EventTopic {
    /// Returns a string representation of the topic suitable for logging.
    pub fn as_str(&self) -> &str {
        match self {
            Self::PeerConnected => "PeerConnected",
            Self::PeerDisconnected => "PeerDisconnected",
            Self::BlockReceived => "BlockReceived",
            Self::BlockRequested => "BlockRequested",
            Self::DhtLookup => "DhtLookup",
            Self::GossipMessage => "GossipMessage",
            Self::Custom(tag) => tag.as_str(),
        }
    }
}

impl std::fmt::Display for EventTopic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// NebNetworkEvent  (aliased at crate-root as NebNetworkEvent to avoid clash
//                   with node::NetworkEvent)
// ---------------------------------------------------------------------------

/// A single network event, as published to the bus.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebNetworkEvent {
    /// Monotonically increasing identifier assigned at publish time.
    pub id: u64,
    /// The topic this event belongs to.
    pub topic: EventTopic,
    /// Opaque payload bytes supplied by the publisher.
    pub payload: Vec<u8>,
    /// Optional peer-id string that identifies the originating peer.
    pub source_peer: Option<String>,
    /// Wall-clock timestamp (e.g. seconds or milliseconds since UNIX epoch)
    /// supplied by the caller; the bus does not call the OS clock itself.
    pub timestamp: u64,
}

// ---------------------------------------------------------------------------
// SubscriberId
// ---------------------------------------------------------------------------

/// An opaque, monotonically increasing identifier for a subscription.
///
/// The sentinel value `SubscriberId(0)` is returned when subscription fails
/// (e.g. because the bus is already at capacity).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubscriberId(pub u64);

impl std::fmt::Display for SubscriberId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sub({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// EventFilter
// ---------------------------------------------------------------------------

/// Determines which events are delivered to a particular subscriber.
#[derive(Clone, Debug)]
pub enum EventFilter {
    /// Accept every event regardless of topic, source, or payload size.
    All,
    /// Accept only events whose topic is in the provided list.
    TopicIn(Vec<EventTopic>),
    /// Accept only events that originated from the specified peer.
    FromPeer(String),
    /// Accept only events whose payload is larger than `n` bytes.
    PayloadSizeAbove(usize),
}

// ---------------------------------------------------------------------------
// NebSubscription  (aliased at crate-root as NebSubscription)
// ---------------------------------------------------------------------------

/// Runtime state for a single subscriber.
#[derive(Clone, Debug)]
pub struct NebSubscription {
    /// The unique identifier for this subscription.
    pub id: SubscriberId,
    /// The filter applied to inbound events.
    pub filter: EventFilter,
    /// Total number of events that passed the filter and were counted as
    /// delivered to this subscriber.
    pub events_received: u64,
    /// Timestamp of the most recently delivered event (as supplied by the
    /// publisher).  Zero if no event has been delivered yet.
    pub last_event_at: u64,
}

// ---------------------------------------------------------------------------
// EventBusConfig
// ---------------------------------------------------------------------------

/// Configuration knobs for [`NetworkEventBus`].
#[derive(Clone, Debug)]
pub struct EventBusConfig {
    /// Maximum number of simultaneous subscribers.  Attempts to subscribe
    /// beyond this limit return `SubscriberId(0)`.
    pub max_subscribers: usize,
    /// Number of events retained in the replay buffer.  Older events are
    /// evicted (FIFO) when this limit is reached.
    pub replay_buffer_size: usize,
    /// Maximum number of bytes allowed in a single event payload.  Events
    /// that exceed this limit are rejected with [`BusError::PayloadTooLarge`].
    pub max_payload_bytes: usize,
}

impl Default for EventBusConfig {
    fn default() -> Self {
        Self {
            max_subscribers: 1_000,
            replay_buffer_size: 10_000,
            max_payload_bytes: 1_048_576, // 1 MiB
        }
    }
}

// ---------------------------------------------------------------------------
// BusError
// ---------------------------------------------------------------------------

/// Errors returned by [`NetworkEventBus`] operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BusError {
    /// No subscription with the given id exists.
    SubscriberNotFound(u64),
    /// The payload exceeds the configured maximum.
    PayloadTooLarge {
        /// Actual size of the rejected payload.
        size: usize,
        /// Configured maximum.
        max: usize,
    },
    /// The bus already has `max_subscribers` active subscriptions.
    MaxSubscribersReached,
}

impl std::fmt::Display for BusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SubscriberNotFound(id) => {
                write!(f, "subscriber {id} not found")
            }
            Self::PayloadTooLarge { size, max } => {
                write!(f, "payload too large: {size} bytes (max {max})")
            }
            Self::MaxSubscribersReached => {
                write!(f, "maximum number of subscribers reached")
            }
        }
    }
}

impl std::error::Error for BusError {}

// ---------------------------------------------------------------------------
// EventBusStats
// ---------------------------------------------------------------------------

/// A snapshot of bus-wide statistics.
#[derive(Clone, Debug)]
pub struct EventBusStats {
    /// Current number of active subscribers.
    pub subscriber_count: usize,
    /// Current number of events held in the replay buffer.
    pub replay_buffer_size: usize,
    /// Total number of events successfully published (not rejected) so far.
    pub total_published: u64,
    /// Total number of `(subscriber, event)` delivery pairs counted so far.
    pub total_delivered: u64,
    /// Average number of deliveries per published event (0.0 if no events
    /// have been published yet).
    pub avg_delivery_per_event: f64,
}

// ---------------------------------------------------------------------------
// NetworkEventBus
// ---------------------------------------------------------------------------

/// Synchronous, in-process publish-subscribe bus for network events.
///
/// ## Thread safety
///
/// `NetworkEventBus` is intentionally **not** `Send` or `Sync`.  If you need
/// to share it across threads, wrap it in `Arc<Mutex<NetworkEventBus>>`.
///
/// ## Example
///
/// ```rust
/// use ipfrs_network::{
///     NebNetworkEvent, NebSubscription,
///     EventBusConfig, EventFilter, EventTopic, NetworkEventBus,
/// };
///
/// let mut bus = NetworkEventBus::new(EventBusConfig::default());
///
/// // Subscribe to peer-connection events only.
/// let sub_id = bus.subscribe(EventFilter::TopicIn(vec![EventTopic::PeerConnected]));
///
/// // Publish an event.
/// let _event_id = bus
///     .publish(
///         EventTopic::PeerConnected,
///         b"hello".to_vec(),
///         Some("peer-abc".to_string()),
///         1_000,
///     )
///     .expect("publish failed");
///
/// // Retrieve matching events for the subscriber.
/// let events = bus.drain_events_for(sub_id).expect("drain failed");
/// assert_eq!(events.len(), 1);
/// ```
pub struct NetworkEventBus {
    /// Configuration that governs limits and buffer sizes.
    pub config: EventBusConfig,
    /// Live subscriptions, keyed by their numeric id.
    pub subscriptions: HashMap<u64, NebSubscription>,
    /// Ring buffer of recent events available for replay.
    pub replay_buffer: VecDeque<NebNetworkEvent>,
    /// Counter used to assign unique ids to new events.
    pub next_event_id: u64,
    /// Counter used to assign unique ids to new subscriptions.
    pub next_sub_id: u64,
    /// Total number of events that have been successfully published.
    pub total_published: u64,
    /// Total number of `(subscriber, event)` delivery pairs.
    pub total_delivered: u64,
}

impl NetworkEventBus {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new bus with the supplied configuration.
    pub fn new(config: EventBusConfig) -> Self {
        // Start next_sub_id at 1 so that 0 is permanently reserved as the
        // "failure sentinel" returned by [`subscribe`] when the bus is full.
        Self {
            replay_buffer: VecDeque::with_capacity(config.replay_buffer_size.min(4096)),
            config,
            subscriptions: HashMap::new(),
            next_event_id: 1,
            next_sub_id: 1,
            total_published: 0,
            total_delivered: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Filter helpers
    // -----------------------------------------------------------------------

    /// Returns `true` if `event` satisfies `filter`.
    pub fn matches_filter(filter: &EventFilter, event: &NebNetworkEvent) -> bool {
        match filter {
            EventFilter::All => true,
            EventFilter::TopicIn(topics) => topics.contains(&event.topic),
            EventFilter::FromPeer(peer) => event.source_peer.as_deref() == Some(peer.as_str()),
            EventFilter::PayloadSizeAbove(n) => event.payload.len() > *n,
        }
    }

    // -----------------------------------------------------------------------
    // Subscription management
    // -----------------------------------------------------------------------

    /// Register a new subscriber with the given filter.
    ///
    /// Returns the assigned [`SubscriberId`].  If the bus is already at
    /// capacity, `SubscriberId(0)` is returned as a sentinel (no subscription
    /// is created).
    pub fn subscribe(&mut self, filter: EventFilter) -> SubscriberId {
        if self.subscriptions.len() >= self.config.max_subscribers {
            return SubscriberId(0);
        }
        let id = self.next_sub_id;
        self.next_sub_id = self.next_sub_id.saturating_add(1);
        let sub = NebSubscription {
            id: SubscriberId(id),
            filter,
            events_received: 0,
            last_event_at: 0,
        };
        self.subscriptions.insert(id, sub);
        SubscriberId(id)
    }

    /// Remove the subscription identified by `id`.
    ///
    /// Returns `true` if the subscription existed and was removed, `false` if
    /// no subscription with that id was found.
    pub fn unsubscribe(&mut self, id: SubscriberId) -> bool {
        self.subscriptions.remove(&id.0).is_some()
    }

    /// Return an immutable reference to the [`NebSubscription`] for `id`, or
    /// `None` if no such subscription exists.
    pub fn subscriber_stats(&self, id: SubscriberId) -> Option<&NebSubscription> {
        self.subscriptions.get(&id.0)
    }

    // -----------------------------------------------------------------------
    // Publishing
    // -----------------------------------------------------------------------

    /// Publish an event to the bus.
    ///
    /// # Errors
    ///
    /// - [`BusError::PayloadTooLarge`] – `payload.len() > config.max_payload_bytes`.
    ///
    /// # Returns
    ///
    /// The assigned event id (monotonically increasing).
    pub fn publish(
        &mut self,
        topic: EventTopic,
        payload: Vec<u8>,
        source_peer: Option<String>,
        now: u64,
    ) -> Result<u64, BusError> {
        // Validate payload size before doing anything.
        if payload.len() > self.config.max_payload_bytes {
            return Err(BusError::PayloadTooLarge {
                size: payload.len(),
                max: self.config.max_payload_bytes,
            });
        }

        // Assign event id and build the event.
        let event_id = self.next_event_id;
        self.next_event_id = self.next_event_id.saturating_add(1);

        let event = NebNetworkEvent {
            id: event_id,
            topic,
            payload,
            source_peer,
            timestamp: now,
        };

        // Deliver to matching subscribers and update their stats.
        let mut delivered_count: u64 = 0;
        for sub in self.subscriptions.values_mut() {
            if Self::matches_filter(&sub.filter, &event) {
                sub.events_received = sub.events_received.saturating_add(1);
                sub.last_event_at = now;
                delivered_count = delivered_count.saturating_add(1);
            }
        }
        self.total_delivered = self.total_delivered.saturating_add(delivered_count);

        // Push to replay buffer, evicting the oldest entry if needed.
        // When replay_buffer_size == 0, no events are retained.
        if self.config.replay_buffer_size > 0 {
            if self.replay_buffer.len() >= self.config.replay_buffer_size {
                self.replay_buffer.pop_front();
            }
            self.replay_buffer.push_back(event);
        }

        self.total_published = self.total_published.saturating_add(1);
        Ok(event_id)
    }

    // -----------------------------------------------------------------------
    // Replay / draining
    // -----------------------------------------------------------------------

    /// Return all buffered events with `id > since_event_id` that match the
    /// subscriber's filter.
    ///
    /// # Errors
    ///
    /// - [`BusError::SubscriberNotFound`] if `id` does not correspond to an
    ///   active subscription.
    pub fn replay_for_subscriber(
        &self,
        id: SubscriberId,
        since_event_id: u64,
    ) -> Result<Vec<&NebNetworkEvent>, BusError> {
        let sub = self
            .subscriptions
            .get(&id.0)
            .ok_or(BusError::SubscriberNotFound(id.0))?;

        let events = self
            .replay_buffer
            .iter()
            .filter(|e| e.id > since_event_id && Self::matches_filter(&sub.filter, e))
            .collect();

        Ok(events)
    }

    /// Clone all buffered events that match the subscriber's filter and return
    /// them.  The replay buffer is **not** modified.
    ///
    /// # Errors
    ///
    /// - [`BusError::SubscriberNotFound`] if `id` does not correspond to an
    ///   active subscription.
    pub fn drain_events_for(&mut self, id: SubscriberId) -> Result<Vec<NebNetworkEvent>, BusError> {
        // Borrow subscriptions immutably first to retrieve the filter, then
        // iterate over replay_buffer.  We clone matching events.
        let filter = {
            let sub = self
                .subscriptions
                .get(&id.0)
                .ok_or(BusError::SubscriberNotFound(id.0))?;
            sub.filter.clone()
        };

        let events: Vec<NebNetworkEvent> = self
            .replay_buffer
            .iter()
            .filter(|e| Self::matches_filter(&filter, e))
            .cloned()
            .collect();

        Ok(events)
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Return a snapshot of bus-wide statistics.
    pub fn event_bus_stats(&self) -> EventBusStats {
        let avg_delivery_per_event = if self.total_published == 0 {
            0.0_f64
        } else {
            self.total_delivered as f64 / self.total_published as f64
        };

        EventBusStats {
            subscriber_count: self.subscriptions.len(),
            replay_buffer_size: self.replay_buffer.len(),
            total_published: self.total_published,
            total_delivered: self.total_delivered,
            avg_delivery_per_event,
        }
    }
}

impl Default for NetworkEventBus {
    fn default() -> Self {
        Self::new(EventBusConfig::default())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::{
        BusError, EventBusConfig, EventFilter, EventTopic, NebSubscription, NetworkEventBus,
        SubscriberId,
    };

    // -----------------------------------------------------------------------
    // Helper
    // -----------------------------------------------------------------------

    fn make_bus() -> NetworkEventBus {
        NetworkEventBus::new(EventBusConfig::default())
    }

    fn make_small_bus(max_subs: usize, buf: usize, max_bytes: usize) -> NetworkEventBus {
        NetworkEventBus::new(EventBusConfig {
            max_subscribers: max_subs,
            replay_buffer_size: buf,
            max_payload_bytes: max_bytes,
        })
    }

    fn publish_peer_connected(bus: &mut NetworkEventBus, peer: &str, now: u64) -> u64 {
        bus.publish(
            EventTopic::PeerConnected,
            peer.as_bytes().to_vec(),
            Some(peer.to_string()),
            now,
        )
        .expect("publish failed")
    }

    // -----------------------------------------------------------------------
    // EventTopic
    // -----------------------------------------------------------------------

    #[test]
    fn test_event_topic_as_str_named_variants() {
        assert_eq!(EventTopic::PeerConnected.as_str(), "PeerConnected");
        assert_eq!(EventTopic::PeerDisconnected.as_str(), "PeerDisconnected");
        assert_eq!(EventTopic::BlockReceived.as_str(), "BlockReceived");
        assert_eq!(EventTopic::BlockRequested.as_str(), "BlockRequested");
        assert_eq!(EventTopic::DhtLookup.as_str(), "DhtLookup");
        assert_eq!(EventTopic::GossipMessage.as_str(), "GossipMessage");
    }

    #[test]
    fn test_event_topic_custom_as_str() {
        let tag = "my-custom-event";
        let topic = EventTopic::Custom(tag.to_string());
        assert_eq!(topic.as_str(), tag);
    }

    #[test]
    fn test_event_topic_display() {
        let s = format!("{}", EventTopic::BlockReceived);
        assert_eq!(s, "BlockReceived");
    }

    #[test]
    fn test_event_topic_equality() {
        assert_eq!(EventTopic::PeerConnected, EventTopic::PeerConnected);
        assert_ne!(EventTopic::PeerConnected, EventTopic::PeerDisconnected);
        assert_eq!(
            EventTopic::Custom("x".to_string()),
            EventTopic::Custom("x".to_string())
        );
        assert_ne!(
            EventTopic::Custom("x".to_string()),
            EventTopic::Custom("y".to_string())
        );
    }

    // -----------------------------------------------------------------------
    // SubscriberId
    // -----------------------------------------------------------------------

    #[test]
    fn test_subscriber_id_display() {
        let s = format!("{}", SubscriberId(42));
        assert_eq!(s, "Sub(42)");
    }

    #[test]
    fn test_subscriber_id_equality() {
        assert_eq!(SubscriberId(1), SubscriberId(1));
        assert_ne!(SubscriberId(1), SubscriberId(2));
    }

    // -----------------------------------------------------------------------
    // subscribe / unsubscribe
    // -----------------------------------------------------------------------

    #[test]
    fn test_subscribe_returns_non_zero_id() {
        let mut bus = make_bus();
        let id = bus.subscribe(EventFilter::All);
        assert_ne!(
            id,
            SubscriberId(0),
            "sentinel value should not be returned on success"
        );
    }

    #[test]
    fn test_subscribe_ids_are_monotonically_increasing() {
        let mut bus = make_bus();
        let id1 = bus.subscribe(EventFilter::All);
        let id2 = bus.subscribe(EventFilter::All);
        let id3 = bus.subscribe(EventFilter::All);
        assert!(id1.0 < id2.0);
        assert!(id2.0 < id3.0);
    }

    #[test]
    fn test_subscribe_at_capacity_returns_sentinel() {
        let mut bus = make_small_bus(2, 100, 1024);
        let id1 = bus.subscribe(EventFilter::All);
        let id2 = bus.subscribe(EventFilter::All);
        // Third subscribe should fail.
        let id3 = bus.subscribe(EventFilter::All);
        assert_ne!(id1, SubscriberId(0));
        assert_ne!(id2, SubscriberId(0));
        assert_eq!(
            id3,
            SubscriberId(0),
            "should return sentinel when at capacity"
        );
    }

    #[test]
    fn test_subscribe_after_unsubscribe_frees_slot() {
        let mut bus = make_small_bus(2, 100, 1024);
        let id1 = bus.subscribe(EventFilter::All);
        let _id2 = bus.subscribe(EventFilter::All);
        // At capacity — unsubscribe one.
        assert!(bus.unsubscribe(id1));
        // Now there is room for one more.
        let id3 = bus.subscribe(EventFilter::All);
        assert_ne!(id3, SubscriberId(0));
    }

    #[test]
    fn test_unsubscribe_unknown_id_returns_false() {
        let mut bus = make_bus();
        assert!(!bus.unsubscribe(SubscriberId(999)));
    }

    #[test]
    fn test_unsubscribe_twice_returns_false_second_time() {
        let mut bus = make_bus();
        let id = bus.subscribe(EventFilter::All);
        assert!(bus.unsubscribe(id));
        assert!(!bus.unsubscribe(id));
    }

    // -----------------------------------------------------------------------
    // matches_filter
    // -----------------------------------------------------------------------

    fn make_event(
        id: u64,
        topic: EventTopic,
        payload: Vec<u8>,
        source_peer: Option<&str>,
    ) -> super::NebNetworkEvent {
        super::NebNetworkEvent {
            id,
            topic,
            payload,
            source_peer: source_peer.map(|s| s.to_string()),
            timestamp: 0,
        }
    }

    #[test]
    fn test_filter_all_matches_any_event() {
        let event = make_event(1, EventTopic::BlockReceived, vec![1, 2, 3], None);
        assert!(NetworkEventBus::matches_filter(&EventFilter::All, &event));
    }

    #[test]
    fn test_filter_topic_in_matches_included_topic() {
        let event = make_event(1, EventTopic::PeerConnected, vec![], None);
        let filter = EventFilter::TopicIn(vec![EventTopic::PeerConnected, EventTopic::DhtLookup]);
        assert!(NetworkEventBus::matches_filter(&filter, &event));
    }

    #[test]
    fn test_filter_topic_in_rejects_excluded_topic() {
        let event = make_event(1, EventTopic::GossipMessage, vec![], None);
        let filter = EventFilter::TopicIn(vec![EventTopic::PeerConnected]);
        assert!(!NetworkEventBus::matches_filter(&filter, &event));
    }

    #[test]
    fn test_filter_from_peer_matches_correct_peer() {
        let event = make_event(1, EventTopic::BlockReceived, vec![], Some("alice"));
        let filter = EventFilter::FromPeer("alice".to_string());
        assert!(NetworkEventBus::matches_filter(&filter, &event));
    }

    #[test]
    fn test_filter_from_peer_rejects_wrong_peer() {
        let event = make_event(1, EventTopic::BlockReceived, vec![], Some("bob"));
        let filter = EventFilter::FromPeer("alice".to_string());
        assert!(!NetworkEventBus::matches_filter(&filter, &event));
    }

    #[test]
    fn test_filter_from_peer_rejects_no_peer() {
        let event = make_event(1, EventTopic::BlockReceived, vec![], None);
        let filter = EventFilter::FromPeer("alice".to_string());
        assert!(!NetworkEventBus::matches_filter(&filter, &event));
    }

    #[test]
    fn test_filter_payload_size_above_matches_larger_payload() {
        let event = make_event(1, EventTopic::BlockReceived, vec![0u8; 100], None);
        let filter = EventFilter::PayloadSizeAbove(50);
        assert!(NetworkEventBus::matches_filter(&filter, &event));
    }

    #[test]
    fn test_filter_payload_size_above_rejects_equal_size() {
        // "above" means strictly greater than, not >=.
        let event = make_event(1, EventTopic::BlockReceived, vec![0u8; 50], None);
        let filter = EventFilter::PayloadSizeAbove(50);
        assert!(!NetworkEventBus::matches_filter(&filter, &event));
    }

    #[test]
    fn test_filter_payload_size_above_rejects_smaller_payload() {
        let event = make_event(1, EventTopic::BlockReceived, vec![0u8; 10], None);
        let filter = EventFilter::PayloadSizeAbove(50);
        assert!(!NetworkEventBus::matches_filter(&filter, &event));
    }

    // -----------------------------------------------------------------------
    // publish
    // -----------------------------------------------------------------------

    #[test]
    fn test_publish_returns_monotonically_increasing_ids() {
        let mut bus = make_bus();
        let id1 = bus
            .publish(EventTopic::PeerConnected, vec![], None, 1)
            .expect("publish failed");
        let id2 = bus
            .publish(EventTopic::PeerDisconnected, vec![], None, 2)
            .expect("publish failed");
        assert!(id1 < id2);
    }

    #[test]
    fn test_publish_rejects_oversized_payload() {
        let mut bus = make_small_bus(100, 100, 10);
        let result = bus.publish(EventTopic::BlockReceived, vec![0u8; 11], None, 0);
        assert_eq!(result, Err(BusError::PayloadTooLarge { size: 11, max: 10 }));
    }

    #[test]
    fn test_publish_accepts_max_size_payload() {
        let mut bus = make_small_bus(100, 100, 10);
        let result = bus.publish(EventTopic::BlockReceived, vec![0u8; 10], None, 0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_publish_increments_total_published() {
        let mut bus = make_bus();
        bus.publish(EventTopic::PeerConnected, vec![], None, 0)
            .expect("publish failed");
        bus.publish(EventTopic::PeerConnected, vec![], None, 1)
            .expect("publish failed");
        assert_eq!(bus.total_published, 2);
    }

    #[test]
    fn test_publish_increments_delivered_for_matching_subscribers() {
        let mut bus = make_bus();
        let _sub = bus.subscribe(EventFilter::All);
        bus.publish(EventTopic::PeerConnected, vec![], None, 0)
            .expect("publish failed");
        bus.publish(EventTopic::PeerConnected, vec![], None, 1)
            .expect("publish failed");
        assert_eq!(bus.total_delivered, 2);
    }

    #[test]
    fn test_publish_does_not_count_non_matching_subscribers() {
        let mut bus = make_bus();
        // Subscriber only wants DhtLookup events.
        let _sub = bus.subscribe(EventFilter::TopicIn(vec![EventTopic::DhtLookup]));
        // Publish a PeerConnected event — should not be counted.
        bus.publish(EventTopic::PeerConnected, vec![], None, 0)
            .expect("publish failed");
        assert_eq!(bus.total_delivered, 0);
    }

    #[test]
    fn test_publish_updates_subscriber_events_received() {
        let mut bus = make_bus();
        let id = bus.subscribe(EventFilter::All);
        publish_peer_connected(&mut bus, "alice", 100);
        publish_peer_connected(&mut bus, "bob", 200);
        let stats = bus.subscriber_stats(id).expect("stats missing");
        assert_eq!(stats.events_received, 2);
    }

    #[test]
    fn test_publish_updates_subscriber_last_event_at() {
        let mut bus = make_bus();
        let id = bus.subscribe(EventFilter::All);
        publish_peer_connected(&mut bus, "alice", 100);
        publish_peer_connected(&mut bus, "bob", 200);
        let stats = bus.subscriber_stats(id).expect("stats missing");
        assert_eq!(stats.last_event_at, 200);
    }

    #[test]
    fn test_publish_fills_replay_buffer() {
        let mut bus = make_small_bus(100, 5, 1024);
        for i in 0..5u64 {
            bus.publish(EventTopic::PeerConnected, vec![], None, i)
                .expect("publish failed");
        }
        assert_eq!(bus.replay_buffer.len(), 5);
    }

    #[test]
    fn test_publish_evicts_oldest_when_buffer_full() {
        let mut bus = make_small_bus(100, 3, 1024);
        let id1 = bus
            .publish(EventTopic::PeerConnected, vec![], None, 1)
            .expect("publish failed");
        bus.publish(EventTopic::PeerConnected, vec![], None, 2)
            .expect("publish failed");
        bus.publish(EventTopic::PeerConnected, vec![], None, 3)
            .expect("publish failed");
        // Buffer is full.  This should evict event with id == id1.
        bus.publish(EventTopic::PeerConnected, vec![], None, 4)
            .expect("publish failed");

        assert_eq!(bus.replay_buffer.len(), 3);
        // The oldest event should no longer be in the buffer.
        let oldest_in_buffer = bus.replay_buffer.front().expect("buffer empty").id;
        assert_ne!(oldest_in_buffer, id1);
    }

    // -----------------------------------------------------------------------
    // replay_for_subscriber
    // -----------------------------------------------------------------------

    #[test]
    fn test_replay_for_subscriber_unknown_id_returns_error() {
        let bus = make_bus();
        let result = bus.replay_for_subscriber(SubscriberId(999), 0);
        assert_eq!(result, Err(BusError::SubscriberNotFound(999)));
    }

    #[test]
    fn test_replay_for_subscriber_returns_events_after_given_id() {
        let mut bus = make_bus();
        let sub = bus.subscribe(EventFilter::All);
        // Publish three events; their ids start at 1.
        let id1 = publish_peer_connected(&mut bus, "a", 1);
        let _id2 = publish_peer_connected(&mut bus, "b", 2);
        let _id3 = publish_peer_connected(&mut bus, "c", 3);
        // Replay events with id > id1.
        let replayed = bus.replay_for_subscriber(sub, id1).expect("replay failed");
        assert_eq!(replayed.len(), 2);
    }

    #[test]
    fn test_replay_for_subscriber_returns_empty_when_up_to_date() {
        let mut bus = make_bus();
        let sub = bus.subscribe(EventFilter::All);
        let id1 = publish_peer_connected(&mut bus, "a", 1);
        let replayed = bus.replay_for_subscriber(sub, id1).expect("replay failed");
        assert!(replayed.is_empty());
    }

    #[test]
    fn test_replay_for_subscriber_respects_filter() {
        let mut bus = make_bus();
        let sub = bus.subscribe(EventFilter::TopicIn(vec![EventTopic::BlockReceived]));
        // Publish mixed topics.
        bus.publish(EventTopic::PeerConnected, vec![], None, 1)
            .expect("publish failed");
        bus.publish(EventTopic::BlockReceived, vec![], None, 2)
            .expect("publish failed");
        bus.publish(EventTopic::PeerConnected, vec![], None, 3)
            .expect("publish failed");
        // Replay from id 0 — only BlockReceived should be included.
        let replayed = bus.replay_for_subscriber(sub, 0).expect("replay failed");
        assert_eq!(replayed.len(), 1);
        assert_eq!(replayed[0].topic, EventTopic::BlockReceived);
    }

    // -----------------------------------------------------------------------
    // drain_events_for
    // -----------------------------------------------------------------------

    #[test]
    fn test_drain_events_for_unknown_id_returns_error() {
        let mut bus = make_bus();
        let result = bus.drain_events_for(SubscriberId(999));
        assert_eq!(result, Err(BusError::SubscriberNotFound(999)));
    }

    #[test]
    fn test_drain_events_for_returns_matching_events() {
        let mut bus = make_bus();
        let sub = bus.subscribe(EventFilter::All);
        publish_peer_connected(&mut bus, "x", 1);
        publish_peer_connected(&mut bus, "y", 2);
        let drained = bus.drain_events_for(sub).expect("drain failed");
        assert_eq!(drained.len(), 2);
    }

    #[test]
    fn test_drain_events_for_does_not_remove_from_buffer() {
        let mut bus = make_bus();
        let sub = bus.subscribe(EventFilter::All);
        publish_peer_connected(&mut bus, "x", 1);
        let _first = bus.drain_events_for(sub).expect("drain failed");
        // The buffer should still contain the event.
        let second = bus.drain_events_for(sub).expect("drain failed");
        assert_eq!(second.len(), 1);
    }

    #[test]
    fn test_drain_events_for_respects_filter() {
        let mut bus = make_bus();
        let sub = bus.subscribe(EventFilter::FromPeer("alice".to_string()));
        bus.publish(
            EventTopic::PeerConnected,
            vec![],
            Some("alice".to_string()),
            1,
        )
        .expect("publish failed");
        bus.publish(
            EventTopic::PeerConnected,
            vec![],
            Some("bob".to_string()),
            2,
        )
        .expect("publish failed");
        let drained = bus.drain_events_for(sub).expect("drain failed");
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].source_peer.as_deref(), Some("alice"));
    }

    // -----------------------------------------------------------------------
    // subscriber_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_subscriber_stats_none_for_unknown_id() {
        let bus = make_bus();
        assert!(bus.subscriber_stats(SubscriberId(42)).is_none());
    }

    #[test]
    fn test_subscriber_stats_some_for_known_id() {
        let mut bus = make_bus();
        let id = bus.subscribe(EventFilter::All);
        assert!(bus.subscriber_stats(id).is_some());
    }

    #[test]
    fn test_subscriber_stats_initial_values() {
        let mut bus = make_bus();
        let id = bus.subscribe(EventFilter::All);
        let stats: &NebSubscription = bus.subscriber_stats(id).expect("stats missing");
        assert_eq!(stats.events_received, 0);
        assert_eq!(stats.last_event_at, 0);
        assert_eq!(stats.id, id);
    }

    // -----------------------------------------------------------------------
    // event_bus_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_event_bus_stats_initial_state() {
        let bus = make_bus();
        let stats = bus.event_bus_stats();
        assert_eq!(stats.subscriber_count, 0);
        assert_eq!(stats.replay_buffer_size, 0);
        assert_eq!(stats.total_published, 0);
        assert_eq!(stats.total_delivered, 0);
        assert_eq!(stats.avg_delivery_per_event, 0.0);
    }

    #[test]
    fn test_event_bus_stats_counts_subscribers() {
        let mut bus = make_bus();
        bus.subscribe(EventFilter::All);
        bus.subscribe(EventFilter::All);
        let stats = bus.event_bus_stats();
        assert_eq!(stats.subscriber_count, 2);
    }

    #[test]
    fn test_event_bus_stats_counts_buffer_size() {
        let mut bus = make_bus();
        publish_peer_connected(&mut bus, "a", 1);
        publish_peer_connected(&mut bus, "b", 2);
        let stats = bus.event_bus_stats();
        assert_eq!(stats.replay_buffer_size, 2);
    }

    #[test]
    fn test_event_bus_stats_avg_delivery_per_event() {
        let mut bus = make_bus();
        // Two subscribers, one matching only PeerConnected.
        bus.subscribe(EventFilter::All);
        bus.subscribe(EventFilter::TopicIn(vec![EventTopic::PeerConnected]));
        // Publish one PeerConnected → both subscribers receive it (2 deliveries).
        bus.publish(EventTopic::PeerConnected, vec![], None, 1)
            .expect("publish failed");
        let stats = bus.event_bus_stats();
        assert_eq!(stats.total_published, 1);
        assert_eq!(stats.total_delivered, 2);
        assert!((stats.avg_delivery_per_event - 2.0_f64).abs() < f64::EPSILON);
    }

    #[test]
    fn test_event_bus_stats_avg_delivery_zero_when_no_events() {
        let bus = make_bus();
        let stats = bus.event_bus_stats();
        assert_eq!(stats.avg_delivery_per_event, 0.0);
    }

    // -----------------------------------------------------------------------
    // BusError display / equality
    // -----------------------------------------------------------------------

    #[test]
    fn test_bus_error_subscriber_not_found_display() {
        let e = BusError::SubscriberNotFound(7);
        assert!(e.to_string().contains("7"));
    }

    #[test]
    fn test_bus_error_payload_too_large_display() {
        let e = BusError::PayloadTooLarge {
            size: 2000,
            max: 1000,
        };
        let s = e.to_string();
        assert!(s.contains("2000"));
        assert!(s.contains("1000"));
    }

    #[test]
    fn test_bus_error_max_subscribers_reached_display() {
        let e = BusError::MaxSubscribersReached;
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn test_bus_error_equality() {
        assert_eq!(
            BusError::SubscriberNotFound(1),
            BusError::SubscriberNotFound(1)
        );
        assert_ne!(
            BusError::SubscriberNotFound(1),
            BusError::SubscriberNotFound(2)
        );
        assert_eq!(
            BusError::PayloadTooLarge { size: 5, max: 4 },
            BusError::PayloadTooLarge { size: 5, max: 4 }
        );
        assert_eq!(
            BusError::MaxSubscribersReached,
            BusError::MaxSubscribersReached
        );
    }

    // -----------------------------------------------------------------------
    // Default impl
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_bus_uses_default_config() {
        let bus = NetworkEventBus::default();
        assert_eq!(bus.config.max_subscribers, 1_000);
        assert_eq!(bus.config.replay_buffer_size, 10_000);
        assert_eq!(bus.config.max_payload_bytes, 1_048_576);
    }

    // -----------------------------------------------------------------------
    // Edge-cases / integration
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_subscribers_with_different_filters() {
        let mut bus = make_bus();
        let sub_all = bus.subscribe(EventFilter::All);
        let sub_dht = bus.subscribe(EventFilter::TopicIn(vec![EventTopic::DhtLookup]));

        bus.publish(EventTopic::PeerConnected, vec![], None, 1)
            .expect("publish failed");
        bus.publish(EventTopic::DhtLookup, vec![], None, 2)
            .expect("publish failed");

        let all_stats = bus.subscriber_stats(sub_all).expect("stats");
        let dht_stats = bus.subscriber_stats(sub_dht).expect("stats");
        assert_eq!(all_stats.events_received, 2);
        assert_eq!(dht_stats.events_received, 1);
    }

    #[test]
    fn test_custom_topic_filtering() {
        let mut bus = make_bus();
        let sub = bus.subscribe(EventFilter::TopicIn(vec![EventTopic::Custom(
            "app::hello".to_string(),
        )]));
        bus.publish(
            EventTopic::Custom("app::hello".to_string()),
            vec![],
            None,
            1,
        )
        .expect("publish failed");
        bus.publish(
            EventTopic::Custom("app::world".to_string()),
            vec![],
            None,
            2,
        )
        .expect("publish failed");
        let stats = bus.subscriber_stats(sub).expect("stats");
        assert_eq!(stats.events_received, 1);
    }

    #[test]
    fn test_publish_rejected_payload_not_counted() {
        let mut bus = make_small_bus(100, 100, 5);
        let _ = bus.publish(EventTopic::BlockReceived, vec![0u8; 10], None, 0);
        assert_eq!(bus.total_published, 0);
    }

    #[test]
    fn test_subscriber_receives_zero_events_when_filter_never_matches() {
        let mut bus = make_bus();
        let sub = bus.subscribe(EventFilter::FromPeer("ghost".to_string()));
        for i in 0..10u64 {
            bus.publish(
                EventTopic::PeerConnected,
                vec![],
                Some("real-peer".to_string()),
                i,
            )
            .expect("publish failed");
        }
        let stats = bus.subscriber_stats(sub).expect("stats");
        assert_eq!(stats.events_received, 0);
    }

    #[test]
    fn test_replay_buffer_size_with_zero_capacity() {
        // A bus with replay_buffer_size == 0 never retains events.
        let mut bus = make_small_bus(100, 0, 1024);
        let sub = bus.subscribe(EventFilter::All);
        bus.publish(EventTopic::PeerConnected, vec![], None, 1)
            .expect("publish failed");
        let replayed = bus.replay_for_subscriber(sub, 0).expect("replay failed");
        assert!(replayed.is_empty(), "buffer size 0 should retain no events");
    }
}
