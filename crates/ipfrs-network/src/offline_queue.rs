//! Offline request queue for mobile and intermittent connectivity
//!
//! This module provides queuing for network operations when offline:
//! - Queue requests when network is unavailable
//! - Automatic replay when connection is restored
//! - Request prioritization
//! - Timeout management
//! - Persistent storage support

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Errors that can occur during offline queue operations
#[derive(Error, Debug, Clone)]
pub enum OfflineQueueError {
    #[error("Queue is full")]
    QueueFull,

    #[error("Request expired")]
    RequestExpired,

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// Configuration for offline queue
#[derive(Debug, Clone)]
pub struct OfflineQueueConfig {
    /// Maximum number of queued requests
    pub max_queue_size: usize,

    /// Enable persistent storage of queue
    pub enable_persistence: bool,

    /// Default request timeout
    pub default_timeout: Duration,

    /// Enable request prioritization
    pub enable_prioritization: bool,

    /// Maximum age for a request before automatic removal
    pub max_request_age: Duration,

    /// Enable automatic replay when online
    pub enable_auto_replay: bool,

    /// Batch size for replay
    pub replay_batch_size: usize,

    /// Delay between replay batches
    pub replay_batch_delay: Duration,
}

impl Default for OfflineQueueConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 1000,
            enable_persistence: false,
            default_timeout: Duration::from_secs(300), // 5 minutes
            enable_prioritization: true,
            max_request_age: Duration::from_secs(3600), // 1 hour
            enable_auto_replay: true,
            replay_batch_size: 10,
            replay_batch_delay: Duration::from_millis(100),
        }
    }
}

impl OfflineQueueConfig {
    /// Configuration for mobile devices
    pub fn mobile() -> Self {
        Self {
            max_queue_size: 500,
            enable_persistence: true,
            default_timeout: Duration::from_secs(600),
            enable_prioritization: true,
            max_request_age: Duration::from_secs(1800), // 30 minutes
            enable_auto_replay: true,
            replay_batch_size: 5,
            replay_batch_delay: Duration::from_millis(200),
        }
    }

    /// Configuration for IoT devices
    pub fn iot() -> Self {
        Self {
            max_queue_size: 100,
            enable_persistence: true,
            default_timeout: Duration::from_secs(900), // 15 minutes
            enable_prioritization: true,
            max_request_age: Duration::from_secs(3600),
            enable_auto_replay: true,
            replay_batch_size: 3,
            replay_batch_delay: Duration::from_millis(500),
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), OfflineQueueError> {
        if self.max_queue_size == 0 {
            return Err(OfflineQueueError::InvalidConfig(
                "max_queue_size must be > 0".to_string(),
            ));
        }

        if self.replay_batch_size == 0 {
            return Err(OfflineQueueError::InvalidConfig(
                "replay_batch_size must be > 0".to_string(),
            ));
        }

        Ok(())
    }
}

/// Priority level for queued requests
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RequestPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Type of queued request
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueuedRequestType {
    /// Provide content to DHT
    ProvideContent(String),
    /// Find providers for content
    FindProviders(String),
    /// Get value from DHT
    GetValue(String),
    /// Put value to DHT
    PutValue { key: String, value: Vec<u8> },
    /// Custom request
    Custom { operation: String, data: Vec<u8> },
}

/// A queued request waiting for network connectivity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueuedRequest {
    /// Unique request ID
    pub id: String,
    /// Request type
    pub request_type: QueuedRequestType,
    /// Priority level
    pub priority: RequestPriority,
    /// When the request was queued
    #[serde(skip)]
    pub queued_at: Option<Instant>,
    /// Serialized timestamp for persistence
    pub queued_timestamp: u64,
    /// Request timeout
    pub timeout: Duration,
    /// Number of retry attempts
    pub retry_count: u32,
    /// Maximum retry attempts
    pub max_retries: u32,
}

impl QueuedRequest {
    /// Create a new queued request
    pub fn new(
        id: String,
        request_type: QueuedRequestType,
        priority: RequestPriority,
        timeout: Duration,
    ) -> Self {
        let now = Instant::now();
        Self {
            id,
            request_type,
            priority,
            queued_at: Some(now),
            queued_timestamp: now.elapsed().as_secs(),
            timeout,
            retry_count: 0,
            max_retries: 3,
        }
    }

    /// Check if request has expired
    pub fn is_expired(&self, max_age: Duration) -> bool {
        if let Some(queued_at) = self.queued_at {
            Instant::now().duration_since(queued_at) > max_age
        } else {
            false
        }
    }

    /// Check if request has timed out
    pub fn is_timed_out(&self) -> bool {
        if let Some(queued_at) = self.queued_at {
            Instant::now().duration_since(queued_at) > self.timeout
        } else {
            false
        }
    }

    /// Check if should retry
    pub fn should_retry(&self) -> bool {
        self.retry_count < self.max_retries
    }
}

/// Offline queue state
struct QueueState {
    /// Pending requests
    requests: VecDeque<QueuedRequest>,
    /// Requests currently being replayed
    in_flight: Vec<String>,
    /// Network status
    is_online: bool,
    /// Last cleanup time
    last_cleanup: Instant,
}

impl QueueState {
    fn new() -> Self {
        Self {
            requests: VecDeque::new(),
            in_flight: Vec::new(),
            is_online: false,
            last_cleanup: Instant::now(),
        }
    }
}

/// Offline request queue
pub struct OfflineQueue {
    config: OfflineQueueConfig,
    state: Arc<RwLock<QueueState>>,
    stats: Arc<RwLock<OfflineQueueStats>>,
}

impl OfflineQueue {
    /// Create a new offline queue
    pub fn new(config: OfflineQueueConfig) -> Result<Self, OfflineQueueError> {
        config.validate()?;

        Ok(Self {
            config,
            state: Arc::new(RwLock::new(QueueState::new())),
            stats: Arc::new(RwLock::new(OfflineQueueStats::default())),
        })
    }

    /// Add a request to the queue
    pub fn enqueue(&self, request: QueuedRequest) -> Result<(), OfflineQueueError> {
        let mut state = self.state.write();
        let mut stats = self.stats.write();

        // Check queue size
        if state.requests.len() >= self.config.max_queue_size {
            stats.requests_dropped += 1;
            return Err(OfflineQueueError::QueueFull);
        }

        // Insert with priority
        if self.config.enable_prioritization {
            // Find insertion point based on priority
            let insert_pos = state
                .requests
                .iter()
                .position(|r| r.priority < request.priority)
                .unwrap_or(state.requests.len());

            state.requests.insert(insert_pos, request);
        } else {
            state.requests.push_back(request);
        }

        stats.requests_queued += 1;

        Ok(())
    }

    /// Get the next request to replay
    pub fn dequeue(&self) -> Option<QueuedRequest> {
        let mut state = self.state.write();

        if !state.is_online {
            return None;
        }

        while let Some(request) = state.requests.pop_front() {
            // Check if expired
            if request.is_expired(self.config.max_request_age) {
                let mut stats = self.stats.write();
                stats.requests_expired += 1;
                continue;
            }

            // Check if timed out
            if request.is_timed_out() {
                let mut stats = self.stats.write();
                stats.requests_timed_out += 1;
                continue;
            }

            state.in_flight.push(request.id.clone());
            return Some(request);
        }

        None
    }

    /// Mark a request as completed
    pub fn mark_completed(&self, request_id: &str, success: bool) {
        let mut state = self.state.write();
        let mut stats = self.stats.write();

        state.in_flight.retain(|id| id != request_id);

        if success {
            stats.requests_completed += 1;
        } else {
            stats.requests_failed += 1;
        }
    }

    /// Requeue a failed request for retry
    pub fn requeue(&self, mut request: QueuedRequest) -> Result<(), OfflineQueueError> {
        let mut state = self.state.write();

        state.in_flight.retain(|id| id != &request.id);

        if !request.should_retry() {
            let mut stats = self.stats.write();
            stats.requests_failed += 1;
            return Ok(());
        }

        request.retry_count += 1;

        if self.config.enable_prioritization {
            let insert_pos = state
                .requests
                .iter()
                .position(|r| r.priority < request.priority)
                .unwrap_or(state.requests.len());

            state.requests.insert(insert_pos, request);
        } else {
            state.requests.push_back(request);
        }

        let mut stats = self.stats.write();
        stats.requests_retried += 1;

        Ok(())
    }

    /// Set network online status
    pub fn set_online(&self, online: bool) {
        let mut state = self.state.write();
        state.is_online = online;

        if online {
            let mut stats = self.stats.write();
            stats.online_transitions += 1;
        } else {
            let mut stats = self.stats.write();
            stats.offline_transitions += 1;
        }
    }

    /// Check if network is online
    pub fn is_online(&self) -> bool {
        self.state.read().is_online
    }

    /// Get number of pending requests
    pub fn pending_count(&self) -> usize {
        self.state.read().requests.len()
    }

    /// Get number of in-flight requests
    pub fn in_flight_count(&self) -> usize {
        self.state.read().in_flight.len()
    }

    /// Clean up expired requests
    pub fn cleanup_expired(&self) {
        let mut state = self.state.write();
        let mut stats = self.stats.write();

        let initial_len = state.requests.len();

        state
            .requests
            .retain(|r| !r.is_expired(self.config.max_request_age));

        let removed = initial_len - state.requests.len();
        stats.requests_expired += removed as u64;

        state.last_cleanup = Instant::now();
    }

    /// Get a batch of requests for replay
    pub fn get_replay_batch(&self) -> Vec<QueuedRequest> {
        let mut batch = Vec::with_capacity(self.config.replay_batch_size);

        for _ in 0..self.config.replay_batch_size {
            if let Some(request) = self.dequeue() {
                batch.push(request);
            } else {
                break;
            }
        }

        batch
    }

    /// Get current statistics
    pub fn stats(&self) -> OfflineQueueStats {
        self.stats.read().clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        *self.stats.write() = OfflineQueueStats::default();
    }

    /// Clear all pending requests
    pub fn clear(&self) {
        let mut state = self.state.write();
        state.requests.clear();
        state.in_flight.clear();
    }
}

/// Statistics for offline queue
#[derive(Debug, Clone, Default)]
pub struct OfflineQueueStats {
    /// Total requests queued
    pub requests_queued: u64,
    /// Requests dropped (queue full)
    pub requests_dropped: u64,
    /// Requests completed successfully
    pub requests_completed: u64,
    /// Requests that failed
    pub requests_failed: u64,
    /// Requests that expired
    pub requests_expired: u64,
    /// Requests that timed out
    pub requests_timed_out: u64,
    /// Requests retried
    pub requests_retried: u64,
    /// Online transitions
    pub online_transitions: u64,
    /// Offline transitions
    pub offline_transitions: u64,
}

impl OfflineQueueStats {
    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.requests_completed + self.requests_failed;
        if total == 0 {
            return 0.0;
        }
        self.requests_completed as f64 / total as f64
    }

    /// Calculate completion rate (including expired/dropped)
    pub fn completion_rate(&self) -> f64 {
        if self.requests_queued == 0 {
            return 0.0;
        }
        self.requests_completed as f64 / self.requests_queued as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = OfflineQueueConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.max_queue_size, 1000);
    }

    #[test]
    fn test_config_mobile() {
        let config = OfflineQueueConfig::mobile();
        assert!(config.validate().is_ok());
        assert_eq!(config.max_queue_size, 500);
    }

    #[test]
    fn test_config_iot() {
        let config = OfflineQueueConfig::iot();
        assert!(config.validate().is_ok());
        assert_eq!(config.max_queue_size, 100);
    }

    #[test]
    fn test_enqueue() {
        let config = OfflineQueueConfig::default();
        let queue = OfflineQueue::new(config).expect("test: default config should create queue");

        let request = QueuedRequest::new(
            "test1".to_string(),
            QueuedRequestType::FindProviders("QmTest".to_string()),
            RequestPriority::Normal,
            Duration::from_secs(60),
        );

        assert!(queue.enqueue(request).is_ok());
        assert_eq!(queue.pending_count(), 1);
    }

    #[test]
    fn test_priority_ordering() {
        let config = OfflineQueueConfig::default();
        let queue = OfflineQueue::new(config).expect("test: default config should create queue");

        // Add low priority
        let req1 = QueuedRequest::new(
            "low".to_string(),
            QueuedRequestType::FindProviders("QmTest1".to_string()),
            RequestPriority::Low,
            Duration::from_secs(60),
        );
        queue
            .enqueue(req1)
            .expect("test: enqueue low priority request should succeed");

        // Add high priority
        let req2 = QueuedRequest::new(
            "high".to_string(),
            QueuedRequestType::FindProviders("QmTest2".to_string()),
            RequestPriority::High,
            Duration::from_secs(60),
        );
        queue
            .enqueue(req2)
            .expect("test: enqueue high priority request should succeed");

        // Set online to enable dequeue
        queue.set_online(true);

        // High priority should come out first
        let next = queue
            .dequeue()
            .expect("test: dequeue should return high priority request");
        assert_eq!(next.id, "high");
    }

    #[test]
    fn test_dequeue_when_offline() {
        let config = OfflineQueueConfig::default();
        let queue = OfflineQueue::new(config).expect("test: default config should create queue");

        let request = QueuedRequest::new(
            "test1".to_string(),
            QueuedRequestType::FindProviders("QmTest".to_string()),
            RequestPriority::Normal,
            Duration::from_secs(60),
        );

        queue
            .enqueue(request)
            .expect("test: enqueue request should succeed");

        // Should return None when offline
        assert!(queue.dequeue().is_none());
    }

    #[test]
    fn test_dequeue_when_online() {
        let config = OfflineQueueConfig::default();
        let queue = OfflineQueue::new(config).expect("test: default config should create queue");

        let request = QueuedRequest::new(
            "test1".to_string(),
            QueuedRequestType::FindProviders("QmTest".to_string()),
            RequestPriority::Normal,
            Duration::from_secs(60),
        );

        queue
            .enqueue(request)
            .expect("test: enqueue request should succeed");
        queue.set_online(true);

        // Should return request when online
        let req = queue.dequeue();
        assert!(req.is_some());
        assert_eq!(
            req.expect("test: dequeue should return queued request").id,
            "test1"
        );
    }

    #[test]
    fn test_mark_completed() {
        let config = OfflineQueueConfig::default();
        let queue = OfflineQueue::new(config).expect("test: default config should create queue");

        let request = QueuedRequest::new(
            "test1".to_string(),
            QueuedRequestType::FindProviders("QmTest".to_string()),
            RequestPriority::Normal,
            Duration::from_secs(60),
        );

        queue
            .enqueue(request)
            .expect("test: enqueue request should succeed");
        queue.set_online(true);
        let req = queue
            .dequeue()
            .expect("test: dequeue should return enqueued request");

        queue.mark_completed(&req.id, true);

        let stats = queue.stats();
        assert_eq!(stats.requests_completed, 1);
    }

    #[test]
    fn test_requeue() {
        let config = OfflineQueueConfig::default();
        let queue = OfflineQueue::new(config).expect("test: default config should create queue");

        let mut request = QueuedRequest::new(
            "test1".to_string(),
            QueuedRequestType::FindProviders("QmTest".to_string()),
            RequestPriority::Normal,
            Duration::from_secs(60),
        );
        request.max_retries = 3;

        queue
            .enqueue(request.clone())
            .expect("test: enqueue request should succeed");
        queue.set_online(true);
        let req = queue
            .dequeue()
            .expect("test: dequeue should return enqueued request");

        queue
            .requeue(req)
            .expect("test: requeue should succeed for request with retries remaining");

        let stats = queue.stats();
        assert_eq!(stats.requests_retried, 1);
    }

    #[test]
    fn test_queue_full() {
        let config = OfflineQueueConfig {
            max_queue_size: 2,
            ..Default::default()
        };
        let queue = OfflineQueue::new(config)
            .expect("test: config with max_queue_size=2 should create queue");

        let req1 = QueuedRequest::new(
            "test1".to_string(),
            QueuedRequestType::FindProviders("QmTest1".to_string()),
            RequestPriority::Normal,
            Duration::from_secs(60),
        );
        let req2 = QueuedRequest::new(
            "test2".to_string(),
            QueuedRequestType::FindProviders("QmTest2".to_string()),
            RequestPriority::Normal,
            Duration::from_secs(60),
        );
        let req3 = QueuedRequest::new(
            "test3".to_string(),
            QueuedRequestType::FindProviders("QmTest3".to_string()),
            RequestPriority::Normal,
            Duration::from_secs(60),
        );

        assert!(queue.enqueue(req1).is_ok());
        assert!(queue.enqueue(req2).is_ok());
        assert!(matches!(
            queue.enqueue(req3),
            Err(OfflineQueueError::QueueFull)
        ));
    }

    #[test]
    fn test_get_replay_batch() {
        let config = OfflineQueueConfig {
            replay_batch_size: 3,
            ..Default::default()
        };
        let queue = OfflineQueue::new(config)
            .expect("test: config with replay_batch_size=3 should create queue");

        for i in 0..5 {
            let req = QueuedRequest::new(
                format!("test{}", i),
                QueuedRequestType::FindProviders(format!("QmTest{}", i)),
                RequestPriority::Normal,
                Duration::from_secs(60),
            );
            queue
                .enqueue(req)
                .expect("test: enqueue batch request should succeed");
        }

        queue.set_online(true);

        let batch = queue.get_replay_batch();
        assert_eq!(batch.len(), 3);
    }

    #[test]
    fn test_stats_success_rate() {
        let stats = OfflineQueueStats {
            requests_completed: 8,
            requests_failed: 2,
            ..Default::default()
        };

        assert_eq!(stats.success_rate(), 0.8);
    }

    #[test]
    fn test_clear() {
        let config = OfflineQueueConfig::default();
        let queue = OfflineQueue::new(config).expect("test: default config should create queue");

        let req = QueuedRequest::new(
            "test1".to_string(),
            QueuedRequestType::FindProviders("QmTest".to_string()),
            RequestPriority::Normal,
            Duration::from_secs(60),
        );
        queue
            .enqueue(req)
            .expect("test: enqueue request should succeed");

        assert_eq!(queue.pending_count(), 1);

        queue.clear();
        assert_eq!(queue.pending_count(), 0);
    }
}
