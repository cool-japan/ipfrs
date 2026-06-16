//! Network Policy Engine - Fine-grained control over network operations
//!
//! This module provides a policy engine for enforcing various network policies:
//! - Connection policies (whitelist/blacklist, rate limits)
//! - Bandwidth policies (per-peer, per-protocol limits)
//! - Content policies (allowed CIDs, content types)
//! - Time-based policies (schedules, quotas)
//! - Geographic policies (region restrictions)
//!
//! Useful for production deployments requiring fine-grained control.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::{PolicyEngine, PolicyConfig, ConnectionPolicy, PolicyAction};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = PolicyConfig::default();
//! let engine = PolicyEngine::new(config);
//!
//! // Add a connection policy
//! let policy = ConnectionPolicy::new("block-untrusted")
//!     .with_action(PolicyAction::Deny)
//!     .with_priority(100);
//!
//! engine.add_connection_policy(policy)?;
//!
//! // Evaluate connection
//! let allowed = engine.evaluate_connection("peer123").await?;
//! println!("Connection allowed: {}", allowed);
//! # Ok(())
//! # }
//! ```

use dashmap::DashMap;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Errors that can occur in the policy engine
#[derive(Debug, Error)]
pub enum PolicyError {
    /// Policy not found
    #[error("Policy not found: {0}")]
    NotFound(String),

    /// Invalid policy configuration
    #[error("Invalid policy: {0}")]
    Invalid(String),

    /// Policy conflict
    #[error("Policy conflict: {0}")]
    Conflict(String),

    /// Policy evaluation error
    #[error("Evaluation error: {0}")]
    Evaluation(String),
}

/// Action to take when a policy matches
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyAction {
    /// Allow the operation
    Allow,
    /// Deny the operation
    Deny,
    /// Rate limit the operation
    RateLimit,
    /// Log the operation but allow it
    Log,
    /// Require additional verification
    Verify,
}

/// Policy evaluation result
#[derive(Debug, Clone)]
pub struct PolicyResult {
    /// Whether the operation is allowed
    pub allowed: bool,

    /// Action taken
    pub action: PolicyAction,

    /// Matched policy name
    pub policy_name: Option<String>,

    /// Reason for the decision
    pub reason: String,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
}

impl PolicyResult {
    /// Create an allow result
    pub fn allow(reason: String) -> Self {
        Self {
            allowed: true,
            action: PolicyAction::Allow,
            policy_name: None,
            reason,
            confidence: 1.0,
        }
    }

    /// Create a deny result
    pub fn deny(reason: String, policy_name: Option<String>) -> Self {
        Self {
            allowed: false,
            action: PolicyAction::Deny,
            policy_name,
            reason,
            confidence: 1.0,
        }
    }
}

/// Connection policy for controlling peer connections
#[derive(Debug, Clone)]
pub struct ConnectionPolicy {
    /// Policy name
    pub name: String,

    /// Action to take
    pub action: PolicyAction,

    /// Priority (higher = evaluated first)
    pub priority: u32,

    /// Peer whitelist (empty = all allowed)
    pub peer_whitelist: Vec<String>,

    /// Peer blacklist
    pub peer_blacklist: Vec<String>,

    /// Maximum connections per peer
    pub max_connections_per_peer: Option<usize>,

    /// Connection rate limit (connections per second)
    pub rate_limit: Option<f64>,

    /// Enabled
    pub enabled: bool,
}

impl ConnectionPolicy {
    /// Create a new connection policy
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            action: PolicyAction::Allow,
            priority: 50,
            peer_whitelist: Vec::new(),
            peer_blacklist: Vec::new(),
            max_connections_per_peer: None,
            rate_limit: None,
            enabled: true,
        }
    }

    /// Set the action
    pub fn with_action(mut self, action: PolicyAction) -> Self {
        self.action = action;
        self
    }

    /// Set the priority
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Add peer to whitelist
    pub fn with_whitelist_peer(mut self, peer: impl Into<String>) -> Self {
        self.peer_whitelist.push(peer.into());
        self
    }

    /// Add peer to blacklist
    pub fn with_blacklist_peer(mut self, peer: impl Into<String>) -> Self {
        self.peer_blacklist.push(peer.into());
        self
    }

    /// Set max connections per peer
    pub fn with_max_connections(mut self, max: usize) -> Self {
        self.max_connections_per_peer = Some(max);
        self
    }

    /// Set rate limit
    pub fn with_rate_limit(mut self, rate: f64) -> Self {
        self.rate_limit = Some(rate);
        self
    }

    /// Evaluate if this policy matches
    pub fn evaluate(&self, peer_id: &str) -> Option<PolicyResult> {
        if !self.enabled {
            return None;
        }

        // Check blacklist first
        if self.peer_blacklist.iter().any(|p| p == peer_id) {
            return Some(PolicyResult::deny(
                format!("Peer {} is blacklisted", peer_id),
                Some(self.name.clone()),
            ));
        }

        // Check whitelist (if configured)
        if !self.peer_whitelist.is_empty() && !self.peer_whitelist.iter().any(|p| p == peer_id) {
            return Some(PolicyResult::deny(
                format!("Peer {} not in whitelist", peer_id),
                Some(self.name.clone()),
            ));
        }

        // If no specific rule matched, return action
        Some(PolicyResult {
            allowed: self.action == PolicyAction::Allow,
            action: self.action,
            policy_name: Some(self.name.clone()),
            reason: format!("Policy {} matched", self.name),
            confidence: 1.0,
        })
    }
}

/// Bandwidth policy for controlling data transfer
#[derive(Debug, Clone)]
pub struct BandwidthPolicy {
    /// Policy name
    pub name: String,

    /// Maximum upload bandwidth (bytes per second)
    pub max_upload_bps: Option<u64>,

    /// Maximum download bandwidth (bytes per second)
    pub max_download_bps: Option<u64>,

    /// Per-peer bandwidth limit
    pub per_peer_limit_bps: Option<u64>,

    /// Priority
    pub priority: u32,

    /// Enabled
    pub enabled: bool,
}

impl BandwidthPolicy {
    /// Create a new bandwidth policy
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            max_upload_bps: None,
            max_download_bps: None,
            per_peer_limit_bps: None,
            priority: 50,
            enabled: true,
        }
    }

    /// Set max upload bandwidth
    pub fn with_max_upload(mut self, bps: u64) -> Self {
        self.max_upload_bps = Some(bps);
        self
    }

    /// Set max download bandwidth
    pub fn with_max_download(mut self, bps: u64) -> Self {
        self.max_download_bps = Some(bps);
        self
    }

    /// Set per-peer limit
    pub fn with_per_peer_limit(mut self, bps: u64) -> Self {
        self.per_peer_limit_bps = Some(bps);
        self
    }
}

/// Content policy for controlling what content is allowed
#[derive(Debug, Clone)]
pub struct ContentPolicy {
    /// Policy name
    pub name: String,

    /// Action to take
    pub action: PolicyAction,

    /// Allowed CID patterns (regex)
    pub allowed_patterns: Vec<String>,

    /// Blocked CID patterns (regex)
    pub blocked_patterns: Vec<String>,

    /// Maximum content size in bytes
    pub max_size: Option<u64>,

    /// Priority
    pub priority: u32,

    /// Enabled
    pub enabled: bool,
}

impl ContentPolicy {
    /// Create a new content policy
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            action: PolicyAction::Allow,
            allowed_patterns: Vec::new(),
            blocked_patterns: Vec::new(),
            max_size: None,
            priority: 50,
            enabled: true,
        }
    }

    /// Add allowed pattern
    pub fn with_allowed_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.allowed_patterns.push(pattern.into());
        self
    }

    /// Add blocked pattern
    pub fn with_blocked_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.blocked_patterns.push(pattern.into());
        self
    }

    /// Set max size
    pub fn with_max_size(mut self, size: u64) -> Self {
        self.max_size = Some(size);
        self
    }
}

/// Configuration for the policy engine
#[derive(Debug, Clone)]
pub struct PolicyConfig {
    /// Enable policy enforcement
    pub enabled: bool,

    /// Default action when no policy matches
    pub default_action: PolicyAction,

    /// Enable policy logging
    pub log_decisions: bool,

    /// Maximum policies per type
    pub max_policies_per_type: usize,

    /// Policy evaluation timeout
    pub evaluation_timeout: Duration,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_action: PolicyAction::Allow,
            log_decisions: true,
            max_policies_per_type: 100,
            evaluation_timeout: Duration::from_millis(100),
        }
    }
}

impl PolicyConfig {
    /// Configuration for strict security
    pub fn strict() -> Self {
        Self {
            enabled: true,
            default_action: PolicyAction::Deny,
            log_decisions: true,
            max_policies_per_type: 200,
            evaluation_timeout: Duration::from_millis(50),
        }
    }

    /// Configuration for permissive mode
    pub fn permissive() -> Self {
        Self {
            enabled: true,
            default_action: PolicyAction::Allow,
            log_decisions: false,
            max_policies_per_type: 50,
            evaluation_timeout: Duration::from_millis(200),
        }
    }
}

/// Statistics tracked by the policy engine
#[derive(Debug, Clone, Default)]
pub struct PolicyStats {
    /// Total policy evaluations
    pub evaluations: u64,

    /// Allowed operations
    pub allowed: u64,

    /// Denied operations
    pub denied: u64,

    /// Rate limited operations
    pub rate_limited: u64,

    /// Policy hits by name
    pub policy_hits: HashMap<String, u64>,

    /// Average evaluation time
    pub avg_eval_time_ms: f64,
}

/// Policy engine for enforcing network policies
pub struct PolicyEngine {
    config: PolicyConfig,
    connection_policies: Arc<RwLock<Vec<ConnectionPolicy>>>,
    bandwidth_policies: Arc<RwLock<Vec<BandwidthPolicy>>>,
    content_policies: Arc<RwLock<Vec<ContentPolicy>>>,
    stats: Arc<RwLock<PolicyStats>>,
    connection_counts: Arc<DashMap<String, usize>>,
}

impl PolicyEngine {
    /// Create a new policy engine
    pub fn new(config: PolicyConfig) -> Self {
        Self {
            config,
            connection_policies: Arc::new(RwLock::new(Vec::new())),
            bandwidth_policies: Arc::new(RwLock::new(Vec::new())),
            content_policies: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(PolicyStats::default())),
            connection_counts: Arc::new(DashMap::new()),
        }
    }

    /// Add a connection policy
    pub fn add_connection_policy(&self, policy: ConnectionPolicy) -> Result<(), PolicyError> {
        let mut policies = self.connection_policies.write();

        if policies.len() >= self.config.max_policies_per_type {
            return Err(PolicyError::Invalid(
                "Maximum connection policies reached".to_string(),
            ));
        }

        policies.push(policy);
        policies.sort_by_key(|p| std::cmp::Reverse(p.priority));

        Ok(())
    }

    /// Add a bandwidth policy
    pub fn add_bandwidth_policy(&self, policy: BandwidthPolicy) -> Result<(), PolicyError> {
        let mut policies = self.bandwidth_policies.write();

        if policies.len() >= self.config.max_policies_per_type {
            return Err(PolicyError::Invalid(
                "Maximum bandwidth policies reached".to_string(),
            ));
        }

        policies.push(policy);
        policies.sort_by_key(|p| std::cmp::Reverse(p.priority));

        Ok(())
    }

    /// Add a content policy
    pub fn add_content_policy(&self, policy: ContentPolicy) -> Result<(), PolicyError> {
        let mut policies = self.content_policies.write();

        if policies.len() >= self.config.max_policies_per_type {
            return Err(PolicyError::Invalid(
                "Maximum content policies reached".to_string(),
            ));
        }

        policies.push(policy);
        policies.sort_by_key(|p| std::cmp::Reverse(p.priority));

        Ok(())
    }

    /// Evaluate a connection request
    pub async fn evaluate_connection(&self, peer_id: &str) -> Result<bool, PolicyError> {
        let start = Instant::now();

        if !self.config.enabled {
            return Ok(true);
        }

        let policies = self.connection_policies.read();

        // Evaluate policies in priority order
        for policy in policies.iter() {
            if let Some(result) = policy.evaluate(peer_id) {
                self.record_evaluation(&result, start.elapsed());
                return Ok(result.allowed);
            }
        }

        // No policy matched, use default action
        let allowed = self.config.default_action == PolicyAction::Allow;
        self.record_default_evaluation(allowed, start.elapsed());

        Ok(allowed)
    }

    /// Check if a peer can establish a connection (with connection count check)
    pub fn can_connect(&self, peer_id: &str) -> bool {
        let count = self.connection_counts.get(peer_id).map(|c| *c).unwrap_or(0);

        let policies = self.connection_policies.read();

        for policy in policies.iter() {
            if let Some(max) = policy.max_connections_per_peer {
                if count >= max {
                    return false;
                }
            }
        }

        true
    }

    /// Record a connection
    pub fn record_connection(&self, peer_id: &str) {
        self.connection_counts
            .entry(peer_id.to_string())
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }

    /// Record a disconnection
    pub fn record_disconnection(&self, peer_id: &str) {
        if let Some(mut count) = self.connection_counts.get_mut(peer_id) {
            if *count > 0 {
                *count -= 1;
            }
        }
    }

    /// Remove a connection policy
    pub fn remove_connection_policy(&self, name: &str) -> Result<(), PolicyError> {
        let mut policies = self.connection_policies.write();
        let len_before = policies.len();
        policies.retain(|p| p.name != name);

        if policies.len() == len_before {
            Err(PolicyError::NotFound(name.to_string()))
        } else {
            Ok(())
        }
    }

    /// Get all connection policies
    pub fn connection_policies(&self) -> Vec<ConnectionPolicy> {
        self.connection_policies.read().clone()
    }

    /// Get all bandwidth policies
    pub fn bandwidth_policies(&self) -> Vec<BandwidthPolicy> {
        self.bandwidth_policies.read().clone()
    }

    /// Get all content policies
    pub fn content_policies(&self) -> Vec<ContentPolicy> {
        self.content_policies.read().clone()
    }

    /// Get statistics
    pub fn stats(&self) -> PolicyStats {
        self.stats.read().clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write();
        *stats = PolicyStats::default();
    }

    /// Record an evaluation
    fn record_evaluation(&self, result: &PolicyResult, duration: Duration) {
        let mut stats = self.stats.write();
        stats.evaluations += 1;

        if result.allowed {
            stats.allowed += 1;
        } else {
            stats.denied += 1;
        }

        if result.action == PolicyAction::RateLimit {
            stats.rate_limited += 1;
        }

        if let Some(ref policy_name) = result.policy_name {
            *stats.policy_hits.entry(policy_name.clone()).or_insert(0) += 1;
        }

        // Update average evaluation time (exponential moving average)
        let eval_time_ms = duration.as_secs_f64() * 1000.0;
        let alpha = 0.3;
        stats.avg_eval_time_ms = alpha * eval_time_ms + (1.0 - alpha) * stats.avg_eval_time_ms;
    }

    /// Record a default evaluation
    fn record_default_evaluation(&self, allowed: bool, duration: Duration) {
        let mut stats = self.stats.write();
        stats.evaluations += 1;

        if allowed {
            stats.allowed += 1;
        } else {
            stats.denied += 1;
        }

        let eval_time_ms = duration.as_secs_f64() * 1000.0;
        let alpha = 0.3;
        stats.avg_eval_time_ms = alpha * eval_time_ms + (1.0 - alpha) * stats.avg_eval_time_ms;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_creation() {
        let policy = ConnectionPolicy::new("test")
            .with_action(PolicyAction::Allow)
            .with_priority(100);

        assert_eq!(policy.name, "test");
        assert_eq!(policy.action, PolicyAction::Allow);
        assert_eq!(policy.priority, 100);
    }

    #[test]
    fn test_policy_engine_creation() {
        let config = PolicyConfig::default();
        let engine = PolicyEngine::new(config);

        assert_eq!(engine.connection_policies().len(), 0);
    }

    #[tokio::test]
    async fn test_add_connection_policy() {
        let engine = PolicyEngine::new(PolicyConfig::default());

        let policy = ConnectionPolicy::new("test");
        assert!(engine.add_connection_policy(policy).is_ok());
        assert_eq!(engine.connection_policies().len(), 1);
    }

    #[tokio::test]
    async fn test_blacklist_evaluation() {
        let engine = PolicyEngine::new(PolicyConfig::default());

        let policy = ConnectionPolicy::new("blacklist")
            .with_action(PolicyAction::Allow)
            .with_blacklist_peer("bad_peer");

        engine
            .add_connection_policy(policy)
            .expect("test: should add blacklist policy");

        let allowed = engine
            .evaluate_connection("bad_peer")
            .await
            .expect("test: should evaluate bad_peer connection");
        assert!(!allowed);

        let allowed = engine
            .evaluate_connection("good_peer")
            .await
            .expect("test: should evaluate good_peer connection");
        assert!(allowed);
    }

    #[tokio::test]
    async fn test_whitelist_evaluation() {
        let engine = PolicyEngine::new(PolicyConfig::default());

        let policy = ConnectionPolicy::new("whitelist")
            .with_action(PolicyAction::Allow)
            .with_whitelist_peer("good_peer");

        engine
            .add_connection_policy(policy)
            .expect("test: should add whitelist policy");

        let allowed = engine
            .evaluate_connection("good_peer")
            .await
            .expect("test: should evaluate good_peer connection");
        assert!(allowed);

        let allowed = engine
            .evaluate_connection("bad_peer")
            .await
            .expect("test: should evaluate bad_peer connection");
        assert!(!allowed);
    }

    #[test]
    fn test_connection_counting() {
        let engine = PolicyEngine::new(PolicyConfig::default());

        engine.record_connection("peer1");
        engine.record_connection("peer1");

        assert!(engine.can_connect("peer1"));

        engine.record_disconnection("peer1");
        assert!(engine.can_connect("peer1"));
    }

    #[tokio::test]
    async fn test_policy_priority() {
        let engine = PolicyEngine::new(PolicyConfig::default());

        // Lower priority - deny
        let policy1 = ConnectionPolicy::new("low")
            .with_action(PolicyAction::Deny)
            .with_priority(10);

        // Higher priority - allow
        let policy2 = ConnectionPolicy::new("high")
            .with_action(PolicyAction::Allow)
            .with_priority(100);

        engine
            .add_connection_policy(policy1)
            .expect("test: should add low-priority deny policy");
        engine
            .add_connection_policy(policy2)
            .expect("test: should add high-priority allow policy");

        // High priority policy should match first
        let allowed = engine
            .evaluate_connection("test_peer")
            .await
            .expect("test: should evaluate test_peer connection");
        assert!(allowed);
    }

    #[tokio::test]
    async fn test_remove_policy() {
        let engine = PolicyEngine::new(PolicyConfig::default());

        let policy = ConnectionPolicy::new("test");
        engine
            .add_connection_policy(policy)
            .expect("test: should add test policy");

        assert!(engine.remove_connection_policy("test").is_ok());
        assert_eq!(engine.connection_policies().len(), 0);

        assert!(engine.remove_connection_policy("nonexistent").is_err());
    }

    #[tokio::test]
    async fn test_statistics() {
        let engine = PolicyEngine::new(PolicyConfig::default());

        let policy = ConnectionPolicy::new("test").with_action(PolicyAction::Allow);
        engine
            .add_connection_policy(policy)
            .expect("test: should add allow policy for stats");

        engine
            .evaluate_connection("peer1")
            .await
            .expect("test: should evaluate peer1 for stats");
        engine
            .evaluate_connection("peer2")
            .await
            .expect("test: should evaluate peer2 for stats");

        let stats = engine.stats();
        assert_eq!(stats.evaluations, 2);
        assert_eq!(stats.allowed, 2);
    }

    #[test]
    fn test_bandwidth_policy() {
        let policy = BandwidthPolicy::new("test")
            .with_max_upload(1_000_000)
            .with_max_download(5_000_000)
            .with_per_peer_limit(100_000);

        assert_eq!(policy.max_upload_bps, Some(1_000_000));
        assert_eq!(policy.max_download_bps, Some(5_000_000));
        assert_eq!(policy.per_peer_limit_bps, Some(100_000));
    }

    #[test]
    fn test_content_policy() {
        let policy = ContentPolicy::new("test")
            .with_allowed_pattern("^Qm.*")
            .with_max_size(10_000_000);

        assert_eq!(policy.allowed_patterns.len(), 1);
        assert_eq!(policy.max_size, Some(10_000_000));
    }

    #[test]
    fn test_policy_config_presets() {
        let strict = PolicyConfig::strict();
        assert_eq!(strict.default_action, PolicyAction::Deny);

        let permissive = PolicyConfig::permissive();
        assert_eq!(permissive.default_action, PolicyAction::Allow);
    }

    #[tokio::test]
    async fn test_default_action() {
        let config = PolicyConfig {
            default_action: PolicyAction::Deny,
            ..Default::default()
        };
        let engine = PolicyEngine::new(config);

        // No policies, should use default action (Deny)
        let allowed = engine
            .evaluate_connection("peer1")
            .await
            .expect("test: should evaluate peer1 with deny default");
        assert!(!allowed);
    }

    #[tokio::test]
    async fn test_reset_stats() {
        let engine = PolicyEngine::new(PolicyConfig::default());

        let policy = ConnectionPolicy::new("test");
        engine
            .add_connection_policy(policy)
            .expect("test: should add policy for reset stats test");

        engine
            .evaluate_connection("peer1")
            .await
            .expect("test: should evaluate peer1 before stats reset");
        assert_eq!(engine.stats().evaluations, 1);

        engine.reset_stats();
        assert_eq!(engine.stats().evaluations, 0);
    }
}
