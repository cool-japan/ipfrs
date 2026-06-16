//! Lifecycle Policies for automatic data management
//!
//! This module provides lifecycle management for storage blocks:
//! - Age-based tiering (move to cold storage after N days)
//! - Access-based tiering (archive rarely accessed data)
//! - Size-based policies (different rules for different block sizes)
//! - Automatic expiration and deletion
//! - Policy evaluation and enforcement
//! - Lifecycle statistics and reporting

use crate::traits::BlockStore;
use dashmap::DashMap;
use ipfrs_core::Cid;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};
use tracing::{debug, info};

/// Lifecycle action
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifecycleAction {
    /// Move to a different storage tier
    Transition(StorageTier),
    /// Delete the block permanently
    Delete,
    /// Archive (compress and move to cold storage)
    Archive,
    /// Mark for manual review
    Review,
}

/// Storage tier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StorageTier {
    /// Hot storage (fast, expensive)
    Hot,
    /// Warm storage (balanced)
    Warm,
    /// Cold storage (slow, cheap)
    Cold,
    /// Archive storage (very slow, very cheap)
    Archive,
}

/// Lifecycle rule condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LifecycleCondition {
    /// Age in days since creation
    AgeDays(u32),
    /// Number of days since last access
    DaysSinceLastAccess(u32),
    /// Access count below threshold
    AccessCountBelow(u64),
    /// Block size in bytes
    SizeBytes { min: Option<u64>, max: Option<u64> },
    /// Current storage tier
    CurrentTier(StorageTier),
    /// Multiple conditions (AND)
    And(Vec<LifecycleCondition>),
    /// Multiple conditions (OR)
    Or(Vec<LifecycleCondition>),
}

/// Lifecycle rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleRule {
    /// Rule identifier
    pub id: String,
    /// Rule description
    pub description: String,
    /// Condition to trigger this rule
    pub condition: LifecycleCondition,
    /// Action to take when condition is met
    pub action: LifecycleAction,
    /// Rule priority (higher = evaluated first)
    pub priority: u32,
    /// Whether the rule is enabled
    pub enabled: bool,
}

impl LifecycleRule {
    /// Create a new lifecycle rule
    pub fn new(
        id: String,
        description: String,
        condition: LifecycleCondition,
        action: LifecycleAction,
    ) -> Self {
        Self {
            id,
            description,
            condition,
            action,
            priority: 100,
            enabled: true,
        }
    }

    /// Check if the rule applies to a block
    #[allow(dead_code)]
    fn matches(&self, metadata: &BlockMetadata) -> bool {
        if !self.enabled {
            return false;
        }
        self.evaluate_condition(&self.condition, metadata)
    }

    fn evaluate_condition(&self, condition: &LifecycleCondition, metadata: &BlockMetadata) -> bool {
        match condition {
            LifecycleCondition::AgeDays(days) => {
                let age = SystemTime::now()
                    .duration_since(metadata.created_at)
                    .unwrap_or_default();
                age >= Duration::from_secs(*days as u64 * 86400)
            }
            LifecycleCondition::DaysSinceLastAccess(days) => {
                if let Some(last_access) = metadata.last_accessed {
                    let duration = SystemTime::now()
                        .duration_since(last_access)
                        .unwrap_or_default();
                    duration >= Duration::from_secs(*days as u64 * 86400)
                } else {
                    // Never accessed = treat as very old
                    *days == 0
                }
            }
            LifecycleCondition::AccessCountBelow(threshold) => metadata.access_count < *threshold,
            LifecycleCondition::SizeBytes { min, max } => {
                if let Some(min_size) = min {
                    if metadata.size < *min_size {
                        return false;
                    }
                }
                if let Some(max_size) = max {
                    if metadata.size > *max_size {
                        return false;
                    }
                }
                true
            }
            LifecycleCondition::CurrentTier(tier) => metadata.tier == *tier,
            LifecycleCondition::And(conditions) => conditions
                .iter()
                .all(|c| self.evaluate_condition(c, metadata)),
            LifecycleCondition::Or(conditions) => conditions
                .iter()
                .any(|c| self.evaluate_condition(c, metadata)),
        }
    }
}

/// Block metadata for lifecycle management
#[derive(Debug, Clone)]
pub struct BlockMetadata {
    /// Block CID
    pub cid: Cid,
    /// Block size in bytes
    pub size: u64,
    /// Creation time
    pub created_at: SystemTime,
    /// Last access time
    pub last_accessed: Option<SystemTime>,
    /// Number of accesses
    pub access_count: u64,
    /// Current storage tier
    pub tier: StorageTier,
}

/// Lifecycle policy configuration
#[derive(Debug, Clone)]
pub struct LifecyclePolicyConfig {
    /// Evaluation interval
    pub evaluation_interval: Duration,
    /// Maximum actions per evaluation
    pub max_actions_per_evaluation: usize,
    /// Dry run mode (don't actually perform actions)
    pub dry_run: bool,
}

impl Default for LifecyclePolicyConfig {
    fn default() -> Self {
        Self {
            evaluation_interval: Duration::from_secs(3600), // 1 hour
            max_actions_per_evaluation: 1000,
            dry_run: false,
        }
    }
}

/// Lifecycle action result
#[derive(Debug, Clone)]
pub struct LifecycleActionResult {
    /// Block CID
    pub cid: Cid,
    /// Rule that triggered the action
    pub rule_id: String,
    /// Action taken
    pub action: LifecycleAction,
    /// Whether the action succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Lifecycle statistics
#[derive(Debug, Default)]
pub struct LifecycleStats {
    /// Total evaluations run
    pub evaluations_run: AtomicU64,
    /// Total blocks evaluated
    pub blocks_evaluated: AtomicU64,
    /// Actions taken by type
    pub transitions: AtomicU64,
    pub deletions: AtomicU64,
    pub archives: AtomicU64,
    pub reviews: AtomicU64,
    /// Failed actions
    pub failures: AtomicU64,
}

impl LifecycleStats {
    fn record_evaluation(&self, blocks_count: u64) {
        self.evaluations_run.fetch_add(1, Ordering::Relaxed);
        self.blocks_evaluated
            .fetch_add(blocks_count, Ordering::Relaxed);
    }

    fn record_action(&self, action: LifecycleAction, success: bool) {
        if success {
            match action {
                LifecycleAction::Transition(_) => {
                    self.transitions.fetch_add(1, Ordering::Relaxed);
                }
                LifecycleAction::Delete => {
                    self.deletions.fetch_add(1, Ordering::Relaxed);
                }
                LifecycleAction::Archive => {
                    self.archives.fetch_add(1, Ordering::Relaxed);
                }
                LifecycleAction::Review => {
                    self.reviews.fetch_add(1, Ordering::Relaxed);
                }
            }
        } else {
            self.failures.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Lifecycle policy manager
pub struct LifecyclePolicyManager {
    rules: parking_lot::RwLock<Vec<LifecycleRule>>,
    metadata: DashMap<Cid, BlockMetadata>,
    config: parking_lot::RwLock<LifecyclePolicyConfig>,
    stats: LifecycleStats,
}

impl LifecyclePolicyManager {
    /// Create a new lifecycle policy manager
    pub fn new(config: LifecyclePolicyConfig) -> Self {
        Self {
            rules: parking_lot::RwLock::new(Vec::new()),
            metadata: DashMap::new(),
            config: parking_lot::RwLock::new(config),
            stats: LifecycleStats::default(),
        }
    }

    /// Add a lifecycle rule
    pub fn add_rule(&self, rule: LifecycleRule) {
        let mut rules = self.rules.write();
        rules.push(rule.clone());
        rules.sort_by_key(|r| std::cmp::Reverse(r.priority));
        debug!("Added lifecycle rule: {}", rule.id);
    }

    /// Remove a lifecycle rule
    pub fn remove_rule(&self, rule_id: &str) -> bool {
        let mut rules = self.rules.write();
        if let Some(pos) = rules.iter().position(|r| r.id == rule_id) {
            rules.remove(pos);
            debug!("Removed lifecycle rule: {}", rule_id);
            true
        } else {
            false
        }
    }

    /// Get all rules
    pub fn get_rules(&self) -> Vec<LifecycleRule> {
        self.rules.read().clone()
    }

    /// Register block metadata
    pub fn register_block(&self, metadata: BlockMetadata) {
        self.metadata.insert(metadata.cid, metadata);
    }

    /// Update block access
    pub fn record_access(&self, cid: &Cid) {
        if let Some(mut entry) = self.metadata.get_mut(cid) {
            entry.last_accessed = Some(SystemTime::now());
            entry.access_count += 1;
        }
    }

    /// Evaluate policies and return actions to take
    pub fn evaluate(&self) -> Vec<LifecycleActionResult> {
        let rules = self.rules.read();
        let config = self.config.read();
        let mut results = Vec::new();

        let blocks_count = self.metadata.len() as u64;
        self.stats.record_evaluation(blocks_count);

        for entry in self.metadata.iter() {
            if results.len() >= config.max_actions_per_evaluation {
                break;
            }

            let metadata = entry.value();

            // Find first matching rule
            for rule in rules.iter() {
                if rule.matches(metadata) {
                    let result = LifecycleActionResult {
                        cid: metadata.cid,
                        rule_id: rule.id.clone(),
                        action: rule.action,
                        success: !config.dry_run,
                        error: if config.dry_run {
                            Some("Dry run mode".to_string())
                        } else {
                            None
                        },
                    };

                    self.stats.record_action(rule.action, !config.dry_run);
                    results.push(result);
                    break; // Only apply first matching rule
                }
            }
        }

        if !results.is_empty() {
            info!(
                "Lifecycle evaluation completed: {} actions recommended",
                results.len()
            );
        }

        results
    }

    /// Apply lifecycle actions to a block store
    pub async fn apply_actions<S: BlockStore>(
        &self,
        store: &S,
        actions: Vec<LifecycleActionResult>,
    ) -> Vec<LifecycleActionResult> {
        let mut results = Vec::new();

        for action in actions {
            if self.config.read().dry_run {
                results.push(action);
                continue;
            }

            let success = match action.action {
                LifecycleAction::Delete => store.delete(&action.cid).await.is_ok(),
                LifecycleAction::Transition(tier) => {
                    // Update metadata
                    if let Some(mut entry) = self.metadata.get_mut(&action.cid) {
                        entry.tier = tier;
                        true
                    } else {
                        false
                    }
                }
                LifecycleAction::Archive | LifecycleAction::Review => {
                    // These would typically involve external systems
                    true
                }
            };

            results.push(LifecycleActionResult {
                success,
                error: if success {
                    None
                } else {
                    Some("Action failed".to_string())
                },
                ..action
            });

            self.stats.record_action(action.action, success);
        }

        results
    }

    /// Get lifecycle statistics
    pub fn get_stats(&self) -> LifecycleStatsSnapshot {
        LifecycleStatsSnapshot {
            evaluations_run: self.stats.evaluations_run.load(Ordering::Relaxed),
            blocks_evaluated: self.stats.blocks_evaluated.load(Ordering::Relaxed),
            transitions: self.stats.transitions.load(Ordering::Relaxed),
            deletions: self.stats.deletions.load(Ordering::Relaxed),
            archives: self.stats.archives.load(Ordering::Relaxed),
            reviews: self.stats.reviews.load(Ordering::Relaxed),
            failures: self.stats.failures.load(Ordering::Relaxed),
        }
    }

    /// Get blocks by tier
    pub fn get_blocks_by_tier(&self, tier: StorageTier) -> Vec<Cid> {
        self.metadata
            .iter()
            .filter_map(|entry| {
                if entry.value().tier == tier {
                    Some(*entry.key())
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Snapshot of lifecycle statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleStatsSnapshot {
    pub evaluations_run: u64,
    pub blocks_evaluated: u64,
    pub transitions: u64,
    pub deletions: u64,
    pub archives: u64,
    pub reviews: u64,
    pub failures: u64,
}

/// Common lifecycle rule presets
impl LifecycleRule {
    /// Move to cold storage after 30 days
    pub fn archive_old_blocks() -> Self {
        Self::new(
            "archive_old".to_string(),
            "Move blocks older than 30 days to cold storage".to_string(),
            LifecycleCondition::AgeDays(30),
            LifecycleAction::Transition(StorageTier::Cold),
        )
    }

    /// Delete blocks not accessed in 90 days
    pub fn delete_unused() -> Self {
        Self::new(
            "delete_unused".to_string(),
            "Delete blocks not accessed in 90 days".to_string(),
            LifecycleCondition::DaysSinceLastAccess(90),
            LifecycleAction::Delete,
        )
    }

    /// Archive large blocks after 7 days
    pub fn archive_large_blocks() -> Self {
        Self::new(
            "archive_large".to_string(),
            "Archive blocks larger than 10MB after 7 days".to_string(),
            LifecycleCondition::And(vec![
                LifecycleCondition::AgeDays(7),
                LifecycleCondition::SizeBytes {
                    min: Some(10 * 1024 * 1024),
                    max: None,
                },
            ]),
            LifecycleAction::Archive,
        )
    }

    /// Move rarely accessed hot storage to warm
    pub fn demote_cold_hot_storage() -> Self {
        Self::new(
            "demote_hot".to_string(),
            "Move rarely accessed hot storage blocks to warm tier".to_string(),
            LifecycleCondition::And(vec![
                LifecycleCondition::CurrentTier(StorageTier::Hot),
                LifecycleCondition::DaysSinceLastAccess(7),
                LifecycleCondition::AccessCountBelow(10),
            ]),
            LifecycleAction::Transition(StorageTier::Warm),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_age_condition() {
        let rule = LifecycleRule::new(
            "test".to_string(),
            "Test rule".to_string(),
            LifecycleCondition::AgeDays(1),
            LifecycleAction::Delete,
        );

        let old_metadata = BlockMetadata {
            cid: Cid::default(),
            size: 100,
            created_at: SystemTime::now() - Duration::from_secs(2 * 86400),
            last_accessed: None,
            access_count: 0,
            tier: StorageTier::Hot,
        };

        assert!(rule.matches(&old_metadata));

        let new_metadata = BlockMetadata {
            cid: Cid::default(),
            size: 100,
            created_at: SystemTime::now(),
            last_accessed: None,
            access_count: 0,
            tier: StorageTier::Hot,
        };

        assert!(!rule.matches(&new_metadata));
    }

    #[test]
    fn test_lifecycle_manager() {
        let manager = LifecyclePolicyManager::new(LifecyclePolicyConfig::default());

        // Add a rule
        manager.add_rule(LifecycleRule::archive_old_blocks());

        // Register an old block
        let metadata = BlockMetadata {
            cid: Cid::default(),
            size: 100,
            created_at: SystemTime::now() - Duration::from_secs(31 * 86400),
            last_accessed: None,
            access_count: 0,
            tier: StorageTier::Hot,
        };

        manager.register_block(metadata);

        // Evaluate
        let actions = manager.evaluate();
        assert_eq!(actions.len(), 1);
        assert_eq!(
            actions[0].action,
            LifecycleAction::Transition(StorageTier::Cold)
        );
    }

    #[test]
    fn test_rule_presets() {
        let rule = LifecycleRule::delete_unused();
        assert_eq!(rule.id, "delete_unused");
        assert_eq!(rule.action, LifecycleAction::Delete);

        let rule = LifecycleRule::archive_large_blocks();
        assert_eq!(rule.action, LifecycleAction::Archive);
    }
}
