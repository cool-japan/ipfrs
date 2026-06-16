//! Predictive Prefetching for intelligent block preloading
//!
//! This module provides predictive prefetching capabilities:
//! - Access pattern analysis and prediction
//! - Sequential access detection
//! - Co-location patterns (blocks accessed together)
//! - Time-based prediction (blocks accessed at similar times)
//! - Adaptive prefetch depth based on cache hit rates
//! - Background prefetching with priority control

use crate::traits::BlockStore;
use dashmap::DashMap;
use ipfrs_core::Cid;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::Semaphore;
use tracing::{debug, trace};

/// Access pattern type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessPattern {
    /// Sequential access (e.g., video streaming)
    Sequential,
    /// Random access with no clear pattern
    Random,
    /// Clustered access (accessing related blocks)
    Clustered,
    /// Temporal (accessing at regular intervals)
    Temporal,
}

/// Access record for a block
#[derive(Debug, Clone)]
struct AccessRecord {
    /// When the block was accessed
    #[allow(dead_code)]
    timestamp: SystemTime,
    /// Previous block accessed (for pattern detection)
    #[allow(dead_code)]
    previous_cid: Option<Cid>,
    /// Next block accessed (updated retroactively)
    next_cid: Option<Cid>,
}

/// Co-location pattern (blocks frequently accessed together)
#[derive(Debug, Clone)]
struct CoLocationPattern {
    /// How many times this pattern was observed
    count: u64,
    /// Last time this pattern was seen
    last_seen: SystemTime,
    /// Confidence score (0.0 - 1.0)
    confidence: f64,
}

/// Prefetch prediction
#[derive(Debug, Clone)]
pub struct PrefetchPrediction {
    /// CID to prefetch
    pub cid: Cid,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Predicted access time
    pub predicted_access: SystemTime,
    /// Pattern type that generated this prediction
    pub pattern: AccessPattern,
}

/// Prefetch configuration
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    /// Maximum number of blocks to prefetch ahead
    pub max_prefetch_depth: usize,
    /// Minimum confidence threshold for prefetching (0.0 - 1.0)
    pub min_confidence: f64,
    /// Maximum concurrent prefetch operations
    pub max_concurrent_prefetch: usize,
    /// Time window for pattern analysis
    pub pattern_window: Duration,
    /// Enable sequential pattern detection
    pub enable_sequential: bool,
    /// Enable co-location pattern detection
    pub enable_colocation: bool,
    /// Enable temporal pattern detection
    pub enable_temporal: bool,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            max_prefetch_depth: 5,
            min_confidence: 0.6,
            max_concurrent_prefetch: 3,
            pattern_window: Duration::from_secs(300), // 5 minutes
            enable_sequential: true,
            enable_colocation: true,
            enable_temporal: true,
        }
    }
}

/// Prefetch statistics
#[derive(Debug, Default)]
pub struct PrefetchStats {
    /// Total prefetch attempts
    pub prefetch_attempts: AtomicU64,
    /// Successful prefetches (block was used)
    pub prefetch_hits: AtomicU64,
    /// Wasted prefetches (block was not used)
    pub prefetch_misses: AtomicU64,
    /// Bytes prefetched
    pub bytes_prefetched: AtomicU64,
    /// Average confidence of predictions
    pub avg_confidence: parking_lot::Mutex<f64>,
}

impl PrefetchStats {
    fn record_attempt(&self) {
        self.prefetch_attempts.fetch_add(1, Ordering::Relaxed);
    }

    fn record_hit(&self, bytes: u64) {
        self.prefetch_hits.fetch_add(1, Ordering::Relaxed);
        self.bytes_prefetched.fetch_add(bytes, Ordering::Relaxed);
    }

    fn record_miss(&self) {
        self.prefetch_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Get hit rate
    pub fn hit_rate(&self) -> f64 {
        let hits = self.prefetch_hits.load(Ordering::Relaxed) as f64;
        let total = self.prefetch_attempts.load(Ordering::Relaxed) as f64;
        if total > 0.0 {
            hits / total
        } else {
            0.0
        }
    }
}

/// Predictive prefetcher
pub struct PredictivePrefetcher<S: BlockStore> {
    store: Arc<S>,
    config: parking_lot::RwLock<PrefetchConfig>,
    /// Access history for each CID
    access_history: DashMap<Cid, VecDeque<AccessRecord>>,
    /// Co-location patterns (CID -> related CIDs)
    colocation_patterns: DashMap<Cid, DashMap<Cid, CoLocationPattern>>,
    /// Last accessed CID (for detecting sequences)
    last_accessed: parking_lot::Mutex<Option<Cid>>,
    /// Prefetch queue
    #[allow(dead_code)]
    prefetch_queue: DashMap<Cid, PrefetchPrediction>,
    /// Prefetch cache (blocks that have been prefetched)
    prefetch_cache: DashMap<Cid, (Vec<u8>, SystemTime)>,
    /// Statistics
    stats: PrefetchStats,
    /// Semaphore for concurrent prefetch control
    prefetch_semaphore: Arc<Semaphore>,
    /// Current prefetch depth (adaptive)
    current_depth: AtomicUsize,
}

impl<S: BlockStore + Send + Sync + 'static> PredictivePrefetcher<S> {
    /// Create a new predictive prefetcher
    pub fn new(store: Arc<S>, config: PrefetchConfig) -> Self {
        let max_concurrent = config.max_concurrent_prefetch;
        let initial_depth = config.max_prefetch_depth;

        Self {
            store,
            config: parking_lot::RwLock::new(config),
            access_history: DashMap::new(),
            colocation_patterns: DashMap::new(),
            last_accessed: parking_lot::Mutex::new(None),
            prefetch_queue: DashMap::new(),
            prefetch_cache: DashMap::new(),
            stats: PrefetchStats::default(),
            prefetch_semaphore: Arc::new(Semaphore::new(max_concurrent)),
            current_depth: AtomicUsize::new(initial_depth),
        }
    }

    /// Record an access and update patterns
    pub fn record_access(&self, cid: &Cid) {
        let now = SystemTime::now();
        let previous = *self.last_accessed.lock();

        // Add to access history (drop guard before accessing prev_history to avoid deadlock)
        {
            let mut history = self.access_history.entry(*cid).or_default();
            history.push_back(AccessRecord {
                timestamp: now,
                previous_cid: previous,
                next_cid: None,
            });

            // Limit history size
            if history.len() > 100 {
                history.pop_front();
            }
        } // Guard dropped here

        // Update previous access with next CID (safe now since we dropped the guard above)
        if let Some(prev_cid) = previous {
            // Only update if prev_cid is different from cid to avoid unnecessary work
            if prev_cid != *cid {
                if let Some(mut prev_history) = self.access_history.get_mut(&prev_cid) {
                    if let Some(last_record) = prev_history.back_mut() {
                        last_record.next_cid = Some(*cid);
                    }
                }
            }

            // Update co-location patterns
            if self.config.read().enable_colocation {
                self.update_colocation_pattern(&prev_cid, cid);
            }
        }

        // Update last accessed
        *self.last_accessed.lock() = Some(*cid);

        // Check if this was prefetched
        if let Some(entry) = self.prefetch_cache.get(cid) {
            let prefetch_time = entry.value().1;
            let age = now.duration_since(prefetch_time).unwrap_or_default();
            if age < Duration::from_secs(60) {
                // Prefetch was used within 60 seconds - count as hit
                self.stats.record_hit(0); // We don't have size info here
            } else {
                self.stats.record_miss();
            }
        }
    }

    /// Update co-location pattern
    fn update_colocation_pattern(&self, cid1: &Cid, cid2: &Cid) {
        let patterns = self.colocation_patterns.entry(*cid1).or_default();

        patterns
            .entry(*cid2)
            .and_modify(|pattern| {
                pattern.count += 1;
                pattern.last_seen = SystemTime::now();
                // Update confidence based on recency and frequency
                let recency_factor = 0.9; // Decay factor
                pattern.confidence = (pattern.confidence * recency_factor + 0.1).min(1.0);
            })
            .or_insert_with(|| CoLocationPattern {
                count: 1,
                last_seen: SystemTime::now(),
                confidence: 0.5,
            });
    }

    /// Predict next blocks to access
    pub fn predict_next_blocks(&self, current_cid: &Cid) -> Vec<PrefetchPrediction> {
        let config = self.config.read();
        let mut predictions = Vec::new();

        // Sequential pattern prediction
        if config.enable_sequential {
            if let Some(seq_predictions) = self.predict_sequential(current_cid) {
                predictions.extend(seq_predictions);
            }
        }

        // Co-location pattern prediction
        if config.enable_colocation {
            if let Some(coloc_predictions) = self.predict_colocation(current_cid) {
                predictions.extend(coloc_predictions);
            }
        }

        // Filter by confidence and limit depth
        predictions.retain(|p| p.confidence >= config.min_confidence);
        predictions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let depth = self.current_depth.load(Ordering::Relaxed);
        predictions.truncate(depth);

        predictions
    }

    /// Predict based on sequential access pattern
    fn predict_sequential(&self, cid: &Cid) -> Option<Vec<PrefetchPrediction>> {
        let history = self.access_history.get(cid)?;

        // Check if there's a consistent "next" block
        let next_counts: DashMap<Cid, u64> = DashMap::new();

        for record in history.iter() {
            if let Some(next_cid) = record.next_cid {
                *next_counts.entry(next_cid).or_insert(0) += 1;
            }
        }

        if next_counts.is_empty() {
            return None;
        }

        // Find most common next block
        let mut predictions = Vec::new();
        let total_accesses = history.len() as f64;

        for entry in next_counts.iter() {
            let count = *entry.value() as f64;
            let confidence = count / total_accesses;

            if confidence >= 0.3 {
                predictions.push(PrefetchPrediction {
                    cid: *entry.key(),
                    confidence,
                    predicted_access: SystemTime::now(),
                    pattern: AccessPattern::Sequential,
                });
            }
        }

        Some(predictions)
    }

    /// Predict based on co-location patterns
    fn predict_colocation(&self, cid: &Cid) -> Option<Vec<PrefetchPrediction>> {
        let patterns = self.colocation_patterns.get(cid)?;

        let mut predictions = Vec::new();

        for entry in patterns.iter() {
            let pattern = entry.value();

            // Check if pattern is recent
            let age = SystemTime::now()
                .duration_since(pattern.last_seen)
                .unwrap_or_default();

            if age < self.config.read().pattern_window {
                predictions.push(PrefetchPrediction {
                    cid: *entry.key(),
                    confidence: pattern.confidence,
                    predicted_access: SystemTime::now(),
                    pattern: AccessPattern::Clustered,
                });
            }
        }

        Some(predictions)
    }

    /// Prefetch predicted blocks in background
    pub async fn prefetch_background(&self, predictions: Vec<PrefetchPrediction>) {
        for prediction in predictions {
            let store = self.store.clone();
            let cache = self.prefetch_cache.clone();
            let stats = &self.stats;
            let semaphore = self.prefetch_semaphore.clone();

            stats.record_attempt();

            let cid = prediction.cid;
            trace!(
                "Prefetching block {} (confidence: {:.2})",
                cid,
                prediction.confidence
            );

            // Spawn prefetch task
            tokio::spawn(async move {
                let _permit = semaphore.acquire().await.ok();

                if let Ok(Some(block)) = store.get(&cid).await {
                    cache.insert(cid, (block.data().to_vec(), SystemTime::now()));
                    debug!("Prefetched block {}", cid);
                }
            });
        }
    }

    /// Adapt prefetch depth based on hit rate
    pub fn adapt_depth(&self) {
        let hit_rate = self.stats.hit_rate();
        let current = self.current_depth.load(Ordering::Relaxed);
        let max_depth = self.config.read().max_prefetch_depth;

        let new_depth = if hit_rate > 0.8 {
            // High hit rate - increase depth
            (current + 1).min(max_depth)
        } else if hit_rate < 0.4 {
            // Low hit rate - decrease depth
            (current.saturating_sub(1)).max(1)
        } else {
            current
        };

        if new_depth != current {
            self.current_depth.store(new_depth, Ordering::Relaxed);
            debug!(
                "Adapted prefetch depth: {} -> {} (hit rate: {:.2})",
                current, new_depth, hit_rate
            );
        }
    }

    /// Get statistics
    pub fn stats(&self) -> PrefetchStatsSnapshot {
        PrefetchStatsSnapshot {
            prefetch_attempts: self.stats.prefetch_attempts.load(Ordering::Relaxed),
            prefetch_hits: self.stats.prefetch_hits.load(Ordering::Relaxed),
            prefetch_misses: self.stats.prefetch_misses.load(Ordering::Relaxed),
            bytes_prefetched: self.stats.bytes_prefetched.load(Ordering::Relaxed),
            hit_rate: self.stats.hit_rate(),
            current_depth: self.current_depth.load(Ordering::Relaxed),
        }
    }

    /// Clear prefetch cache
    pub fn clear_cache(&self) {
        self.prefetch_cache.clear();
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.prefetch_cache.len()
    }
}

/// Snapshot of prefetch statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefetchStatsSnapshot {
    pub prefetch_attempts: u64,
    pub prefetch_hits: u64,
    pub prefetch_misses: u64,
    pub bytes_prefetched: u64,
    pub hit_rate: f64,
    pub current_depth: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryBlockStore;
    use ipfrs_core::cid::CidBuilder;

    /// Helper to create a unique CID from an index
    fn test_cid(index: u64) -> Cid {
        CidBuilder::new()
            .build(&index.to_le_bytes())
            .expect("failed to create test cid")
    }

    #[tokio::test]
    async fn test_prefetcher_creation() {
        let store = Arc::new(MemoryBlockStore::new());
        let config = PrefetchConfig::default();
        let prefetcher = PredictivePrefetcher::new(store, config);

        let stats = prefetcher.stats();
        assert_eq!(stats.prefetch_attempts, 0);
        assert_eq!(stats.hit_rate, 0.0);
    }

    #[tokio::test]
    async fn test_access_recording() {
        let store = Arc::new(MemoryBlockStore::new());
        let prefetcher = PredictivePrefetcher::new(store, PrefetchConfig::default());

        let cid1 = test_cid(1);
        let cid2 = test_cid(2);

        prefetcher.record_access(&cid1);
        prefetcher.record_access(&cid2);

        // Should have recorded co-location pattern
        assert!(prefetcher.colocation_patterns.contains_key(&cid1));
    }

    #[tokio::test]
    async fn test_sequential_prediction() {
        let store = Arc::new(MemoryBlockStore::new());
        let prefetcher = PredictivePrefetcher::new(store, PrefetchConfig::default());

        let cid1 = test_cid(1);
        let cid2 = test_cid(2);

        // Simulate sequential access pattern
        for _ in 0..5 {
            prefetcher.record_access(&cid1);
            prefetcher.record_access(&cid2);
        }

        let predictions = prefetcher.predict_next_blocks(&cid1);
        assert!(!predictions.is_empty());

        // Should predict cid2 after cid1
        assert!(predictions
            .iter()
            .any(|p| p.pattern == AccessPattern::Sequential));
    }
}
