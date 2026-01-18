//! Intelligent peer selection combining geographic proximity and connection quality
//!
//! This module provides a smart peer selector that combines multiple factors:
//! - Geographic proximity (via geo_routing)
//! - Connection quality prediction (via quality_predictor)
//! - Network topology optimization
//!
//! ## Features
//!
//! - **Multi-factor scoring**: Combines distance, latency, bandwidth, reliability
//! - **Configurable weights**: Adjust importance of each factor
//! - **Smart caching**: Cache selection decisions to reduce overhead
//! - **Adaptive scoring**: Learn from connection outcomes
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_network::peer_selector::{PeerSelector, PeerSelectorConfig, SelectionCriteria};
//! use ipfrs_network::geo_routing::GeoLocation;
//! use libp2p::PeerId;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = PeerSelectorConfig::balanced();
//! let mut selector = PeerSelector::new(config);
//!
//! // Add peer with location and quality metrics
//! let peer = PeerId::random();
//! selector.add_peer_location(peer, GeoLocation::new(40.7128, -74.0060));
//!
//! // Select best peers based on criteria
//! let my_location = GeoLocation::new(37.7749, -122.4194);
//! let criteria = SelectionCriteria {
//!     reference_location: Some(my_location),
//!     min_quality_score: 0.5,
//!     max_distance_km: Some(5000.0),
//!     max_results: 10,
//! };
//!
//! let selected = selector.select_peers(&criteria);
//! println!("Selected {} peers", selected.len());
//! # Ok(())
//! # }
//! ```

use crate::geo_routing::{GeoLocation, GeoRouter, GeoRouterConfig};
use crate::quality_predictor::{QualityPredictor, QualityPredictorConfig};
use dashmap::DashMap;
use libp2p::PeerId;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for peer selector
#[derive(Debug, Clone)]
pub struct PeerSelectorConfig {
    /// Weight for geographic distance (0.0 - 1.0)
    pub distance_weight: f64,
    /// Weight for connection quality (0.0 - 1.0)
    pub quality_weight: f64,
    /// Weight for latency (0.0 - 1.0)
    pub latency_weight: f64,
    /// Weight for bandwidth (0.0 - 1.0)
    pub bandwidth_weight: f64,
    /// Enable selection caching
    pub enable_caching: bool,
    /// Cache TTL in seconds
    pub cache_ttl_secs: u64,
    /// Maximum cache size
    pub max_cache_entries: usize,
}

impl Default for PeerSelectorConfig {
    fn default() -> Self {
        Self {
            distance_weight: 0.3,
            quality_weight: 0.3,
            latency_weight: 0.2,
            bandwidth_weight: 0.2,
            enable_caching: true,
            cache_ttl_secs: 300, // 5 minutes
            max_cache_entries: 1000,
        }
    }
}

impl PeerSelectorConfig {
    /// Configuration optimized for low latency
    pub fn low_latency() -> Self {
        Self {
            distance_weight: 0.4,
            quality_weight: 0.1,
            latency_weight: 0.4,
            bandwidth_weight: 0.1,
            enable_caching: true,
            cache_ttl_secs: 180,
            max_cache_entries: 500,
        }
    }

    /// Configuration optimized for high bandwidth
    pub fn high_bandwidth() -> Self {
        Self {
            distance_weight: 0.1,
            quality_weight: 0.2,
            latency_weight: 0.1,
            bandwidth_weight: 0.6,
            enable_caching: true,
            cache_ttl_secs: 300,
            max_cache_entries: 1000,
        }
    }

    /// Balanced configuration
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Configuration for mobile/constrained devices
    pub fn mobile() -> Self {
        Self {
            distance_weight: 0.5, // Prefer nearby peers
            quality_weight: 0.3,
            latency_weight: 0.1,
            bandwidth_weight: 0.1,
            enable_caching: true,
            cache_ttl_secs: 600, // Cache longer
            max_cache_entries: 200,
        }
    }
}

/// Criteria for peer selection
#[derive(Debug, Clone)]
pub struct SelectionCriteria {
    /// Reference location for distance calculation
    pub reference_location: Option<GeoLocation>,
    /// Minimum quality score (0.0 - 1.0)
    pub min_quality_score: f64,
    /// Maximum distance in kilometers
    pub max_distance_km: Option<f64>,
    /// Maximum number of results
    pub max_results: usize,
}

impl Default for SelectionCriteria {
    fn default() -> Self {
        Self {
            reference_location: None,
            min_quality_score: 0.0,
            max_distance_km: None,
            max_results: 10,
        }
    }
}

/// Selected peer with score details
#[derive(Debug, Clone)]
pub struct SelectedPeer {
    /// Peer ID
    pub peer_id: PeerId,
    /// Overall score (0.0 - 1.0, higher is better)
    pub score: f64,
    /// Distance score component
    pub distance_score: f64,
    /// Quality score component
    pub quality_score: f64,
    /// Latency score component
    pub latency_score: f64,
    /// Bandwidth score component
    pub bandwidth_score: f64,
    /// Geographic location (if available)
    pub location: Option<GeoLocation>,
    /// Distance in kilometers (if location available)
    pub distance_km: Option<f64>,
}

/// Cached selection result
#[derive(Debug, Clone)]
struct CachedSelection {
    /// Selected peers
    peers: Vec<SelectedPeer>,
    /// Timestamp when cached
    cached_at: Instant,
}

/// Statistics for peer selector
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PeerSelectorStats {
    /// Total selections performed
    pub total_selections: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Total peers evaluated
    pub total_peers_evaluated: u64,
    /// Average selection time in microseconds
    pub avg_selection_time_us: f64,
}

impl PeerSelectorStats {
    /// Calculate cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total > 0 {
            self.cache_hits as f64 / total as f64
        } else {
            0.0
        }
    }
}

/// Intelligent peer selector
pub struct PeerSelector {
    /// Configuration
    config: PeerSelectorConfig,
    /// Geographic router
    geo_router: Arc<GeoRouter>,
    /// Quality predictor
    quality_predictor: Arc<QualityPredictor>,
    /// Selection cache
    cache: Arc<DashMap<String, CachedSelection>>,
    /// Statistics
    stats: Arc<RwLock<PeerSelectorStats>>,
}

impl PeerSelector {
    /// Create a new peer selector with default configuration
    pub fn new(config: PeerSelectorConfig) -> Self {
        let geo_config = GeoRouterConfig::default();
        let quality_config = QualityPredictorConfig::default();

        Self {
            config,
            geo_router: Arc::new(GeoRouter::new(geo_config)),
            quality_predictor: Arc::new(
                QualityPredictor::new(quality_config).expect("Default config should be valid"),
            ),
            cache: Arc::new(DashMap::new()),
            stats: Arc::new(RwLock::new(PeerSelectorStats::default())),
        }
    }

    /// Create with custom geo and quality configurations
    pub fn with_configs(
        config: PeerSelectorConfig,
        geo_config: GeoRouterConfig,
        quality_config: QualityPredictorConfig,
    ) -> Self {
        Self {
            config,
            geo_router: Arc::new(GeoRouter::new(geo_config)),
            quality_predictor: Arc::new(
                QualityPredictor::new(quality_config).expect("Config should be valid"),
            ),
            cache: Arc::new(DashMap::new()),
            stats: Arc::new(RwLock::new(PeerSelectorStats::default())),
        }
    }

    /// Add or update peer location
    pub fn add_peer_location(&self, peer_id: PeerId, location: GeoLocation) {
        self.geo_router.update_peer_location(peer_id, location);
        self.invalidate_cache();
    }

    /// Remove peer
    pub fn remove_peer(&self, peer_id: &PeerId) {
        self.geo_router.remove_peer(peer_id);
        self.quality_predictor.remove_peer(peer_id);
        self.invalidate_cache();
    }

    /// Update peer quality metrics
    pub fn update_peer_quality(
        &self,
        peer_id: PeerId,
        latency_ms: f64,
        bandwidth_mbps: f64,
        success: bool,
    ) {
        self.quality_predictor
            .record_latency(peer_id, latency_ms as u64);
        // Convert Mbps to bytes per second
        let bytes_per_sec = (bandwidth_mbps * 1_000_000.0 / 8.0) as u64;
        self.quality_predictor
            .record_bandwidth(peer_id, bytes_per_sec);
        if !success {
            self.quality_predictor.record_failure(peer_id);
        }
        self.invalidate_cache();
    }

    /// Select best peers based on criteria
    pub fn select_peers(&self, criteria: &SelectionCriteria) -> Vec<SelectedPeer> {
        let start = Instant::now();

        // Check cache if enabled
        if self.config.enable_caching {
            let cache_key = self.make_cache_key(criteria);
            if let Some(cached) = self.cache.get(&cache_key) {
                let age = start.duration_since(cached.cached_at);
                if age.as_secs() < self.config.cache_ttl_secs {
                    let mut stats = self.stats.write();
                    stats.total_selections += 1;
                    stats.cache_hits += 1;
                    return cached.peers.clone();
                }
            }
        }

        // Perform selection
        let selected = self.select_peers_impl(criteria);

        // Update statistics
        let elapsed = start.elapsed();
        let mut stats = self.stats.write();
        stats.total_selections += 1;
        stats.cache_misses += 1;
        stats.total_peers_evaluated += selected.len() as u64;
        let new_avg = if stats.total_selections > 1 {
            (stats.avg_selection_time_us * (stats.total_selections - 1) as f64
                + elapsed.as_micros() as f64)
                / stats.total_selections as f64
        } else {
            elapsed.as_micros() as f64
        };
        stats.avg_selection_time_us = new_avg;
        drop(stats);

        // Cache result
        if self.config.enable_caching {
            let cache_key = self.make_cache_key(criteria);
            self.cache.insert(
                cache_key,
                CachedSelection {
                    peers: selected.clone(),
                    cached_at: start,
                },
            );

            // Enforce cache size limit
            if self.cache.len() > self.config.max_cache_entries {
                self.evict_old_cache_entries();
            }
        }

        selected
    }

    /// Internal implementation of peer selection
    fn select_peers_impl(&self, criteria: &SelectionCriteria) -> Vec<SelectedPeer> {
        // Get all peers with locations
        let mut scored_peers = Vec::new();

        // Get geo-ranked peers if location provided
        let geo_peers = if let Some(ref_location) = &criteria.reference_location {
            self.geo_router.rank_peers_by_proximity(ref_location)
        } else {
            vec![]
        };

        for geo_peer in geo_peers {
            // Check distance constraint
            if let Some(max_dist) = criteria.max_distance_km {
                if let Some(dist) = geo_peer.distance_km {
                    if dist > max_dist {
                        continue;
                    }
                }
            }

            // Calculate component scores
            let distance_score = self.calculate_distance_score(&geo_peer.distance_km);
            let quality_prediction = self.quality_predictor.predict_quality(&geo_peer.peer_id);

            let quality_score = quality_prediction
                .as_ref()
                .map(|p| p.overall_score)
                .unwrap_or(0.5);
            let latency_score = quality_prediction
                .as_ref()
                .map(|p| p.latency_score)
                .unwrap_or(0.5);
            let bandwidth_score = quality_prediction
                .as_ref()
                .map(|p| p.bandwidth_score)
                .unwrap_or(0.5);

            // Check minimum quality
            if quality_score < criteria.min_quality_score {
                continue;
            }

            // Calculate overall score
            let overall_score = self.calculate_overall_score(
                distance_score,
                quality_score,
                latency_score,
                bandwidth_score,
            );

            scored_peers.push(SelectedPeer {
                peer_id: geo_peer.peer_id,
                score: overall_score,
                distance_score,
                quality_score,
                latency_score,
                bandwidth_score,
                location: Some(geo_peer.location),
                distance_km: geo_peer.distance_km,
            });
        }

        // Sort by score (descending)
        scored_peers.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return top N results
        scored_peers.truncate(criteria.max_results);
        scored_peers
    }

    /// Calculate distance score (0.0 - 1.0, higher is better/closer)
    fn calculate_distance_score(&self, distance_km: &Option<f64>) -> f64 {
        match distance_km {
            Some(dist) => {
                // Use exponential decay: score = e^(-dist/1000)
                // At 0 km: score = 1.0
                // At 1000 km: score ≈ 0.368
                // At 5000 km: score ≈ 0.007
                (-dist / 1000.0).exp()
            }
            None => 0.5, // Neutral score if no location
        }
    }

    /// Calculate overall score from components
    fn calculate_overall_score(
        &self,
        distance: f64,
        quality: f64,
        latency: f64,
        bandwidth: f64,
    ) -> f64 {
        distance * self.config.distance_weight
            + quality * self.config.quality_weight
            + latency * self.config.latency_weight
            + bandwidth * self.config.bandwidth_weight
    }

    /// Generate cache key from criteria
    fn make_cache_key(&self, criteria: &SelectionCriteria) -> String {
        format!(
            "loc:{:?}_qual:{}_dist:{:?}_max:{}",
            criteria.reference_location,
            criteria.min_quality_score,
            criteria.max_distance_km,
            criteria.max_results
        )
    }

    /// Invalidate entire cache
    fn invalidate_cache(&self) {
        self.cache.clear();
    }

    /// Evict old cache entries to maintain size limit
    fn evict_old_cache_entries(&self) {
        let now = Instant::now();
        let ttl = Duration::from_secs(self.config.cache_ttl_secs);

        self.cache
            .retain(|_, entry| now.duration_since(entry.cached_at) < ttl);

        // If still over limit, remove random entries
        while self.cache.len() > self.config.max_cache_entries {
            if let Some(entry) = self.cache.iter().next() {
                let key = entry.key().clone();
                drop(entry);
                self.cache.remove(&key);
            } else {
                break;
            }
        }
    }

    /// Get statistics
    pub fn stats(&self) -> PeerSelectorStats {
        self.stats.read().clone()
    }

    /// Clear all caches and reset
    #[allow(dead_code)]
    pub fn reset(&self) {
        self.cache.clear();
        let mut stats = self.stats.write();
        *stats = PeerSelectorStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peer_selector_config() {
        let config = PeerSelectorConfig::default();
        assert_eq!(config.distance_weight, 0.3);
        assert_eq!(config.quality_weight, 0.3);
    }

    #[test]
    fn test_config_presets() {
        let low_latency = PeerSelectorConfig::low_latency();
        assert!(low_latency.latency_weight > 0.3);

        let high_bandwidth = PeerSelectorConfig::high_bandwidth();
        assert!(high_bandwidth.bandwidth_weight > 0.5);

        let mobile = PeerSelectorConfig::mobile();
        assert!(mobile.distance_weight >= 0.5);
    }

    #[test]
    fn test_peer_selector_creation() {
        let config = PeerSelectorConfig::default();
        let selector = PeerSelector::new(config);
        let stats = selector.stats();
        assert_eq!(stats.total_selections, 0);
    }

    #[test]
    fn test_add_peer_location() {
        let config = PeerSelectorConfig::default();
        let selector = PeerSelector::new(config);

        let peer = PeerId::random();
        let location = GeoLocation::new(40.7128, -74.0060);
        selector.add_peer_location(peer, location);

        // Verify peer was added (indirectly through geo_router)
        let loc = selector.geo_router.get_peer_location(&peer);
        assert!(loc.is_some());
    }

    #[test]
    fn test_remove_peer() {
        let config = PeerSelectorConfig::default();
        let selector = PeerSelector::new(config);

        let peer = PeerId::random();
        let location = GeoLocation::new(40.7128, -74.0060);
        selector.add_peer_location(peer, location);
        selector.remove_peer(&peer);

        let loc = selector.geo_router.get_peer_location(&peer);
        assert!(loc.is_none());
    }

    #[test]
    fn test_selection_criteria() {
        let criteria = SelectionCriteria {
            reference_location: Some(GeoLocation::new(37.7749, -122.4194)),
            min_quality_score: 0.5,
            max_distance_km: Some(1000.0),
            max_results: 5,
        };

        assert!(criteria.reference_location.is_some());
        assert_eq!(criteria.max_results, 5);
    }

    #[test]
    fn test_select_peers_empty() {
        let config = PeerSelectorConfig::default();
        let selector = PeerSelector::new(config);

        let criteria = SelectionCriteria::default();
        let selected = selector.select_peers(&criteria);
        assert_eq!(selected.len(), 0);
    }

    #[test]
    fn test_select_peers_with_location() {
        let config = PeerSelectorConfig::default();
        let selector = PeerSelector::new(config);

        // Add peers
        let peer1 = PeerId::random();
        let peer2 = PeerId::random();
        selector.add_peer_location(peer1, GeoLocation::new(40.7128, -74.0060)); // NY
        selector.add_peer_location(peer2, GeoLocation::new(34.0522, -118.2437)); // LA

        // Select from SF
        let criteria = SelectionCriteria {
            reference_location: Some(GeoLocation::new(37.7749, -122.4194)),
            min_quality_score: 0.0,
            max_distance_km: None,
            max_results: 10,
        };

        let selected = selector.select_peers(&criteria);
        assert!(!selected.is_empty());
        // LA should be closer to SF than NY
        if selected.len() == 2 {
            assert_eq!(selected[0].peer_id, peer2);
        }
    }

    #[test]
    fn test_distance_score_calculation() {
        let config = PeerSelectorConfig::default();
        let selector = PeerSelector::new(config);

        let score_close = selector.calculate_distance_score(&Some(100.0));
        let score_far = selector.calculate_distance_score(&Some(5000.0));
        let score_none = selector.calculate_distance_score(&None);

        assert!(score_close > score_far);
        assert_eq!(score_none, 0.5);
    }

    #[test]
    fn test_overall_score_calculation() {
        let config = PeerSelectorConfig::default();
        let selector = PeerSelector::new(config);

        let score = selector.calculate_overall_score(1.0, 1.0, 1.0, 1.0);
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_cache_functionality() {
        let config = PeerSelectorConfig {
            enable_caching: true,
            ..Default::default()
        };
        let selector = PeerSelector::new(config);

        let peer = PeerId::random();
        selector.add_peer_location(peer, GeoLocation::new(40.7128, -74.0060));

        let criteria = SelectionCriteria {
            reference_location: Some(GeoLocation::new(37.7749, -122.4194)),
            min_quality_score: 0.0,
            max_distance_km: None,
            max_results: 10,
        };

        // First call - cache miss
        selector.select_peers(&criteria);
        let stats1 = selector.stats();
        assert_eq!(stats1.cache_misses, 1);

        // Second call - cache hit
        selector.select_peers(&criteria);
        let stats2 = selector.stats();
        assert_eq!(stats2.cache_hits, 1);
    }

    #[test]
    fn test_stats_cache_hit_rate() {
        let stats = PeerSelectorStats {
            cache_hits: 7,
            cache_misses: 3,
            ..Default::default()
        };
        assert!((stats.cache_hit_rate() - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_max_distance_filtering() {
        let config = PeerSelectorConfig::default();
        let selector = PeerSelector::new(config);

        // Add peers at different distances from SF
        let peer_la = PeerId::random();
        let peer_london = PeerId::random();
        selector.add_peer_location(peer_la, GeoLocation::new(34.0522, -118.2437)); // ~550 km
        selector.add_peer_location(peer_london, GeoLocation::new(51.5074, -0.1278)); // ~8600 km

        let criteria = SelectionCriteria {
            reference_location: Some(GeoLocation::new(37.7749, -122.4194)), // SF
            min_quality_score: 0.0,
            max_distance_km: Some(1000.0),
            max_results: 10,
        };

        let selected = selector.select_peers(&criteria);
        assert_eq!(selected.len(), 1); // Only LA should be selected
        assert_eq!(selected[0].peer_id, peer_la);
    }

    #[test]
    fn test_min_quality_filtering() {
        let config = PeerSelectorConfig::default();
        let selector = PeerSelector::new(config);

        let peer = PeerId::random();
        selector.add_peer_location(peer, GeoLocation::new(40.7128, -74.0060));

        // Record poor quality
        selector.update_peer_quality(peer, 1000.0, 0.1, false);

        let criteria = SelectionCriteria {
            reference_location: Some(GeoLocation::new(37.7749, -122.4194)),
            min_quality_score: 0.9, // Very high threshold
            max_distance_km: None,
            max_results: 10,
        };

        let selected = selector.select_peers(&criteria);
        // Peer might be filtered out due to quality
        assert!(selected.is_empty() || selected[0].quality_score >= criteria.min_quality_score);
    }
}
