//! Geographic routing optimization for IPFRS network
//!
//! This module provides geographic-aware peer selection and routing to optimize
//! network latency by preferring geographically closer peers.
//!
//! ## Features
//!
//! - **GeoIP Lookup**: Determine peer geographic location from IP addresses
//! - **Distance Calculation**: Great-circle distance using Haversine formula
//! - **Proximity Ranking**: Rank peers by geographic proximity
//! - **Regional Clustering**: Group peers by geographic regions
//! - **Latency Prediction**: Estimate latency based on distance
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_network::geo_routing::{GeoRouter, GeoLocation, GeoRouterConfig};
//! use std::net::IpAddr;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = GeoRouterConfig::default();
//! let mut router = GeoRouter::new(config);
//!
//! // Add peer locations
//! let peer1 = libp2p::PeerId::random();
//! let peer2 = libp2p::PeerId::random();
//!
//! router.update_peer_location(peer1, GeoLocation::new(40.7128, -74.0060)); // New York
//! router.update_peer_location(peer2, GeoLocation::new(51.5074, -0.1278));   // London
//!
//! // Get proximity-ranked peers from San Francisco
//! let sf_location = GeoLocation::new(37.7749, -122.4194);
//! let ranked = router.rank_peers_by_proximity(&sf_location);
//!
//! println!("Closest peer: {:?}", ranked.first());
//! # Ok(())
//! # }
//! ```

use dashmap::DashMap;
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;

/// Geographic location with latitude and longitude
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GeoLocation {
    /// Latitude in degrees (-90 to 90)
    pub latitude: f64,
    /// Longitude in degrees (-180 to 180)
    pub longitude: f64,
}

impl GeoLocation {
    /// Create a new geographic location
    ///
    /// # Arguments
    ///
    /// * `latitude` - Latitude in degrees (-90 to 90)
    /// * `longitude` - Longitude in degrees (-180 to 180)
    ///
    /// # Panics
    ///
    /// Panics if latitude or longitude are out of valid ranges
    pub fn new(latitude: f64, longitude: f64) -> Self {
        assert!(
            (-90.0..=90.0).contains(&latitude),
            "Latitude must be between -90 and 90"
        );
        assert!(
            (-180.0..=180.0).contains(&longitude),
            "Longitude must be between -180 and 180"
        );
        Self {
            latitude,
            longitude,
        }
    }

    /// Calculate great-circle distance to another location in kilometers
    /// using the Haversine formula
    pub fn distance_to(&self, other: &GeoLocation) -> f64 {
        const EARTH_RADIUS_KM: f64 = 6371.0;

        let lat1 = self.latitude.to_radians();
        let lat2 = other.latitude.to_radians();
        let delta_lat = (other.latitude - self.latitude).to_radians();
        let delta_lon = (other.longitude - self.longitude).to_radians();

        let a = (delta_lat / 2.0).sin().powi(2)
            + lat1.cos() * lat2.cos() * (delta_lon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

        EARTH_RADIUS_KM * c
    }

    /// Estimate network latency in milliseconds based on distance
    ///
    /// Uses a simple model: base_latency + (distance_km / speed_of_light_factor)
    /// Speed of light in fiber is roughly 200,000 km/s, giving ~5ms per 1000km
    pub fn estimate_latency_ms(&self, other: &GeoLocation) -> f64 {
        const BASE_LATENCY_MS: f64 = 10.0; // Base latency for processing
        const MS_PER_1000_KM: f64 = 5.0; // Network propagation delay

        let distance = self.distance_to(other);
        BASE_LATENCY_MS + (distance / 1000.0) * MS_PER_1000_KM
    }
}

/// Geographic region for clustering peers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GeoRegion {
    /// North America
    NorthAmerica,
    /// South America
    SouthAmerica,
    /// Europe
    Europe,
    /// Asia
    Asia,
    /// Africa
    Africa,
    /// Oceania (Australia, Pacific Islands)
    Oceania,
    /// Unknown region
    Unknown,
}

impl GeoRegion {
    /// Determine region from geographic location
    pub fn from_location(location: &GeoLocation) -> Self {
        let lat = location.latitude;
        let lon = location.longitude;

        // Simple region classification based on lat/lon ranges
        if (-170.0..=-30.0).contains(&lon) {
            // Americas
            if lat >= 15.0 {
                GeoRegion::NorthAmerica
            } else {
                GeoRegion::SouthAmerica
            }
        } else if (-30.0..=60.0).contains(&lon) {
            // Europe and Africa
            if lat >= 35.0 {
                GeoRegion::Europe
            } else {
                GeoRegion::Africa
            }
        } else if (60.0..=150.0).contains(&lon) {
            // Asia and Oceania
            if lat >= -10.0 {
                GeoRegion::Asia
            } else {
                GeoRegion::Oceania
            }
        } else {
            GeoRegion::Unknown
        }
    }

    /// Get representative location for region (approximate center)
    #[allow(dead_code)]
    pub fn representative_location(&self) -> GeoLocation {
        match self {
            GeoRegion::NorthAmerica => GeoLocation::new(39.8283, -98.5795), // US center
            GeoRegion::SouthAmerica => GeoLocation::new(-14.2350, -51.9253), // Brazil
            GeoRegion::Europe => GeoLocation::new(50.0, 10.0),              // Germany
            GeoRegion::Asia => GeoLocation::new(34.0, 100.0),               // Central China
            GeoRegion::Africa => GeoLocation::new(1.0, 18.0),               // Central Africa
            GeoRegion::Oceania => GeoLocation::new(-25.2744, 133.7751),     // Australia
            GeoRegion::Unknown => GeoLocation::new(0.0, 0.0),
        }
    }
}

/// Peer with geographic metadata
#[derive(Debug, Clone)]
pub struct GeoPeer {
    /// Peer ID
    pub peer_id: PeerId,
    /// Geographic location
    pub location: GeoLocation,
    /// Region
    pub region: GeoRegion,
    /// Distance from reference point (if applicable)
    pub distance_km: Option<f64>,
}

impl GeoPeer {
    /// Create a new GeoPeer
    pub fn new(peer_id: PeerId, location: GeoLocation) -> Self {
        let region = GeoRegion::from_location(&location);
        Self {
            peer_id,
            location,
            region,
            distance_km: None,
        }
    }

    /// Set distance from a reference point
    pub fn with_distance(mut self, distance_km: f64) -> Self {
        self.distance_km = Some(distance_km);
        self
    }
}

/// Configuration for geographic router
#[derive(Debug, Clone)]
pub struct GeoRouterConfig {
    /// Maximum distance in km to consider peers as "nearby"
    pub nearby_threshold_km: f64,
    /// Prefer peers in same region with this bonus (subtracted from distance)
    pub same_region_bonus_km: f64,
    /// Enable region-based clustering
    pub enable_region_clustering: bool,
    /// Maximum number of peers to track per region
    pub max_peers_per_region: usize,
}

impl Default for GeoRouterConfig {
    fn default() -> Self {
        Self {
            nearby_threshold_km: 500.0,   // 500km radius for "nearby"
            same_region_bonus_km: 1000.0, // Same region gets 1000km bonus
            enable_region_clustering: true,
            max_peers_per_region: 100,
        }
    }
}

impl GeoRouterConfig {
    /// Configuration optimized for low latency
    pub fn low_latency() -> Self {
        Self {
            nearby_threshold_km: 1000.0,
            same_region_bonus_km: 2000.0,
            enable_region_clustering: true,
            max_peers_per_region: 50,
        }
    }

    /// Configuration optimized for global distribution
    pub fn global() -> Self {
        Self {
            nearby_threshold_km: 5000.0,
            same_region_bonus_km: 500.0,
            enable_region_clustering: true,
            max_peers_per_region: 200,
        }
    }

    /// Configuration for regional focus
    pub fn regional() -> Self {
        Self {
            nearby_threshold_km: 200.0,
            same_region_bonus_km: 3000.0,
            enable_region_clustering: true,
            max_peers_per_region: 150,
        }
    }
}

/// Geographic router for proximity-based peer selection
pub struct GeoRouter {
    /// Configuration
    config: GeoRouterConfig,
    /// Peer locations
    peer_locations: Arc<DashMap<PeerId, GeoLocation>>,
    /// Peers grouped by region
    region_peers: Arc<DashMap<GeoRegion, Vec<PeerId>>>,
    /// Statistics
    stats: Arc<parking_lot::RwLock<GeoRouterStats>>,
}

/// Statistics for geographic router
#[derive(Debug, Clone, Default)]
pub struct GeoRouterStats {
    /// Total peers tracked
    pub total_peers: usize,
    /// Peers per region
    pub peers_per_region: HashMap<GeoRegion, usize>,
    /// Total proximity queries
    pub proximity_queries: u64,
    /// Total region lookups
    pub region_lookups: u64,
}

impl GeoRouter {
    /// Create a new geographic router
    pub fn new(config: GeoRouterConfig) -> Self {
        Self {
            config,
            peer_locations: Arc::new(DashMap::new()),
            region_peers: Arc::new(DashMap::new()),
            stats: Arc::new(parking_lot::RwLock::new(GeoRouterStats::default())),
        }
    }

    /// Update or add a peer's location
    pub fn update_peer_location(&self, peer_id: PeerId, location: GeoLocation) {
        // Remove from old region if exists
        if let Some(old_location) = self.peer_locations.get(&peer_id) {
            let old_region = GeoRegion::from_location(&old_location);
            if let Some(mut peers) = self.region_peers.get_mut(&old_region) {
                peers.retain(|p| p != &peer_id);
            }
        }

        // Update location
        self.peer_locations.insert(peer_id, location);

        // Add to new region
        if self.config.enable_region_clustering {
            let region = GeoRegion::from_location(&location);
            self.region_peers.entry(region).or_default().push(peer_id);

            // Enforce max peers per region
            if let Some(mut peers) = self.region_peers.get_mut(&region) {
                if peers.len() > self.config.max_peers_per_region {
                    peers.truncate(self.config.max_peers_per_region);
                }
            }
        }

        // Update stats
        let mut stats = self.stats.write();
        stats.total_peers = self.peer_locations.len();
        stats.peers_per_region.clear();
        for entry in self.region_peers.iter() {
            stats
                .peers_per_region
                .insert(*entry.key(), entry.value().len());
        }
    }

    /// Remove a peer's location
    pub fn remove_peer(&self, peer_id: &PeerId) {
        if let Some((_, location)) = self.peer_locations.remove(peer_id) {
            let region = GeoRegion::from_location(&location);
            if let Some(mut peers) = self.region_peers.get_mut(&region) {
                peers.retain(|p| p != peer_id);
            }

            // Update stats
            let mut stats = self.stats.write();
            stats.total_peers = self.peer_locations.len();
            stats.peers_per_region.insert(
                region,
                self.region_peers.get(&region).map(|p| p.len()).unwrap_or(0),
            );
        }
    }

    /// Get location for a peer
    pub fn get_peer_location(&self, peer_id: &PeerId) -> Option<GeoLocation> {
        self.peer_locations.get(peer_id).map(|loc| *loc)
    }

    /// Get all peers in a region
    pub fn get_peers_in_region(&self, region: GeoRegion) -> Vec<PeerId> {
        self.stats.write().region_lookups += 1;
        self.region_peers
            .get(&region)
            .map(|peers| peers.clone())
            .unwrap_or_default()
    }

    /// Rank all known peers by proximity to a location
    pub fn rank_peers_by_proximity(&self, reference: &GeoLocation) -> Vec<GeoPeer> {
        self.stats.write().proximity_queries += 1;

        let mut peers: Vec<GeoPeer> = self
            .peer_locations
            .iter()
            .map(|entry| {
                let peer_id = *entry.key();
                let location = *entry.value();
                let distance = reference.distance_to(&location);
                GeoPeer::new(peer_id, location).with_distance(distance)
            })
            .collect();

        // Apply same-region bonus
        if self.config.enable_region_clustering {
            let ref_region = GeoRegion::from_location(reference);
            for peer in &mut peers {
                if peer.region == ref_region {
                    if let Some(dist) = peer.distance_km.as_mut() {
                        *dist = (*dist - self.config.same_region_bonus_km).max(0.0);
                    }
                }
            }
        }

        // Sort by distance
        peers.sort_by(|a, b| {
            a.distance_km
                .unwrap_or(f64::INFINITY)
                .partial_cmp(&b.distance_km.unwrap_or(f64::INFINITY))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        peers
    }

    /// Get nearby peers within threshold distance
    pub fn get_nearby_peers(&self, reference: &GeoLocation) -> Vec<GeoPeer> {
        let all_peers = self.rank_peers_by_proximity(reference);
        all_peers
            .into_iter()
            .filter(|peer| {
                peer.distance_km
                    .map(|d| d <= self.config.nearby_threshold_km)
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Estimate IP location using simple heuristics (placeholder)
    ///
    /// In production, this would use a GeoIP database like MaxMind GeoLite2
    #[allow(dead_code)]
    pub fn estimate_ip_location(&self, _ip: IpAddr) -> Option<GeoLocation> {
        // Placeholder: In production, integrate with GeoIP database
        // For now, return None to indicate unknown location
        None
    }

    /// Get statistics
    pub fn stats(&self) -> GeoRouterStats {
        self.stats.read().clone()
    }

    /// Clear all peer locations
    #[allow(dead_code)]
    pub fn clear(&self) {
        self.peer_locations.clear();
        self.region_peers.clear();
        let mut stats = self.stats.write();
        stats.total_peers = 0;
        stats.peers_per_region.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geo_location_new() {
        let loc = GeoLocation::new(40.7128, -74.0060);
        assert_eq!(loc.latitude, 40.7128);
        assert_eq!(loc.longitude, -74.0060);
    }

    #[test]
    #[should_panic(expected = "Latitude must be between -90 and 90")]
    fn test_geo_location_invalid_latitude() {
        GeoLocation::new(100.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "Longitude must be between -180 and 180")]
    fn test_geo_location_invalid_longitude() {
        GeoLocation::new(0.0, 200.0);
    }

    #[test]
    fn test_distance_calculation() {
        let ny = GeoLocation::new(40.7128, -74.0060);
        let london = GeoLocation::new(51.5074, -0.1278);

        let distance = ny.distance_to(&london);
        // NY to London is approximately 5570 km
        assert!((distance - 5570.0).abs() < 100.0, "Distance: {}", distance);
    }

    #[test]
    fn test_distance_same_location() {
        let loc = GeoLocation::new(40.7128, -74.0060);
        let distance = loc.distance_to(&loc);
        assert!(distance < 0.1, "Same location distance should be ~0");
    }

    #[test]
    fn test_latency_estimation() {
        let sf = GeoLocation::new(37.7749, -122.4194);
        let tokyo = GeoLocation::new(35.6762, 139.6503);

        let latency = sf.estimate_latency_ms(&tokyo);
        // SF to Tokyo is ~8300 km, should be ~50-60ms
        assert!(latency > 40.0 && latency < 70.0, "Latency: {}", latency);
    }

    #[test]
    fn test_region_from_location() {
        // Test North America
        let ny = GeoLocation::new(40.7128, -74.0060);
        assert_eq!(GeoRegion::from_location(&ny), GeoRegion::NorthAmerica);

        // Test Europe
        let london = GeoLocation::new(51.5074, -0.1278);
        assert_eq!(GeoRegion::from_location(&london), GeoRegion::Europe);

        // Test Asia
        let tokyo = GeoLocation::new(35.6762, 139.6503);
        assert_eq!(GeoRegion::from_location(&tokyo), GeoRegion::Asia);

        // Test South America
        let sao_paulo = GeoLocation::new(-23.5505, -46.6333);
        assert_eq!(
            GeoRegion::from_location(&sao_paulo),
            GeoRegion::SouthAmerica
        );
    }

    #[test]
    fn test_geo_router_basic() {
        let config = GeoRouterConfig::default();
        let router = GeoRouter::new(config);

        let peer1 = PeerId::random();
        let peer2 = PeerId::random();

        router.update_peer_location(peer1, GeoLocation::new(40.7128, -74.0060));
        router.update_peer_location(peer2, GeoLocation::new(51.5074, -0.1278));

        assert_eq!(router.stats().total_peers, 2);
    }

    #[test]
    fn test_proximity_ranking() {
        let config = GeoRouterConfig::default();
        let router = GeoRouter::new(config);

        let peer_ny = PeerId::random();
        let peer_la = PeerId::random();
        let peer_london = PeerId::random();

        router.update_peer_location(peer_ny, GeoLocation::new(40.7128, -74.0060));
        router.update_peer_location(peer_la, GeoLocation::new(34.0522, -118.2437));
        router.update_peer_location(peer_london, GeoLocation::new(51.5074, -0.1278));

        // From SF, LA should be closest, then NY, then London
        let sf = GeoLocation::new(37.7749, -122.4194);
        let ranked = router.rank_peers_by_proximity(&sf);

        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].peer_id, peer_la);
        assert_eq!(ranked[1].peer_id, peer_ny);
        assert_eq!(ranked[2].peer_id, peer_london);
    }

    #[test]
    fn test_nearby_peers() {
        let config = GeoRouterConfig {
            nearby_threshold_km: 500.0,
            ..Default::default()
        };
        let router = GeoRouter::new(config);

        let peer_nearby = PeerId::random();
        let peer_far = PeerId::random();

        // NY and Philadelphia are ~130 km apart
        router.update_peer_location(peer_nearby, GeoLocation::new(39.9526, -75.1652)); // Philly
                                                                                       // London is far from NY
        router.update_peer_location(peer_far, GeoLocation::new(51.5074, -0.1278));

        let ny = GeoLocation::new(40.7128, -74.0060);
        let nearby = router.get_nearby_peers(&ny);

        assert_eq!(nearby.len(), 1);
        assert_eq!(nearby[0].peer_id, peer_nearby);
    }

    #[test]
    fn test_region_clustering() {
        let config = GeoRouterConfig::default();
        let router = GeoRouter::new(config);

        let peer1 = PeerId::random();
        let peer2 = PeerId::random();
        let peer3 = PeerId::random();

        router.update_peer_location(peer1, GeoLocation::new(40.7128, -74.0060)); // NY
        router.update_peer_location(peer2, GeoLocation::new(34.0522, -118.2437)); // LA
        router.update_peer_location(peer3, GeoLocation::new(51.5074, -0.1278)); // London

        let na_peers = router.get_peers_in_region(GeoRegion::NorthAmerica);
        let eu_peers = router.get_peers_in_region(GeoRegion::Europe);

        assert_eq!(na_peers.len(), 2);
        assert_eq!(eu_peers.len(), 1);
        assert!(na_peers.contains(&peer1));
        assert!(na_peers.contains(&peer2));
        assert!(eu_peers.contains(&peer3));
    }

    #[test]
    fn test_remove_peer() {
        let config = GeoRouterConfig::default();
        let router = GeoRouter::new(config);

        let peer = PeerId::random();
        router.update_peer_location(peer, GeoLocation::new(40.7128, -74.0060));
        assert_eq!(router.stats().total_peers, 1);

        router.remove_peer(&peer);
        assert_eq!(router.stats().total_peers, 0);
        assert!(router.get_peer_location(&peer).is_none());
    }

    #[test]
    fn test_same_region_bonus() {
        let config = GeoRouterConfig {
            same_region_bonus_km: 1000.0,
            enable_region_clustering: true,
            ..Default::default()
        };
        let router = GeoRouter::new(config);

        let peer_same_region = PeerId::random();
        let peer_diff_region = PeerId::random();

        // Both are similar distance from NY, but one is in NA, one in Europe
        router.update_peer_location(peer_same_region, GeoLocation::new(34.0522, -118.2437)); // LA
        router.update_peer_location(peer_diff_region, GeoLocation::new(51.5074, -0.1278)); // London

        let ny = GeoLocation::new(40.7128, -74.0060);
        let ranked = router.rank_peers_by_proximity(&ny);

        // LA should rank first due to same-region bonus, even though distance is similar
        assert_eq!(ranked[0].peer_id, peer_same_region);
    }

    #[test]
    fn test_max_peers_per_region() {
        let config = GeoRouterConfig {
            max_peers_per_region: 2,
            enable_region_clustering: true,
            ..Default::default()
        };
        let router = GeoRouter::new(config);

        // Add 3 peers in North America
        for _ in 0..3 {
            let peer = PeerId::random();
            router.update_peer_location(peer, GeoLocation::new(40.0, -100.0));
        }

        let na_peers = router.get_peers_in_region(GeoRegion::NorthAmerica);
        assert!(na_peers.len() <= 2, "Should enforce max peers per region");
    }

    #[test]
    fn test_statistics_tracking() {
        let config = GeoRouterConfig::default();
        let router = GeoRouter::new(config);

        let peer1 = PeerId::random();
        let peer2 = PeerId::random();

        router.update_peer_location(peer1, GeoLocation::new(40.7128, -74.0060));
        router.update_peer_location(peer2, GeoLocation::new(51.5074, -0.1278));

        let stats = router.stats();
        assert_eq!(stats.total_peers, 2);
        assert_eq!(stats.proximity_queries, 0);

        router.rank_peers_by_proximity(&GeoLocation::new(0.0, 0.0));
        let stats = router.stats();
        assert_eq!(stats.proximity_queries, 1);
    }

    #[test]
    fn test_config_presets() {
        let low_latency = GeoRouterConfig::low_latency();
        assert_eq!(low_latency.nearby_threshold_km, 1000.0);

        let global = GeoRouterConfig::global();
        assert_eq!(global.nearby_threshold_km, 5000.0);

        let regional = GeoRouterConfig::regional();
        assert_eq!(regional.nearby_threshold_km, 200.0);
    }
}
