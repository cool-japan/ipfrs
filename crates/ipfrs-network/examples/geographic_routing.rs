//! Geographic routing optimization example
//!
//! This example demonstrates how to use the geographic routing module
//! for proximity-based peer selection and regional clustering.
//!
//! Run with: cargo run --example geographic_routing

use ipfrs_network::geo_routing::{GeoLocation, GeoRegion, GeoRouter, GeoRouterConfig};
use libp2p::PeerId;

fn main() {
    println!("=== Geographic Routing Optimization Example ===\n");

    // Create different router configurations
    demo_basic_routing();
    println!();
    demo_regional_clustering();
    println!();
    demo_latency_prediction();
    println!();
    demo_configuration_presets();
}

fn demo_basic_routing() {
    println!("--- Basic Geographic Routing ---");

    let config = GeoRouterConfig::default();
    let router = GeoRouter::new(config);

    // Add peers in different cities
    let peer_ny = PeerId::random();
    let peer_la = PeerId::random();
    let peer_london = PeerId::random();
    let peer_tokyo = PeerId::random();
    let peer_sydney = PeerId::random();

    router.update_peer_location(peer_ny, GeoLocation::new(40.7128, -74.0060)); // New York
    router.update_peer_location(peer_la, GeoLocation::new(34.0522, -118.2437)); // Los Angeles
    router.update_peer_location(peer_london, GeoLocation::new(51.5074, -0.1278)); // London
    router.update_peer_location(peer_tokyo, GeoLocation::new(35.6762, 139.6503)); // Tokyo
    router.update_peer_location(peer_sydney, GeoLocation::new(-33.8688, 151.2093)); // Sydney

    println!("Added 5 peers in different global locations");

    // Query from San Francisco
    let sf_location = GeoLocation::new(37.7749, -122.4194);
    println!("\nRanking peers by proximity to San Francisco:");

    let ranked_peers = router.rank_peers_by_proximity(&sf_location);
    for (i, peer) in ranked_peers.iter().enumerate() {
        let distance = peer.distance_km.unwrap_or(0.0);
        let location = router.get_peer_location(&peer.peer_id).unwrap();
        println!(
            "  {}. Distance: {:.0} km, Region: {:?}, Location: ({:.2}, {:.2})",
            i + 1,
            distance,
            peer.region,
            location.latitude,
            location.longitude
        );
    }

    // Get nearby peers (within 5000 km)
    let nearby = router.get_nearby_peers(&sf_location);
    println!("\nPeers within 500 km: {}", nearby.len());
}

fn demo_regional_clustering() {
    println!("--- Regional Clustering ---");

    let config = GeoRouterConfig::default();
    let router = GeoRouter::new(config);

    // Add multiple peers per region
    // North America
    let peers_na = vec![
        (PeerId::random(), GeoLocation::new(40.7128, -74.0060)), // New York
        (PeerId::random(), GeoLocation::new(41.8781, -87.6298)), // Chicago
        (PeerId::random(), GeoLocation::new(29.7604, -95.3698)), // Houston
        (PeerId::random(), GeoLocation::new(33.7490, -84.3880)), // Atlanta
    ];

    // Europe
    let peers_eu = vec![
        (PeerId::random(), GeoLocation::new(51.5074, -0.1278)), // London
        (PeerId::random(), GeoLocation::new(48.8566, 2.3522)),  // Paris
        (PeerId::random(), GeoLocation::new(52.5200, 13.4050)), // Berlin
    ];

    // Asia
    let peers_asia = vec![
        (PeerId::random(), GeoLocation::new(35.6762, 139.6503)), // Tokyo
        (PeerId::random(), GeoLocation::new(37.5665, 126.9780)), // Seoul
        (PeerId::random(), GeoLocation::new(31.2304, 121.4737)), // Shanghai
    ];

    // Add all peers
    for (peer_id, location) in peers_na {
        router.update_peer_location(peer_id, location);
    }
    for (peer_id, location) in peers_eu {
        router.update_peer_location(peer_id, location);
    }
    for (peer_id, location) in peers_asia {
        router.update_peer_location(peer_id, location);
    }

    println!("Added peers across 3 regions");

    // Get peers by region
    let na_peers = router.get_peers_in_region(GeoRegion::NorthAmerica);
    let eu_peers = router.get_peers_in_region(GeoRegion::Europe);
    let asia_peers = router.get_peers_in_region(GeoRegion::Asia);

    println!("\nPeers per region:");
    println!("  North America: {}", na_peers.len());
    println!("  Europe: {}", eu_peers.len());
    println!("  Asia: {}", asia_peers.len());

    // Show statistics
    let stats = router.stats();
    println!("\nRouter statistics:");
    println!("  Total peers tracked: {}", stats.total_peers);
    println!("  Proximity queries: {}", stats.proximity_queries);
    println!("  Region lookups: {}", stats.region_lookups);
}

fn demo_latency_prediction() {
    println!("--- Latency Prediction ---");

    // Define some major cities
    let cities = vec![
        ("New York", GeoLocation::new(40.7128, -74.0060)),
        ("London", GeoLocation::new(51.5074, -0.1278)),
        ("Tokyo", GeoLocation::new(35.6762, 139.6503)),
        ("Sydney", GeoLocation::new(-33.8688, 151.2093)),
        ("São Paulo", GeoLocation::new(-23.5505, -46.6333)),
    ];

    let reference = GeoLocation::new(37.7749, -122.4194); // San Francisco
    println!("Estimated latencies from San Francisco:\n");

    for (name, location) in cities {
        let distance = reference.distance_to(&location);
        let latency = reference.estimate_latency_ms(&location);
        println!(
            "  {} ({:.2}, {:.2}):",
            name, location.latitude, location.longitude
        );
        println!("    Distance: {:.0} km", distance);
        println!("    Estimated latency: {:.1} ms", latency);
        println!();
    }
}

fn demo_configuration_presets() {
    println!("--- Configuration Presets ---");

    // Low-latency config
    let low_latency = GeoRouterConfig::low_latency();
    println!("Low-latency configuration:");
    println!("  Nearby threshold: {} km", low_latency.nearby_threshold_km);
    println!(
        "  Same-region bonus: {} km",
        low_latency.same_region_bonus_km
    );
    println!(
        "  Max peers per region: {}",
        low_latency.max_peers_per_region
    );

    // Global config
    let global = GeoRouterConfig::global();
    println!("\nGlobal distribution configuration:");
    println!("  Nearby threshold: {} km", global.nearby_threshold_km);
    println!("  Same-region bonus: {} km", global.same_region_bonus_km);
    println!("  Max peers per region: {}", global.max_peers_per_region);

    // Regional config
    let regional = GeoRouterConfig::regional();
    println!("\nRegional focus configuration:");
    println!("  Nearby threshold: {} km", regional.nearby_threshold_km);
    println!("  Same-region bonus: {} km", regional.same_region_bonus_km);
    println!("  Max peers per region: {}", regional.max_peers_per_region);

    println!("\n--- Same-Region Bonus Demonstration ---");

    // Create router with same-region bonus
    let config = GeoRouterConfig {
        same_region_bonus_km: 2000.0,
        enable_region_clustering: true,
        ..Default::default()
    };
    let router = GeoRouter::new(config);

    // Add two peers: one nearby in different region, one farther in same region
    let peer_toronto = PeerId::random();
    let peer_london = PeerId::random();

    router.update_peer_location(peer_toronto, GeoLocation::new(43.6532, -79.3832)); // Toronto (NA)
    router.update_peer_location(peer_london, GeoLocation::new(51.5074, -0.1278)); // London (EU)

    let ny = GeoLocation::new(40.7128, -74.0060); // New York (NA)
    println!("\nFrom New York:");
    println!(
        "  Toronto distance: {:.0} km (same region: North America)",
        ny.distance_to(&GeoLocation::new(43.6532, -79.3832))
    );
    println!(
        "  London distance: {:.0} km (different region: Europe)",
        ny.distance_to(&GeoLocation::new(51.5074, -0.1278))
    );

    let ranked = router.rank_peers_by_proximity(&ny);
    println!("\nWith same-region bonus of 2000 km:");
    for (i, peer) in ranked.iter().enumerate() {
        println!(
            "  {}. Effective distance: {:.0} km, Region: {:?}",
            i + 1,
            peer.distance_km.unwrap_or(0.0),
            peer.region
        );
    }
    println!("  Toronto ranks higher due to same-region bonus!");
}
