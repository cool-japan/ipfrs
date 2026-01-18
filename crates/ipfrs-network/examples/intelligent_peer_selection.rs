//! Intelligent peer selection example
//!
//! This example demonstrates how to use the PeerSelector to intelligently
//! choose peers based on geographic proximity and connection quality.
//!
//! Run with: cargo run --example intelligent_peer_selection

use ipfrs_network::geo_routing::GeoLocation;
use ipfrs_network::peer_selector::{PeerSelector, PeerSelectorConfig, SelectionCriteria};
use libp2p::PeerId;

fn main() {
    println!("=== Intelligent Peer Selection Example ===\n");

    demo_basic_selection();
    println!();
    demo_configuration_presets();
    println!();
    demo_quality_based_selection();
    println!();
    demo_distance_filtering();
    println!();
    demo_caching();
}

fn demo_basic_selection() {
    println!("--- Basic Peer Selection ---");

    let config = PeerSelectorConfig::balanced();
    let selector = PeerSelector::new(config);

    // Add peers in different cities with realistic locations
    let peers = vec![
        (
            PeerId::random(),
            "New York",
            GeoLocation::new(40.7128, -74.0060),
        ),
        (
            PeerId::random(),
            "Los Angeles",
            GeoLocation::new(34.0522, -118.2437),
        ),
        (
            PeerId::random(),
            "London",
            GeoLocation::new(51.5074, -0.1278),
        ),
        (
            PeerId::random(),
            "Tokyo",
            GeoLocation::new(35.6762, 139.6503),
        ),
        (
            PeerId::random(),
            "Sydney",
            GeoLocation::new(-33.8688, 151.2093),
        ),
    ];

    println!(
        "Added {} peers from different global locations",
        peers.len()
    );
    for (peer_id, name, location) in &peers {
        selector.add_peer_location(*peer_id, *location);
        println!(
            "  - {} at ({:.2}, {:.2})",
            name, location.latitude, location.longitude
        );
    }

    // Select best peers from San Francisco
    let sf_location = GeoLocation::new(37.7749, -122.4194);
    println!("\nSelecting best 3 peers from San Francisco:");

    let criteria = SelectionCriteria {
        reference_location: Some(sf_location),
        min_quality_score: 0.0,
        max_distance_km: None,
        max_results: 3,
    };

    let selected = selector.select_peers(&criteria);
    println!(
        "\nSelected {} peers (ranked by overall score):",
        selected.len()
    );
    for (i, peer) in selected.iter().enumerate() {
        println!("\n{}. Peer:", i + 1);
        println!("   Overall Score: {:.3}", peer.score);
        println!("   Distance Score: {:.3}", peer.distance_score);
        println!("   Quality Score: {:.3}", peer.quality_score);
        if let Some(dist) = peer.distance_km {
            println!("   Distance: {:.0} km", dist);
        }

        // Find peer name for display
        for (pid, name, _) in &peers {
            if pid == &peer.peer_id {
                println!("   Location: {}", name);
                break;
            }
        }
    }
}

fn demo_configuration_presets() {
    println!("--- Configuration Presets ---");

    let configs = vec![
        ("Low Latency", PeerSelectorConfig::low_latency()),
        ("High Bandwidth", PeerSelectorConfig::high_bandwidth()),
        ("Balanced", PeerSelectorConfig::balanced()),
        ("Mobile", PeerSelectorConfig::mobile()),
    ];

    for (name, config) in configs {
        println!("\n{}:", name);
        println!("  Distance weight: {:.2}", config.distance_weight);
        println!("  Quality weight: {:.2}", config.quality_weight);
        println!("  Latency weight: {:.2}", config.latency_weight);
        println!("  Bandwidth weight: {:.2}", config.bandwidth_weight);
        println!("  Cache TTL: {}s", config.cache_ttl_secs);
    }
}

fn demo_quality_based_selection() {
    println!("--- Quality-Based Selection ---");

    let config = PeerSelectorConfig::balanced();
    let selector = PeerSelector::new(config);

    // Add peers with different quality characteristics
    let good_peer = PeerId::random();
    let bad_peer = PeerId::random();

    // Both in similar locations (New York area)
    selector.add_peer_location(good_peer, GeoLocation::new(40.7128, -74.0060));
    selector.add_peer_location(bad_peer, GeoLocation::new(40.7589, -73.9851));

    // Update quality metrics
    println!("\nSimulating connection history:");
    println!("  Good peer: Low latency (20ms), high bandwidth (100 Mbps), successful");
    selector.update_peer_quality(good_peer, 20.0, 100.0, true);

    println!("  Bad peer: High latency (500ms), low bandwidth (1 Mbps), failed");
    selector.update_peer_quality(bad_peer, 500.0, 1.0, false);

    // Select from nearby location
    let criteria = SelectionCriteria {
        reference_location: Some(GeoLocation::new(40.7306, -73.9352)), // Brooklyn
        min_quality_score: 0.0,
        max_distance_km: None,
        max_results: 10,
    };

    let selected = selector.select_peers(&criteria);
    println!("\nSelected peers (should rank good peer higher):");
    for (i, peer) in selected.iter().enumerate() {
        let peer_type = if peer.peer_id == good_peer {
            "Good"
        } else {
            "Bad"
        };
        println!(
            "  {}. {} Peer - Score: {:.3} (Quality: {:.3}, Latency: {:.3})",
            i + 1,
            peer_type,
            peer.score,
            peer.quality_score,
            peer.latency_score
        );
    }
}

fn demo_distance_filtering() {
    println!("--- Distance Filtering ---");

    let config = PeerSelectorConfig::balanced();
    let selector = PeerSelector::new(config);

    // Add peers at various distances from New York
    let locations = vec![
        ("Philadelphia", GeoLocation::new(39.9526, -75.1652)), // ~130 km
        ("Washington DC", GeoLocation::new(38.9072, -77.0369)), // ~330 km
        ("Chicago", GeoLocation::new(41.8781, -87.6298)),      // ~1150 km
        ("Denver", GeoLocation::new(39.7392, -104.9903)),      // ~2600 km
        ("Los Angeles", GeoLocation::new(34.0522, -118.2437)), // ~3940 km
    ];

    for (name, location) in &locations {
        selector.add_peer_location(PeerId::random(), *location);
        let ny = GeoLocation::new(40.7128, -74.0060);
        let dist = ny.distance_to(location);
        println!("  {} - {:.0} km from New York", name, dist);
    }

    // Select only peers within 1000 km
    println!("\nSelecting peers within 1000 km of New York:");
    let criteria = SelectionCriteria {
        reference_location: Some(GeoLocation::new(40.7128, -74.0060)),
        min_quality_score: 0.0,
        max_distance_km: Some(1000.0),
        max_results: 10,
    };

    let selected = selector.select_peers(&criteria);
    println!("Found {} peers within range:", selected.len());
    for peer in &selected {
        if let Some(dist) = peer.distance_km {
            println!("  - Distance: {:.0} km, Score: {:.3}", dist, peer.score);
        }
    }
}

fn demo_caching() {
    println!("--- Selection Caching ---");

    let config = PeerSelectorConfig {
        enable_caching: true,
        cache_ttl_secs: 300,
        ..PeerSelectorConfig::default()
    };
    let selector = PeerSelector::new(config);

    // Add some peers
    for _ in 0..5 {
        selector.add_peer_location(PeerId::random(), GeoLocation::new(40.7128, -74.0060));
    }

    let criteria = SelectionCriteria {
        reference_location: Some(GeoLocation::new(37.7749, -122.4194)),
        min_quality_score: 0.0,
        max_distance_km: None,
        max_results: 3,
    };

    println!("First selection (cache miss):");
    selector.select_peers(&criteria);
    let stats1 = selector.stats();
    println!("  Total selections: {}", stats1.total_selections);
    println!("  Cache hits: {}", stats1.cache_hits);
    println!("  Cache misses: {}", stats1.cache_misses);

    println!("\nSecond selection with same criteria (cache hit):");
    selector.select_peers(&criteria);
    let stats2 = selector.stats();
    println!("  Total selections: {}", stats2.total_selections);
    println!("  Cache hits: {}", stats2.cache_hits);
    println!("  Cache misses: {}", stats2.cache_misses);
    println!("  Cache hit rate: {:.1}%", stats2.cache_hit_rate() * 100.0);

    println!("\nPerformance metrics:");
    println!(
        "  Average selection time: {:.2} μs",
        stats2.avg_selection_time_us
    );
    println!("  Total peers evaluated: {}", stats2.total_peers_evaluated);
}
