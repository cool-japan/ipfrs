//! Custom DHT provider example
//!
//! This example demonstrates how to use the pluggable DHT provider interface
//! to support custom DHT implementations.
//!
//! Run with: cargo run --example custom_dht

use cid::Cid;
use ipfrs_network::dht_provider::kademlia::KademliaDhtProvider;
use ipfrs_network::dht_provider::{
    DhtCapabilities, DhtPeerInfo, DhtProvider, DhtProviderError, DhtProviderRegistry,
    DhtProviderStats, DhtQueryResult,
};
use libp2p::PeerId;
use std::sync::Arc;

fn main() {
    println!("=== Custom DHT Provider Example ===\n");

    demo_provider_registry();
    println!();
    demo_kademlia_provider();
    println!();
    demo_custom_provider();
    println!();
    demo_provider_capabilities();
}

fn demo_provider_registry() {
    println!("--- DHT Provider Registry ---");

    let mut registry = DhtProviderRegistry::new();

    // Register Kademlia provider
    let kademlia = Arc::new(KademliaDhtProvider::new());
    registry.register("kademlia", kademlia.clone());

    // Register a custom provider
    let custom = Arc::new(SimpleDhtProvider::new("simple-dht"));
    registry.register("simple", custom);

    println!("Registered providers: {:?}", registry.list_providers());
    println!("Number of providers: {}", registry.count());

    // Get active provider
    if let Some(active) = registry.get_active() {
        println!("\nActive provider: {}", active.name());
        println!("Version: {}", active.version());
        let caps = active.capabilities();
        println!(
            "Supports content routing: {}",
            caps.supports_content_routing
        );
        println!("Supports peer routing: {}", caps.supports_peer_routing);
    }

    // Switch active provider
    println!("\nSwitching to 'simple' provider...");
    registry.set_active("simple").unwrap();
    if let Some(active) = registry.get_active() {
        println!("New active provider: {}", active.name());
    }
}

fn demo_kademlia_provider() {
    println!("--- Kademlia DHT Provider ---");

    let provider = KademliaDhtProvider::new();
    println!("Provider: {}", provider.name());
    println!("Version: {}", provider.version());

    let caps = provider.capabilities();
    println!("\nCapabilities:");
    println!("  Content routing: {}", caps.supports_content_routing);
    println!("  Peer routing: {}", caps.supports_peer_routing);
    println!("  KV storage: {}", caps.supports_kv_storage);
    println!("  Range queries: {}", caps.supports_range_queries);
    println!("  Max query hops: {:?}", caps.max_query_hops);

    // Bootstrap with peers
    let bootstrap_peers = vec![PeerId::random(), PeerId::random(), PeerId::random()];
    println!("\nBootstrapping with {} peers...", bootstrap_peers.len());
    provider.bootstrap(bootstrap_peers).unwrap();

    // Check health
    println!("DHT healthy: {}", provider.is_healthy());

    // Get statistics
    let stats = provider.stats();
    println!("\nStatistics:");
    println!("  Routing table size: {}", stats.routing_table_size);
    println!("  Total queries: {}", stats.total_queries);
    println!("  Success rate: {:.2}", stats.success_rate);
}

fn demo_custom_provider() {
    println!("--- Custom DHT Provider ---");

    let provider = SimpleDhtProvider::new("my-custom-dht");
    println!("Provider: {}", provider.name());
    println!("Version: {}", provider.version());

    // Bootstrap
    provider.bootstrap(vec![PeerId::random()]).unwrap();

    // Provide content
    let cid = Cid::default();
    println!("\nAnnouncing content: {:?}", cid);
    provider.provide(&cid).unwrap();

    // Find providers
    println!("Finding providers for content...");
    match provider.find_providers(&cid) {
        Ok(result) => {
            println!("  Found {} providers", result.providers.len());
            println!("  Hops: {}", result.hops);
            println!("  Duration: {} ms", result.duration_ms);
            println!("  Success: {}", result.success);
        }
        Err(e) => println!("  Error: {}", e),
    }

    // Get statistics
    let stats = provider.stats();
    println!("\nStatistics:");
    println!("  Total queries: {}", stats.total_queries);
    println!("  Successful: {}", stats.successful_queries);
    println!("  Failed: {}", stats.failed_queries);
    println!("  Success rate: {:.2}", stats.success_rate);
}

fn demo_provider_capabilities() {
    println!("--- Provider Capabilities ---");

    println!("Basic DHT capabilities:");
    let basic = DhtCapabilities::basic();
    print_capabilities(&basic);

    println!("\nAdvanced DHT capabilities:");
    let advanced = DhtCapabilities::advanced();
    print_capabilities(&advanced);

    println!("\n--- Capability Comparison ---");
    let providers = vec![
        ("Kademlia", KademliaDhtProvider::new().capabilities()),
        ("Simple", SimpleDhtProvider::new("simple").capabilities()),
    ];

    for (name, caps) in providers {
        println!("\n{}:", name);
        println!("  Content routing: {}", caps.supports_content_routing);
        println!("  Peer routing: {}", caps.supports_peer_routing);
        println!("  KV storage: {}", caps.supports_kv_storage);
        println!("  Semantic queries: {}", caps.supports_semantic_queries);
    }
}

fn print_capabilities(caps: &DhtCapabilities) {
    println!("  Content routing: {}", caps.supports_content_routing);
    println!("  Peer routing: {}", caps.supports_peer_routing);
    println!("  KV storage: {}", caps.supports_kv_storage);
    println!("  Range queries: {}", caps.supports_range_queries);
    println!("  Semantic queries: {}", caps.supports_semantic_queries);
    println!("  Max hops: {:?}", caps.max_query_hops);
    println!("  Custom routing: {}", caps.supports_custom_routing);
}

// Example custom DHT provider implementation
struct SimpleDhtProvider {
    name: String,
    stats: parking_lot::RwLock<DhtProviderStats>,
}

impl SimpleDhtProvider {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            stats: parking_lot::RwLock::new(DhtProviderStats::default()),
        }
    }
}

impl DhtProvider for SimpleDhtProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        "0.2.0"
    }

    fn capabilities(&self) -> DhtCapabilities {
        DhtCapabilities {
            supports_content_routing: true,
            supports_peer_routing: true,
            supports_kv_storage: false,
            supports_range_queries: false,
            supports_semantic_queries: false,
            max_query_hops: Some(10),
            supports_custom_routing: true,
        }
    }

    fn bootstrap(&self, peers: Vec<PeerId>) -> Result<(), DhtProviderError> {
        println!("  [{}] Bootstrapping with {} peers", self.name, peers.len());
        let mut stats = self.stats.write();
        stats.routing_table_size = peers.len();
        Ok(())
    }

    fn provide(&self, cid: &Cid) -> Result<(), DhtProviderError> {
        println!("  [{}] Providing content: {:?}", self.name, cid);
        Ok(())
    }

    fn find_providers(&self, cid: &Cid) -> Result<DhtQueryResult, DhtProviderError> {
        println!("  [{}] Finding providers for: {:?}", self.name, cid);

        let mut stats = self.stats.write();
        stats.total_queries += 1;
        stats.successful_queries += 1;
        stats.calculate_success_rate();

        Ok(DhtQueryResult {
            providers: vec![PeerId::random()],
            hops: 5,
            duration_ms: 100,
            success: true,
        })
    }

    fn find_peer(&self, peer_id: &PeerId) -> Result<DhtPeerInfo, DhtProviderError> {
        Ok(DhtPeerInfo {
            peer_id: *peer_id,
            addresses: vec!["/ip4/127.0.0.1/tcp/4001".to_string()],
            distance: Some(42),
        })
    }

    fn get_closest_peers(
        &self,
        _key: &[u8],
        count: usize,
    ) -> Result<Vec<DhtPeerInfo>, DhtProviderError> {
        let peers: Vec<DhtPeerInfo> = (0..count)
            .map(|_| DhtPeerInfo {
                peer_id: PeerId::random(),
                addresses: vec![],
                distance: Some(rand::random()),
            })
            .collect();
        Ok(peers)
    }

    fn stats(&self) -> DhtProviderStats {
        self.stats.read().clone()
    }
}
