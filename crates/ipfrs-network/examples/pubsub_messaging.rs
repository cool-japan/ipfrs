//! GossipSub pub/sub messaging example
//!
//! This example demonstrates topic-based publish/subscribe messaging with GossipSub.
//!
//! Run with:
//! ```bash
//! cargo run --example pubsub_messaging
//! ```

use ipfrs_network::{GossipSubConfig, GossipSubManager, GossipSubMessage, MessageId, TopicId};
use libp2p::PeerId;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== IPFRS GossipSub Example ===\n");

    // 1. Create GossipSub manager with configuration
    println!("1. Creating GossipSub manager...");
    let config = GossipSubConfig {
        mesh_n_low: 4,
        mesh_n: 6,
        mesh_n_high: 12,
        gossip_n: 3,
        heartbeat_interval: Duration::from_secs(1),
        max_message_size: 1024 * 1024, // 1 MB
        enable_scoring: true,
        duplicate_cache_time: Duration::from_secs(120),
        max_duplicate_cache_size: 10000,
        enable_validation: true,
    };

    let manager = GossipSubManager::new(config);
    println!("   ✓ GossipSub manager created\n");

    // 2. Subscribe to topics
    println!("2. Subscribing to topics...");

    // Standard IPFRS topics
    manager.subscribe(TopicId::content_announce())?;
    println!("   ✓ Subscribed to: {}", TopicId::content_announce().0);

    manager.subscribe(TopicId::peer_announce())?;
    println!("   ✓ Subscribed to: {}", TopicId::peer_announce().0);

    manager.subscribe(TopicId::dht_events())?;
    println!("   ✓ Subscribed to: {}", TopicId::dht_events().0);

    // Custom topics
    let custom_topic = TopicId::new("/myapp/chat/general");
    manager.subscribe(custom_topic.clone())?;
    println!("   ✓ Subscribed to: {}\n", custom_topic.0);

    // 3. Build mesh network
    println!("3. Building topic mesh...");
    let peers = vec![
        PeerId::random(),
        PeerId::random(),
        PeerId::random(),
        PeerId::random(),
        PeerId::random(),
        PeerId::random(),
    ];

    for peer in &peers {
        manager.add_peer_to_mesh(&TopicId::content_announce(), *peer)?;
    }
    println!("   ✓ Added {} peers to mesh", peers.len());

    let mesh_peers = manager.get_mesh_peers(&TopicId::content_announce())?;
    println!("   Current mesh size: {}\n", mesh_peers.len());

    // 4. Publish messages
    println!("4. Publishing messages...");

    let source_peer = PeerId::random();

    // Publish content announcement
    let content_announcement = b"New content available: QmXYZ123...";
    let msg_id = manager.publish(
        TopicId::content_announce(),
        content_announcement.to_vec(),
        source_peer,
    )?;
    println!("   ✓ Published content announcement");
    println!("     Message ID: {:?}", &msg_id.0[..8]);

    // Publish peer announcement
    let peer_announcement = b"New peer joined: 12D3KooWABC...";
    let msg_id = manager.publish(
        TopicId::peer_announce(),
        peer_announcement.to_vec(),
        source_peer,
    )?;
    println!("   ✓ Published peer announcement");
    println!("     Message ID: {:?}\n", &msg_id.0[..8]);

    // 5. Handle incoming messages
    println!("5. Handling incoming messages...");

    // Simulate receiving messages
    let messages = [
        GossipSubMessage {
            id: MessageId::new(&peers[0], 1),
            source: peers[0],
            topic: TopicId::content_announce(),
            data: b"Content from peer 1".to_vec(),
            sequence: 1,
            timestamp: Instant::now(),
        },
        GossipSubMessage {
            id: MessageId::new(&peers[1], 1),
            source: peers[1],
            topic: TopicId::content_announce(),
            data: b"Content from peer 2".to_vec(),
            sequence: 1,
            timestamp: Instant::now(),
        },
        // Duplicate message (should be filtered)
        GossipSubMessage {
            id: MessageId::new(&peers[0], 1),
            source: peers[0],
            topic: TopicId::content_announce(),
            data: b"Content from peer 1".to_vec(),
            sequence: 1,
            timestamp: Instant::now(),
        },
    ];

    for (i, msg) in messages.iter().enumerate() {
        let is_new = manager.handle_message(msg.clone())?;
        if is_new {
            println!("   ✓ Message {}: NEW (from {})", i + 1, msg.source);
        } else {
            println!("   ⊗ Message {}: DUPLICATE (filtered)", i + 1);
        }
    }
    println!();

    // 6. Peer scoring
    println!("6. Managing peer scores...");

    // Update scores for different peers
    manager.update_peer_score(&peers[0], TopicId::content_announce(), 0.9);
    manager.update_peer_score(&peers[1], TopicId::content_announce(), 0.7);
    manager.update_peer_score(&peers[2], TopicId::content_announce(), 0.3); // Low score
    manager.update_peer_score(&peers[3], TopicId::content_announce(), 0.4); // Low score

    println!("   Peer scores:");
    for peer in &peers[..4] {
        if let Some(score) = manager.get_peer_score(peer) {
            println!("   - {}: {:.2}", peer, score.total_score);
        }
    }
    println!();

    // Identify peers to prune (below threshold)
    let threshold = 0.5;
    let peers_to_prune = manager.get_peers_to_prune(&TopicId::content_announce(), threshold);
    println!("   Peers below threshold ({:.1}):", threshold);
    for peer in &peers_to_prune {
        println!("   - {} (should be pruned)", peer);
    }
    println!();

    // Prune low-scoring peers
    for peer in peers_to_prune {
        manager.remove_peer_from_mesh(&TopicId::content_announce(), &peer)?;
        println!("   ✗ Removed {} from mesh", peer);
    }
    println!();

    // 7. Display statistics
    println!("7. GossipSub Statistics:");
    let stats = manager.stats();
    println!("   Subscribed topics: {}", stats.subscribed_topics);
    println!("   Messages published: {}", stats.messages_published);
    println!("   Messages received: {}", stats.messages_received);
    println!("   Duplicate messages: {}", stats.duplicate_messages);
    println!("   Invalid messages: {}", stats.invalid_messages);
    println!("   Active mesh peers: {}", stats.active_mesh_peers);
    println!("   Mesh graft events: {}", stats.mesh_graft_count);
    println!("   Mesh prune events: {}", stats.mesh_prune_count);

    println!("\n   Messages per topic:");
    for (topic, count) in &stats.messages_per_topic {
        println!("   - {}: {}", topic, count);
    }
    println!();

    // 8. List all subscribed topics
    println!("8. Subscribed topics:");
    let topics = manager.list_topics();
    for topic in topics {
        let _is_subscribed = manager.is_subscribed(&topic);
        let peers = manager.get_mesh_peers(&topic).unwrap_or_default();
        println!("   - {} ({} peers)", topic.0, peers.len());
    }

    println!("\n=== Example Complete ===");

    Ok(())
}

/// Example: Content announcement workflow
#[allow(dead_code)]
fn content_announcement_workflow() -> Result<(), Box<dyn std::error::Error>> {
    let manager = GossipSubManager::new(GossipSubConfig::default());

    // Subscribe to content announcement topic
    manager.subscribe(TopicId::content_announce())?;

    // Application stores new content
    let cid = "QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco";

    // Announce the content to the network
    let announcement = format!(
        r#"{{"type":"content_available","cid":"{}","provider":"12D3KooW..."}}"#,
        cid
    );

    let source = PeerId::random();
    manager.publish(
        TopicId::content_announce(),
        announcement.as_bytes().to_vec(),
        source,
    )?;

    println!("Content announced to network");

    Ok(())
}

/// Example: Chat application using custom topics
#[allow(dead_code)]
fn chat_application_example() -> Result<(), Box<dyn std::error::Error>> {
    let manager = GossipSubManager::new(GossipSubConfig::default());

    // Subscribe to multiple chat rooms
    let rooms = vec!["general", "dev", "random"];

    for room in rooms {
        let topic = TopicId::new(format!("/chat/{}", room));
        manager.subscribe(topic.clone())?;
        println!("Joined room: {}", room);
    }

    // Send message to a room
    let room_topic = TopicId::new("/chat/general");
    let message = r#"{"user":"alice","text":"Hello everyone!"}"#;

    let source = PeerId::random();
    manager.publish(room_topic, message.as_bytes().to_vec(), source)?;

    println!("Message sent to general room");

    Ok(())
}

/// Example: Event streaming
#[allow(dead_code)]
fn event_streaming_example() -> Result<(), Box<dyn std::error::Error>> {
    let manager = GossipSubManager::new(GossipSubConfig::default());

    // Subscribe to DHT events
    manager.subscribe(TopicId::dht_events())?;

    // Simulate DHT events
    let events = vec![
        r#"{"event":"peer_discovered","peer":"12D3KooWABC..."}"#,
        r#"{"event":"provider_found","cid":"QmXYZ...","peer":"12D3KooWDEF..."}"#,
        r#"{"event":"routing_table_updated","buckets":15,"peers":234}"#,
    ];

    let source = PeerId::random();
    for event in &events {
        manager.publish(TopicId::dht_events(), event.as_bytes().to_vec(), source)?;
    }

    println!("Published {} DHT events", events.len());

    Ok(())
}

/// Example: Message validation and filtering
#[allow(dead_code)]
fn message_validation_example() -> Result<(), Box<dyn std::error::Error>> {
    let config = GossipSubConfig {
        enable_validation: true,
        max_message_size: 1024, // 1 KB limit
        ..Default::default()
    };

    let manager = GossipSubManager::new(config);
    let topic = TopicId::new("/validated/topic");
    manager.subscribe(topic.clone())?;

    // Try to publish oversized message
    let large_message = vec![0u8; 2048]; // 2 KB (exceeds limit)
    let source = PeerId::random();

    match manager.publish(topic, large_message, source) {
        Ok(_) => println!("Message published"),
        Err(e) => println!("Message rejected: {}", e),
    }

    Ok(())
}

/// Example: Mesh optimization
#[allow(dead_code)]
fn mesh_optimization_example() -> Result<(), Box<dyn std::error::Error>> {
    let config = GossipSubConfig {
        mesh_n_low: 4,
        mesh_n: 6,
        mesh_n_high: 12,
        enable_scoring: true,
        ..Default::default()
    };

    let manager = GossipSubManager::new(config);
    let topic = TopicId::new("/optimized/topic");
    manager.subscribe(topic.clone())?;

    // Add peers to mesh
    let peers: Vec<_> = (0..15).map(|_| PeerId::random()).collect();
    for peer in &peers {
        manager.add_peer_to_mesh(&topic, *peer)?;
    }

    // Mesh is now over D_high (12), should trigger pruning
    let mesh_size = manager.get_mesh_peers(&topic)?.len();
    println!("Mesh size: {} (target: 6, max: 12)", mesh_size);

    // Score peers and prune low-scoring ones
    for (i, peer) in peers.iter().enumerate() {
        let score = 0.5 + (i as f64 / peers.len() as f64) * 0.5;
        manager.update_peer_score(peer, topic.clone(), score);
    }

    let to_prune = manager.get_peers_to_prune(&topic, 0.6);
    println!("Peers to prune: {}", to_prune.len());

    for peer in to_prune {
        manager.remove_peer_from_mesh(&topic, &peer)?;
    }

    let final_mesh_size = manager.get_mesh_peers(&topic)?.len();
    println!("Final mesh size: {}", final_mesh_size);

    Ok(())
}
