//! Statistics commands
//!
//! This module provides statistics-related commands:
//! - `stats_repo` - Repository statistics
//! - `stats_bw` - Bandwidth statistics
//! - `stats_bitswap` - Bitswap statistics
//! - `print_info` - Show node info
//! - `print_version` - Show version
//! - `show_id` - Show peer identity
//! - `ping_peer` - Ping a peer

use anyhow::Result;

use crate::output::{self, format_bytes, print_header, print_kv};
use crate::progress;

/// Show repository statistics
pub async fn stats_repo(format: &str) -> Result<()> {
    use ipfrs_storage::{BlockStoreConfig, BlockStoreTrait, SledBlockStore};

    // Initialize storage
    let config = BlockStoreConfig::default();
    let storage_path = config.path.clone();
    let store = SledBlockStore::new(config)?;

    // Gather statistics
    let cids = store.list_cids()?;
    let num_blocks = cids.len();

    // Calculate total size
    let mut total_size: u64 = 0;
    for cid in &cids {
        if let Some(block) = store.get(cid).await? {
            total_size += block.size();
        }
    }

    match format {
        "json" => {
            println!("{{");
            println!("  \"num_blocks\": {},", num_blocks);
            println!("  \"total_size\": {},", total_size);
            println!("  \"storage_path\": \"{}\"", storage_path.display());
            println!("}}");
        }
        _ => {
            print_header("IPFRS Repository Statistics");
            print_kv("Number of blocks", &num_blocks.to_string());
            print_kv("Total size", &format_bytes(total_size));
            print_kv("Storage path", &storage_path.display().to_string());

            if num_blocks > 0 {
                let avg_size = total_size / num_blocks as u64;
                print_kv("Average block size", &format_bytes(avg_size));
            }
        }
    }

    Ok(())
}

/// Show bandwidth statistics
pub async fn stats_bw(format: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    let pb = progress::spinner("Collecting bandwidth statistics");
    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    let stats = node.network_stats()?;
    progress::finish_spinner_success(&pb, "Statistics collected");

    match format {
        "json" => {
            println!("{{");
            println!("  \"total_in\": {},", stats.bytes_received);
            println!("  \"total_out\": {},", stats.bytes_sent);
            println!("  \"rate_in\": 0,");
            println!("  \"rate_out\": 0");
            println!("}}");
        }
        _ => {
            print_header("Bandwidth Statistics");
            print_kv("Total received", &format_bytes(stats.bytes_received));
            print_kv("Total sent", &format_bytes(stats.bytes_sent));
            print_kv("Connected peers", &stats.connected_peers.to_string());
        }
    }

    node.stop().await?;
    Ok(())
}

/// Show bitswap statistics
pub async fn stats_bitswap(format: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    let pb = progress::spinner("Collecting bitswap statistics");
    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    let stats = node.bitswap_stats()?;
    progress::finish_spinner_success(&pb, "Statistics collected");

    match format {
        "json" => {
            println!("{{");
            println!("  \"want_list_size\": {},", stats.want_list_size);
            println!("  \"have_list_size\": {},", stats.have_list_size);
            println!("  \"pending_requests\": {},", stats.pending_requests);
            println!(
                "  \"peers_with_pending_requests\": {}",
                stats.peers_with_pending_requests
            );
            println!("}}");
        }
        _ => {
            print_header("Bitswap Statistics");
            print_kv("Want list size", &stats.want_list_size.to_string());
            print_kv("Have list size", &stats.have_list_size.to_string());
            print_kv("Pending requests", &stats.pending_requests.to_string());
            print_kv(
                "Peers with requests",
                &stats.peers_with_pending_requests.to_string(),
            );
        }
    }

    node.stop().await?;
    Ok(())
}

/// Print IPFRS info
pub fn print_info() {
    println!("IPFRS - Inter-Planet File RUST System");
    println!("Version: 0.3.0 (The Fast & The Wise)");
    println!();
    println!("Architecture:");
    println!("  Logical Layer:");
    println!("    - Semantic Router (Vector Search / Logic Solver)");
    println!("    - Differentiable Storage (Gradient Tracking)");
    println!();
    println!("  Physical Layer:");
    println!("    - TensorSwap (Optimized Tensor Streaming)");
    println!("    - Rust Native Store (Sled / ParityDB)");
    println!("    - Network Stack (libp2p / QUIC)");
    println!();
    println!("Technology Stack:");
    println!("  - Runtime: Tokio async");
    println!("  - Network: libp2p, QUIC");
    println!("  - Storage: Sled");
    println!("  - Zero-Copy: Apache Arrow, Safetensors");
    println!("  - Vector Search: HNSW");
}

/// Print version
pub fn print_version() {
    println!("ipfrs {}", env!("CARGO_PKG_VERSION"));
}

/// Show peer identity
pub async fn show_id(format: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    let peer_id = node.peer_id()?;
    let stats = node.network_stats()?;

    match format {
        "json" => {
            println!("{{");
            println!("  \"peer_id\": \"{}\",", peer_id);
            println!("  \"addresses\": [");
            for (i, addr) in stats.listen_addrs.iter().enumerate() {
                if i > 0 {
                    println!(",");
                }
                print!("    \"{}\"", addr);
            }
            println!();
            println!("  ]");
            println!("}}");
        }
        _ => {
            println!("Peer ID: {}", peer_id);
            println!("Addresses:");
            for addr in &stats.listen_addrs {
                println!("  {}", addr);
            }
        }
    }

    node.stop().await?;
    Ok(())
}

/// Ping a peer
pub async fn ping_peer(peer_id: &str, count: u32) -> Result<()> {
    use ipfrs::{Node, NodeConfig};
    use std::time::Instant;

    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    output::info(&format!("PING {} ({} pings)", peer_id, count));

    let mut successful = 0u32;
    let mut total_time = std::time::Duration::ZERO;

    for i in 0..count {
        let start = Instant::now();
        match node.ping(peer_id).await {
            Ok(_) => {
                let elapsed = start.elapsed();
                total_time += elapsed;
                successful += 1;
                println!("seq={} time={:.2}ms", i + 1, elapsed.as_secs_f64() * 1000.0);
            }
            Err(e) => {
                println!("seq={} error: {}", i + 1, e);
            }
        }

        if i < count - 1 {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        }
    }

    println!();
    println!("--- {} ping statistics ---", peer_id);
    println!(
        "{} packets transmitted, {} received, {:.1}% packet loss",
        count,
        successful,
        ((count - successful) as f64 / count as f64) * 100.0
    );

    if successful > 0 {
        let avg_time = total_time.as_secs_f64() * 1000.0 / successful as f64;
        println!("rtt avg = {:.2}ms", avg_time);
    }

    node.stop().await?;
    Ok(())
}
