//! Repository management commands
//!
//! This module provides repository operations:
//! - `repo_gc` - Run garbage collection
//! - `repo_stat` - Show repository statistics
//! - `repo_fsck` - Verify repository integrity
//! - `repo_version` - Show repository version

use anyhow::Result;

use crate::output::{self, error, format_bytes, print_header, print_kv, success};
use crate::progress;

use super::stats::stats_repo;

/// Run garbage collection
pub async fn repo_gc(dry_run: bool, format: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    let action = if dry_run { "Analyzing" } else { "Running" };
    let pb = progress::spinner(&format!("{} garbage collection", action));
    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    let result = node.repo_gc(dry_run).await?;
    progress::finish_spinner_success(&pb, "GC complete");

    match format {
        "json" => {
            println!("{{");
            println!("  \"blocks_collected\": {},", result.blocks_collected);
            println!("  \"bytes_freed\": {},", result.bytes_freed);
            println!("  \"blocks_marked\": {},", result.blocks_marked);
            println!("  \"blocks_scanned\": {},", result.blocks_scanned);
            println!("  \"duration_ms\": {}", result.duration.as_millis());
            println!("}}");
        }
        _ => {
            print_header("Garbage Collection Results");
            print_kv("Blocks scanned", &result.blocks_scanned.to_string());
            print_kv("Blocks marked", &result.blocks_marked.to_string());
            print_kv("Blocks collected", &result.blocks_collected.to_string());
            print_kv("Bytes freed", &format_bytes(result.bytes_freed));
            print_kv(
                "Duration",
                &format!("{:.2}s", result.duration.as_secs_f64()),
            );

            if dry_run {
                output::warning("Dry run - no blocks were actually deleted");
            } else if result.blocks_collected > 0 {
                success(&format!(
                    "Freed {} from {} blocks",
                    format_bytes(result.bytes_freed),
                    result.blocks_collected
                ));
            } else {
                output::info("No unreferenced blocks found");
            }
        }
    }

    node.stop().await?;
    Ok(())
}

/// Show repository statistics
pub async fn repo_stat(format: &str) -> Result<()> {
    // Reuse the existing stats_repo function
    stats_repo(format).await
}

/// Verify repository integrity
pub async fn repo_fsck(format: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    let pb = progress::spinner("Checking repository integrity");
    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    let result = node.repo_fsck().await?;
    progress::finish_spinner_success(&pb, "Integrity check complete");

    match format {
        "json" => {
            println!("{{");
            println!("  \"blocks_checked\": {},", result.blocks_checked);
            println!("  \"blocks_valid\": {},", result.blocks_valid);
            println!("  \"blocks_corrupt\": {},", result.blocks_corrupt.len());
            println!("  \"blocks_missing\": {}", result.blocks_missing.len());
            println!("}}");
        }
        _ => {
            print_header("Repository Integrity Check");
            print_kv("Blocks checked", &result.blocks_checked.to_string());
            print_kv("Valid blocks", &result.blocks_valid.to_string());
            print_kv("Corrupt blocks", &result.blocks_corrupt.len().to_string());
            print_kv("Missing blocks", &result.blocks_missing.len().to_string());

            if result.blocks_corrupt.is_empty() && result.blocks_missing.is_empty() {
                success("Repository integrity verified - no issues found");
            } else {
                if !result.blocks_corrupt.is_empty() {
                    println!("\nCorrupt blocks:");
                    for cid in &result.blocks_corrupt {
                        error(&format!("  {}", cid));
                    }
                }
                if !result.blocks_missing.is_empty() {
                    println!("\nMissing blocks:");
                    for cid in &result.blocks_missing {
                        error(&format!("  {}", cid));
                    }
                }
            }
        }
    }

    node.stop().await?;
    Ok(())
}

/// Show repository version
pub async fn repo_version(_format: &str) -> Result<()> {
    print_header("Repository Version");
    print_kv("Format version", "1");
    print_kv("IPFRS version", env!("CARGO_PKG_VERSION"));
    Ok(())
}
