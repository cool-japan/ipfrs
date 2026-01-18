//! Pin management commands
//!
//! This module provides pin operations:
//! - `pin_add` - Pin content
//! - `pin_rm` - Unpin content
//! - `pin_ls` - List pins
//! - `pin_verify` - Verify pin integrity

use anyhow::Result;

use crate::output::{self, error, print_header, print_kv, success};
use crate::progress;

/// Pin content
pub async fn pin_add(cid_str: &str, recursive: bool, name: Option<&str>) -> Result<()> {
    use ipfrs::{Node, NodeConfig};
    use ipfrs_core::Cid;

    let cid = cid_str
        .parse::<Cid>()
        .map_err(|e| anyhow::anyhow!("Invalid CID: {}", e))?;

    let pb = progress::spinner(&format!("Pinning {}", cid));
    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    node.pin_add(&cid, recursive, name.map(|s| s.to_string()))
        .await?;
    progress::finish_spinner_success(&pb, "Pinned successfully");

    let pin_type = if recursive { "recursively" } else { "directly" };
    success(&format!("Pinned {} {}", cid, pin_type));
    if let Some(n) = name {
        print_kv("Name", n);
    }

    node.stop().await?;
    Ok(())
}

/// Unpin content
pub async fn pin_rm(cid_str: &str, recursive: bool) -> Result<()> {
    use ipfrs::{Node, NodeConfig};
    use ipfrs_core::Cid;

    let cid = cid_str
        .parse::<Cid>()
        .map_err(|e| anyhow::anyhow!("Invalid CID: {}", e))?;

    let pb = progress::spinner(&format!("Unpinning {}", cid));
    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    node.pin_rm(&cid, recursive).await?;
    progress::finish_spinner_success(&pb, "Unpinned successfully");

    success(&format!("Unpinned {}", cid));

    node.stop().await?;
    Ok(())
}

/// List pins
pub async fn pin_ls(_pin_type: &str, format: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    let pb = progress::spinner("Listing pins");
    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    let pins = node.pin_ls()?;
    progress::finish_spinner_success(&pb, "Pins listed");

    match format {
        "json" => {
            println!("[");
            for (i, pin_info) in pins.iter().enumerate() {
                print!(
                    "  {{\"cid\": \"{}\", \"type\": \"{:?}\"}}",
                    pin_info.cid, pin_info.pin_type
                );
                if i < pins.len() - 1 {
                    println!(",");
                } else {
                    println!();
                }
            }
            println!("]");
        }
        _ => {
            if pins.is_empty() {
                output::info("No pinned content");
            } else {
                print_header(&format!("Pinned Content ({})", pins.len()));
                for pin_info in pins {
                    println!("  {} ({:?})", pin_info.cid, pin_info.pin_type);
                }
            }
        }
    }

    node.stop().await?;
    Ok(())
}

/// Verify pin integrity
pub async fn pin_verify(format: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    let pb = progress::spinner("Verifying pins");
    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    let results = node.pin_verify().await?;
    progress::finish_spinner_success(&pb, "Verification complete");

    let valid_count = results.iter().filter(|(_, valid)| *valid).count();
    let invalid_count = results.len() - valid_count;

    match format {
        "json" => {
            println!("{{");
            println!("  \"total\": {},", results.len());
            println!("  \"valid\": {},", valid_count);
            println!("  \"invalid\": {}", invalid_count);
            println!("}}");
        }
        _ => {
            print_header("Pin Verification Results");
            print_kv("Total pins", &results.len().to_string());
            print_kv("Valid", &valid_count.to_string());
            print_kv("Invalid", &invalid_count.to_string());

            if invalid_count > 0 {
                println!("\nInvalid pins:");
                for (cid, valid) in &results {
                    if !valid {
                        error(&format!("  {}", cid));
                    }
                }
            } else {
                success("All pins verified successfully");
            }
        }
    }

    node.stop().await?;
    Ok(())
}
