//! Block operation commands
//!
//! This module provides raw block operations:
//! - `block_get` - Get raw block data
//! - `block_stat` - Show block statistics
//! - `block_put` - Store raw block
//! - `block_rm` - Remove block
//! - `list_blocks` - List all stored blocks

use anyhow::Result;

use crate::output::{self, error, format_bytes, print_cid, print_kv, success};
use crate::progress;

/// Get raw block data
pub async fn block_get(cid_str: String) -> Result<()> {
    use ipfrs_core::Cid;
    use ipfrs_storage::{BlockStoreConfig, BlockStoreTrait, SledBlockStore};

    let cid = cid_str
        .parse::<Cid>()
        .map_err(|e| anyhow::anyhow!("Invalid CID: {}", e))?;

    let config = BlockStoreConfig::default();
    let store = SledBlockStore::new(config)?;

    match store.get(&cid).await? {
        Some(block) => {
            use std::io::Write;
            std::io::stdout().write_all(block.data())?;
            Ok(())
        }
        None => {
            eprintln!("Block not found: {}", cid);
            std::process::exit(1);
        }
    }
}

/// Show block statistics
pub async fn block_stat(cid_str: String, format: &str) -> Result<()> {
    use ipfrs_core::Cid;
    use ipfrs_storage::{BlockStoreConfig, BlockStoreTrait, SledBlockStore};

    let cid = cid_str
        .parse::<Cid>()
        .map_err(|e| anyhow::anyhow!("Invalid CID: {}", e))?;

    let config = BlockStoreConfig::default();
    let store = SledBlockStore::new(config)?;

    match store.get(&cid).await? {
        Some(block) => {
            match format {
                "json" => {
                    println!("{{");
                    println!("  \"cid\": \"{}\",", cid);
                    println!("  \"size\": {}", block.size());
                    println!("}}");
                }
                _ => {
                    println!("CID: {}", cid);
                    println!("Size: {} bytes", block.size());
                }
            }
            Ok(())
        }
        None => {
            eprintln!("Block not found: {}", cid);
            std::process::exit(1);
        }
    }
}

/// Store raw block
pub async fn block_put(path: String, format: &str) -> Result<()> {
    use bytes::Bytes;
    use ipfrs_core::Block;
    use ipfrs_storage::{BlockStoreConfig, BlockStoreTrait, SledBlockStore};

    let file_path = std::path::Path::new(&path);
    let filename = file_path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| path.clone());

    // Read raw block data
    let pb = progress::spinner(&format!("Reading {}", filename));
    let data = tokio::fs::read(&path).await?;
    let size = data.len() as u64;
    let bytes_data = Bytes::from(data);
    progress::finish_spinner_success(&pb, &format!("Read {} bytes", size));

    // Create block from raw data
    let pb = progress::spinner("Creating block");
    let block = Block::new(bytes_data)?;
    let cid = *block.cid();
    progress::finish_spinner_success(&pb, "Block created");

    // Initialize storage
    let config = BlockStoreConfig::default();
    let store = SledBlockStore::new(config)?;

    // Store block
    let pb = progress::spinner("Storing raw block");
    store.put(&block).await?;
    progress::finish_spinner_success(&pb, "Block stored");

    match format {
        "json" => {
            println!("{{");
            println!("  \"cid\": \"{}\",", cid);
            println!("  \"size\": {}", block.size());
            println!("}}");
        }
        _ => {
            success("Raw block stored");
            print_cid("CID", &cid.to_string());
            print_kv("Size", &format_bytes(block.size()));
        }
    }

    Ok(())
}

/// Remove block
pub async fn block_rm(cid_str: String, force: bool) -> Result<()> {
    use ipfrs_core::Cid;
    use ipfrs_storage::{BlockStoreConfig, BlockStoreTrait, SledBlockStore};
    use std::io::{self, Write};

    let cid = cid_str
        .parse::<Cid>()
        .map_err(|e| anyhow::anyhow!("Invalid CID: {}", e))?;

    let config = BlockStoreConfig::default();
    let store = SledBlockStore::new(config)?;

    if !store.has(&cid).await? {
        error(&format!("Block not found: {}", cid));
        std::process::exit(1);
    }

    // Confirm deletion unless --force is used
    if !force {
        print!("Remove block {}? [y/N] ", cid);
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if !input.trim().eq_ignore_ascii_case("y") {
            output::info("Aborted");
            return Ok(());
        }
    }

    store.delete(&cid).await?;
    success(&format!("Removed block: {}", cid));

    Ok(())
}

/// List all stored blocks
pub async fn list_blocks(format: &str) -> Result<()> {
    use ipfrs_storage::{BlockStoreConfig, BlockStoreTrait, SledBlockStore};

    // Initialize storage
    let config = BlockStoreConfig::default();
    let store = SledBlockStore::new(config)?;

    // List all CIDs
    let cids = store.list_cids()?;

    if cids.is_empty() {
        match format {
            "json" => println!("[]"),
            _ => println!("No blocks stored"),
        }
    } else {
        match format {
            "json" => {
                println!("[");
                for (i, cid) in cids.iter().enumerate() {
                    if let Some(block) = store.get(cid).await? {
                        print!("  {{");
                        print!("\"cid\": \"{}\", ", cid);
                        print!("\"size\": {}", block.size());
                        if i < cids.len() - 1 {
                            println!("}},");
                        } else {
                            println!("}}");
                        }
                    }
                }
                println!("]");
            }
            _ => {
                println!("Stored blocks ({} total):", cids.len());
                for cid in cids {
                    if let Some(block) = store.get(&cid).await? {
                        println!("  {} ({} bytes)", cid, block.size());
                    }
                }
            }
        }
    }

    Ok(())
}
