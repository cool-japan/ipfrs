//! File operation commands
//!
//! This module provides file-related operations:
//! - `init_repo` - Initialize repository
//! - `add_file` - Add file to IPFRS
//! - `get_file` - Retrieve file from IPFRS
//! - `cat_file` - Output file contents to stdout
//! - `ls_directory` - List directory contents

use anyhow::Result;

use crate::config::Config;
use crate::output::{self, error, format_bytes, print_cid, print_header, print_kv, success};
use crate::progress;

/// Initialize IPFRS repository
pub async fn init_repo(data_dir: String) -> Result<()> {
    use std::fs;

    let path = std::path::Path::new(&data_dir);

    if path.exists() {
        if path.is_file() {
            return Err(anyhow::anyhow!(
                "Path exists as a file, not a directory: {}\nPlease choose a different location or remove the file.",
                data_dir
            ));
        }

        // Check if already initialized
        if path.join("config.toml").exists() {
            output::warning(&format!("Repository already initialized at: {}", data_dir));
            println!("\nTo reinitialize, first remove the existing repository:");
            println!("  rm -rf {}", data_dir);
            return Ok(());
        }
    }

    let pb = progress::spinner("Initializing repository");

    // Create directory structure
    fs::create_dir_all(path.join("blocks"))?;
    fs::create_dir_all(path.join("keystore"))?;
    fs::create_dir_all(path.join("datastore"))?;

    // Generate default configuration
    let config_path = path.join("config.toml");
    let config_content = Config::generate_default_config();
    fs::write(&config_path, config_content)?;

    progress::finish_spinner_success(&pb, "Repository initialized");

    success(&format!("Initialized IPFRS repository at: {}", data_dir));

    println!();
    print_header("Repository Structure");
    print_kv("blocks", &path.join("blocks").display().to_string());
    print_kv("keystore", &path.join("keystore").display().to_string());
    print_kv("datastore", &path.join("datastore").display().to_string());
    print_kv("config", &config_path.display().to_string());

    println!();
    output::print_section("Next Steps");
    println!("  1. Review configuration: {}", config_path.display());
    println!("  2. Start the daemon: ipfrs daemon");
    println!("  3. Add content: ipfrs add <file>");
    println!();
    output::info("Repository ready to use!");

    Ok(())
}

/// Add file to IPFRS
pub async fn add_file(path: String, format: &str) -> Result<()> {
    use bytes::Bytes;
    use ipfrs_core::Block;
    use ipfrs_storage::{BlockStoreConfig, BlockStoreTrait, SledBlockStore};

    let file_path = std::path::Path::new(&path);

    // Validate file exists
    if !file_path.exists() {
        return Err(anyhow::anyhow!(
            "File not found: {}\nPlease check the path and try again.",
            path
        ));
    }

    // Validate it's a file (not a directory)
    if !file_path.is_file() {
        return Err(anyhow::anyhow!(
            "Path is not a file: {}\nTo add a directory, use 'ipfrs add -r <directory>'",
            path
        ));
    }

    let filename = file_path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| path.clone());

    // Get file size for progress
    let metadata = tokio::fs::metadata(&path).await?;
    let file_size = metadata.len();

    // Warn about very large files that will consume significant memory.
    const VERY_LARGE_FILE_THRESHOLD: u64 = 100 * 1024 * 1024; // 100 MB
    if file_size > VERY_LARGE_FILE_THRESHOLD {
        output::warning(&format!(
            "Large file detected: {}. This may take a while.",
            format_bytes(file_size)
        ));
    }

    // For files >= 10 MB on a TTY, show a progress bar; otherwise use a spinner.
    let read_pb = progress::file_progress_bar(file_size, "Reading");
    let spinner_pb = if read_pb.is_hidden() {
        Some(progress::spinner(&format!("Reading {}", filename)))
    } else {
        None
    };

    // Read file
    let data = tokio::fs::read(&path).await?;
    let bytes_data = Bytes::from(data);

    // Advance the bar to completion (we read in one shot).
    read_pb.inc(file_size);
    read_pb.finish_and_clear();

    if let Some(ref pb) = spinner_pb {
        progress::finish_spinner_success(
            pb,
            &format!("Read {} ({})", filename, format_bytes(file_size)),
        );
    }

    // Create block
    let pb = progress::spinner("Creating block");
    let block = Block::new(bytes_data)?;
    let cid = *block.cid();
    progress::finish_spinner_success(&pb, "Block created");

    // Initialize storage
    let config = BlockStoreConfig::default();
    let store = SledBlockStore::new(config)?;

    // Store block
    let pb = progress::spinner("Storing block");
    store.put(&block).await?;
    progress::finish_spinner_success(&pb, "Block stored");

    match format {
        "json" => {
            println!("{{");
            println!("  \"path\": \"{}\",", path);
            println!("  \"cid\": \"{}\",", cid);
            println!("  \"size\": {}", block.size());
            println!("}}");
        }
        _ => {
            success(&format!("Added {}", filename));
            print_cid("CID", &cid.to_string());
            print_kv("Size", &format_bytes(block.size()));
        }
    }

    Ok(())
}

/// Get file from IPFRS and save to disk.
///
/// `timeout_secs` bounds the entire block-fetch operation.  A value of `0`
/// disables the timeout (waits indefinitely).
pub async fn get_file(cid_str: String, output: Option<String>, timeout_secs: u64) -> Result<()> {
    use ipfrs_core::Cid;
    use ipfrs_storage::{BlockStoreConfig, BlockStoreTrait, SledBlockStore};
    use std::time::Duration;
    use tokio::fs;

    // Parse CID
    let cid = cid_str.parse::<Cid>().map_err(|e| {
        anyhow::anyhow!(
            "Invalid CID format: {}\n\nExpected format: QmXXXXXXXXXX or bafyXXXXXXXXXX",
            e
        )
    })?;

    let pb = progress::spinner(&format!("Retrieving {}", cid));

    // Initialize storage
    let config = BlockStoreConfig::default();
    let store = SledBlockStore::new(config)?;

    // Retrieve block, wrapping with an optional timeout.
    let fetch_result = if timeout_secs > 0 {
        tokio::time::timeout(Duration::from_secs(timeout_secs), store.get(&cid))
            .await
            .map_err(|_| {
                anyhow::anyhow!(
                    "Timeout after {}s fetching {}\n\nThe block may be available on the network but \
                     is unreachable right now.\nTry increasing --timeout or checking connectivity.",
                    timeout_secs, cid
                )
            })??
    } else {
        store.get(&cid).await?
    };

    match fetch_result {
        Some(block) => {
            progress::finish_spinner_success(&pb, "Block retrieved");

            let output_path = output.unwrap_or_else(|| cid_str.clone());

            // Check if output file already exists
            if std::path::Path::new(&output_path).exists() {
                output::warning(&format!("Overwriting existing file: {}", output_path));
            }

            // Show a progress bar for large blocks being written to disk.
            let write_pb = progress::file_progress_bar(block.size(), "Saving");
            fs::write(&output_path, block.data()).await?;
            write_pb.inc(block.size());
            write_pb.finish_and_clear();

            success(&format!("Saved to: {}", output_path));
            print_kv("Size", &format_bytes(block.size()));
            Ok(())
        }
        None => {
            progress::finish_spinner_error(&pb, "Block not found");
            Err(anyhow::anyhow!(
                "Block not found: {}\n\nPossible reasons:\n  • Content was never added to IPFRS\n  • Content was garbage collected\n  • Wrong CID format\n\nTry: ipfrs dht findprovs {} to find providers",
                cid, cid
            ))
        }
    }
}

/// Output file contents to stdout.
///
/// `timeout_secs` bounds the entire block-fetch operation.  A value of `0`
/// disables the timeout (waits indefinitely).
pub async fn cat_file(cid_str: String, timeout_secs: u64) -> Result<()> {
    use ipfrs_core::Cid;
    use ipfrs_storage::{BlockStoreConfig, BlockStoreTrait, SledBlockStore};
    use std::time::Duration;

    // Parse CID
    let cid = cid_str.parse::<Cid>().map_err(|e| {
        anyhow::anyhow!(
            "Invalid CID format: {}\n\nExpected format: QmXXXXXXXXXX or bafyXXXXXXXXXX",
            e
        )
    })?;

    // Initialize storage
    let config = BlockStoreConfig::default();
    let store = SledBlockStore::new(config)?;

    // Retrieve block, wrapping with an optional timeout.
    let fetch_result = if timeout_secs > 0 {
        tokio::time::timeout(Duration::from_secs(timeout_secs), store.get(&cid))
            .await
            .map_err(|_| {
                anyhow::anyhow!(
                    "Timeout after {}s fetching {}\n\nThe block may be available on the network but \
                     is unreachable right now.\nTry increasing --timeout or checking connectivity.",
                    timeout_secs, cid
                )
            })??
    } else {
        store.get(&cid).await?
    };

    match fetch_result {
        Some(block) => {
            // Write to stdout
            use std::io::Write;
            std::io::stdout().write_all(block.data())?;
            std::io::stdout().flush()?;
            Ok(())
        }
        None => Err(anyhow::anyhow!(
            "Block not found: {}\n\nPossible reasons:\n  • Content was never added to IPFRS\n  • Content was garbage collected\n  • Wrong CID format\n\nTry: ipfrs dht findprovs {} to find providers",
            cid, cid
        )),
    }
}

/// Directory entry for ls command
#[derive(Debug)]
pub struct DirectoryEntry {
    pub name: String,
    pub cid: String,
    pub size: u64,
    pub entry_type: String,
}

/// List directory contents
pub async fn ls_directory(cid_str: String, format: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};
    use ipfrs_core::Cid;

    let cid = cid_str
        .parse::<Cid>()
        .map_err(|e| anyhow::anyhow!("Invalid CID: {}", e))?;

    let pb = progress::spinner(&format!("Listing directory {}", cid));
    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    match node.dag_get(&cid).await? {
        Some(ipld) => {
            progress::finish_spinner_success(&pb, "Directory retrieved");

            // Extract links from IPLD node (UnixFS directory structure)
            let entries = extract_directory_entries(&ipld)?;

            match format {
                "json" => {
                    println!("[");
                    for (i, entry) in entries.iter().enumerate() {
                        print!("  {{");
                        print!("\"name\": \"{}\", ", entry.name);
                        print!("\"cid\": \"{}\", ", entry.cid);
                        print!("\"size\": {}, ", entry.size);
                        print!("\"type\": \"{}\"", entry.entry_type);
                        print!("}}");
                        if i < entries.len() - 1 {
                            println!(",");
                        } else {
                            println!();
                        }
                    }
                    println!("]");
                }
                _ => {
                    if entries.is_empty() {
                        output::info("Empty directory");
                    } else {
                        print_header(&format!("Directory: {}", cid));
                        for entry in entries {
                            println!(
                                "  {} {} {}",
                                entry.entry_type,
                                format_bytes(entry.size),
                                entry.name
                            );
                            println!("    CID: {}", entry.cid);
                        }
                    }
                }
            }
        }
        None => {
            progress::finish_spinner_error(&pb, "Directory not found");
            error(&format!("Directory not found: {}", cid));
            std::process::exit(1);
        }
    }

    node.stop().await?;
    Ok(())
}

/// Extract directory entries from IPLD structure
pub fn extract_directory_entries(ipld: &ipfrs_core::Ipld) -> Result<Vec<DirectoryEntry>> {
    use ipfrs_core::Ipld;

    let mut entries = Vec::new();

    // Try to extract links from the IPLD structure
    match ipld {
        Ipld::Map(map) => {
            // Check if this is a UnixFS directory with "Links" field
            if let Some(Ipld::List(links)) = map.get("Links") {
                for link in links {
                    if let Ipld::Map(link_map) = link {
                        let name = link_map
                            .get("Name")
                            .and_then(|v| match v {
                                Ipld::String(s) => Some(s.clone()),
                                _ => None,
                            })
                            .unwrap_or_else(|| "<unnamed>".to_string());

                        let cid = link_map
                            .get("Hash")
                            .and_then(|v| match v {
                                Ipld::Link(c) => Some(c.to_string()),
                                Ipld::String(s) => Some(s.clone()),
                                _ => None,
                            })
                            .unwrap_or_else(|| "<unknown>".to_string());

                        let size = link_map
                            .get("Size")
                            .and_then(|v| match v {
                                Ipld::Integer(n) => Some(*n as u64),
                                _ => None,
                            })
                            .unwrap_or(0);

                        // Try to determine type from the structure
                        let entry_type = if link_map.contains_key("Links") {
                            "dir"
                        } else {
                            "file"
                        };

                        entries.push(DirectoryEntry {
                            name,
                            cid,
                            size,
                            entry_type: entry_type.to_string(),
                        });
                    }
                }
            } else {
                // Fallback: treat all map entries as directory entries
                for (key, value) in map {
                    let (cid_str, size, entry_type) = match value {
                        Ipld::Link(c) => (c.to_string(), 0, "link"),
                        Ipld::Map(m) => {
                            let has_links = m.contains_key("Links");
                            let size = m
                                .get("Size")
                                .and_then(|v| match v {
                                    Ipld::Integer(n) => Some(*n as u64),
                                    _ => None,
                                })
                                .unwrap_or(0);
                            let typ = if has_links { "dir" } else { "file" };
                            ("<embedded>".to_string(), size, typ)
                        }
                        _ => (format!("{:?}", value), 0, "unknown"),
                    };

                    entries.push(DirectoryEntry {
                        name: key.clone(),
                        cid: cid_str,
                        size,
                        entry_type: entry_type.to_string(),
                    });
                }
            }
        }
        Ipld::List(list) => {
            // If it's a list, enumerate entries
            for (i, item) in list.iter().enumerate() {
                let (cid_str, size, entry_type) = match item {
                    Ipld::Link(c) => (c.to_string(), 0, "link"),
                    Ipld::Map(m) => {
                        let has_links = m.contains_key("Links");
                        let size = m
                            .get("Size")
                            .and_then(|v| match v {
                                Ipld::Integer(n) => Some(*n as u64),
                                _ => None,
                            })
                            .unwrap_or(0);
                        let typ = if has_links { "dir" } else { "file" };
                        ("<embedded>".to_string(), size, typ)
                    }
                    _ => (format!("{:?}", item), 0, "unknown"),
                };

                entries.push(DirectoryEntry {
                    name: format!("item-{}", i),
                    cid: cid_str,
                    size,
                    entry_type: entry_type.to_string(),
                });
            }
        }
        _ => {
            return Err(anyhow::anyhow!(
                "Not a directory: expected Map or List structure"
            ));
        }
    }

    Ok(entries)
}
