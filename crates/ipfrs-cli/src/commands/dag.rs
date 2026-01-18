//! DAG (Directed Acyclic Graph) commands
//!
//! This module provides DAG operations:
//! - `dag_get` - Get DAG node
//! - `dag_put` - Store DAG node
//! - `dag_resolve` - Resolve IPLD path
//! - `dag_export` - Export DAG to CAR file
//! - `dag_import` - Import DAG from CAR file

use anyhow::Result;

use crate::output::{error, format_bytes, print_cid, print_header, print_kv, success};
use crate::progress;

/// Get DAG node
pub async fn dag_get(cid_str: &str, format: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};
    use ipfrs_core::Cid;

    let cid = cid_str
        .parse::<Cid>()
        .map_err(|e| anyhow::anyhow!("Invalid CID: {}", e))?;

    let pb = progress::spinner(&format!("Retrieving DAG node {}", cid));
    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    match node.dag_get(&cid).await? {
        Some(ipld) => {
            progress::finish_spinner_success(&pb, "DAG node retrieved");

            match format {
                "json" => {
                    // Serialize IPLD to JSON
                    let json = serde_json::to_string_pretty(&ipld)?;
                    println!("{}", json);
                }
                _ => {
                    // Text format - pretty print IPLD
                    print_header(&format!("DAG Node: {}", cid));
                    let json = serde_json::to_string_pretty(&ipld)?;
                    println!("{}", json);
                }
            }
        }
        None => {
            progress::finish_spinner_error(&pb, "DAG node not found");
            error(&format!("DAG node not found: {}", cid));
            std::process::exit(1);
        }
    }

    node.stop().await?;
    Ok(())
}

/// Store DAG node
pub async fn dag_put(data: &str, format: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    let pb = progress::spinner("Parsing JSON data");

    // Parse JSON to IPLD
    let ipld: ipfrs_core::Ipld =
        serde_json::from_str(data).map_err(|e| anyhow::anyhow!("Invalid JSON: {}", e))?;

    progress::finish_spinner_success(&pb, "JSON parsed");

    let pb = progress::spinner("Storing DAG node");
    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    let cid = node.dag_put(ipld).await?;
    progress::finish_spinner_success(&pb, "DAG node stored");

    match format {
        "json" => {
            println!("{{");
            println!("  \"cid\": \"{}\"", cid);
            println!("}}");
        }
        _ => {
            success("DAG node stored");
            print_cid("CID", &cid.to_string());
        }
    }

    node.stop().await?;
    Ok(())
}

/// Resolve IPLD path
pub async fn dag_resolve(path_str: &str, format: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};
    use ipfrs_core::Cid;

    // Parse the path - format: /ipfs/CID/path/to/data or just CID/path
    let parts: Vec<&str> = path_str.trim_start_matches("/ipfs/").split('/').collect();

    if parts.is_empty() {
        return Err(anyhow::anyhow!("Invalid path: {}", path_str));
    }

    let root_cid_str = parts[0];
    let sub_path = if parts.len() > 1 {
        parts[1..].join("/")
    } else {
        String::new()
    };

    let root_cid = root_cid_str
        .parse::<Cid>()
        .map_err(|e| anyhow::anyhow!("Invalid CID: {}", e))?;

    let pb = progress::spinner(&format!("Resolving path: {}", path_str));
    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    match node.dag_resolve(&root_cid, &sub_path).await? {
        Some(resolved_cid) => {
            progress::finish_spinner_success(&pb, "Path resolved");

            match format {
                "json" => {
                    println!("{{");
                    println!("  \"cid\": \"{}\"", resolved_cid);
                    println!("}}");
                }
                _ => {
                    success(&format!("Resolved: {}", path_str));
                    print_cid("CID", &resolved_cid.to_string());
                }
            }
        }
        None => {
            progress::finish_spinner_error(&pb, "Path not found");
            error(&format!("Could not resolve path: {}", path_str));
            std::process::exit(1);
        }
    }

    node.stop().await?;
    Ok(())
}

/// Export DAG to CAR file
pub async fn dag_export(cid_str: &str, output_path: &str, format: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};
    use ipfrs_core::Cid;

    let cid = cid_str
        .parse::<Cid>()
        .map_err(|e| anyhow::anyhow!("Invalid CID: {}", e))?;

    let pb = progress::spinner(&format!("Exporting DAG {} to {}", cid, output_path));
    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    let stats = node.dag_export(&cid, output_path).await?;
    progress::finish_spinner_success(&pb, "DAG exported successfully");

    match format {
        "json" => {
            println!("{{");
            println!("  \"blocks_exported\": {},", stats.blocks_exported);
            println!("  \"bytes_exported\": {}", stats.bytes_exported);
            println!("}}");
        }
        _ => {
            success(&format!("Exported DAG to {}", output_path));
            print_header("Export Statistics");
            print_kv("Blocks exported", &stats.blocks_exported.to_string());
            print_kv("Bytes exported", &format_bytes(stats.bytes_exported));
            print_kv("File", output_path);
        }
    }

    node.stop().await?;
    Ok(())
}

/// Import DAG from CAR file
pub async fn dag_import(car_path: &str, format: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    let pb = progress::spinner(&format!("Importing DAG from {}", car_path));
    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    let stats = node.dag_import(car_path).await?;
    progress::finish_spinner_success(&pb, "DAG imported successfully");

    match format {
        "json" => {
            println!("{{");
            println!("  \"blocks_imported\": {},", stats.blocks_imported);
            println!("  \"bytes_imported\": {}", stats.bytes_imported);
            println!("}}");
        }
        _ => {
            success(&format!("Imported DAG from {}", car_path));
            print_header("Import Statistics");
            print_kv("Blocks imported", &stats.blocks_imported.to_string());
            print_kv("Bytes imported", &format_bytes(stats.bytes_imported));
        }
    }

    node.stop().await?;
    Ok(())
}
