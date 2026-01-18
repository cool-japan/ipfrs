//! Tensor operation commands
//!
//! This module provides tensor-related operations:
//! - `tensor_add` - Add tensor file
//! - `tensor_get` - Get tensor by CID
//! - `tensor_info` - Show tensor metadata
//! - `tensor_export` - Export tensor to different format

use anyhow::Result;

use crate::output::{self, error, format_bytes, print_cid, print_header, print_kv, success};
use crate::progress;

/// Safetensors metadata structure
#[derive(Debug)]
pub struct SafetensorsInfo {
    pub num_tensors: usize,
    pub tensors: Vec<(String, Vec<usize>, String)>,
}

/// Extract metadata from safetensors file
pub fn extract_safetensors_metadata(data: &[u8]) -> Option<SafetensorsInfo> {
    // Safetensors format starts with an 8-byte header containing the JSON metadata length
    if data.len() < 8 {
        return None;
    }

    // Read the first 8 bytes as u64 (little-endian)
    let metadata_len = u64::from_le_bytes([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ]) as usize;

    if data.len() < 8 + metadata_len {
        return None;
    }

    // Extract and parse JSON metadata
    let metadata_bytes = &data[8..8 + metadata_len];
    let metadata_str = std::str::from_utf8(metadata_bytes).ok()?;
    let metadata: serde_json::Value = serde_json::from_str(metadata_str).ok()?;

    // Extract tensor information
    let mut tensors = Vec::new();
    if let Some(obj) = metadata.as_object() {
        for (name, info) in obj {
            if name == "__metadata__" {
                continue; // Skip metadata field
            }

            if let Some(tensor_info) = info.as_object() {
                let shape = tensor_info
                    .get("shape")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as usize))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();

                let dtype = tensor_info
                    .get("dtype")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();

                tensors.push((name.clone(), shape, dtype));
            }
        }
    }

    Some(SafetensorsInfo {
        num_tensors: tensors.len(),
        tensors,
    })
}

/// Add tensor file
pub async fn tensor_add(path: &str, format: &str) -> Result<()> {
    use bytes::Bytes;
    use ipfrs_core::Block;
    use ipfrs_storage::{BlockStoreConfig, BlockStoreTrait, SledBlockStore};

    let file_path = std::path::Path::new(path);
    let filename = file_path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string());

    // Get file size for progress
    let metadata = tokio::fs::metadata(path).await?;
    let file_size = metadata.len();

    let pb = progress::spinner(&format!("Reading tensor file {}", filename));

    // Read tensor file
    let data = tokio::fs::read(path).await?;
    let bytes_data = Bytes::from(data.clone());

    progress::finish_spinner_success(
        &pb,
        &format!("Read {} ({})", filename, format_bytes(file_size)),
    );

    // Try to extract tensor metadata if it's a safetensors file
    let tensor_info = if path.ends_with(".safetensors") {
        extract_safetensors_metadata(&data)
    } else {
        None
    };

    // Create block
    let pb = progress::spinner("Creating tensor block");
    let block = Block::new(bytes_data)?;
    let cid = *block.cid();
    progress::finish_spinner_success(&pb, "Tensor block created");

    // Initialize storage
    let config = BlockStoreConfig::default();
    let store = SledBlockStore::new(config)?;

    // Store block
    let pb = progress::spinner("Storing tensor");
    store.put(&block).await?;
    progress::finish_spinner_success(&pb, "Tensor stored");

    match format {
        "json" => {
            println!("{{");
            println!("  \"path\": \"{}\",", path);
            println!("  \"cid\": \"{}\",", cid);
            println!("  \"size\": {}", block.size());
            if let Some(info) = tensor_info {
                println!("  ,\"metadata\": {{");
                println!("    \"format\": \"safetensors\",");
                println!("    \"tensors\": {}", info.num_tensors);
                println!("  }}");
            }
            println!("}}");
        }
        _ => {
            success(&format!("Added tensor {}", filename));
            print_cid("CID", &cid.to_string());
            print_kv("Size", &format_bytes(block.size()));
            if let Some(info) = tensor_info {
                print_kv("Format", "safetensors");
                print_kv("Tensors", &info.num_tensors.to_string());
            }
        }
    }

    Ok(())
}

/// Get tensor by CID
pub async fn tensor_get(cid_str: &str, output: Option<&str>) -> Result<()> {
    use ipfrs_core::Cid;
    use ipfrs_storage::{BlockStoreConfig, BlockStoreTrait, SledBlockStore};
    use tokio::fs;

    // Parse CID
    let cid = cid_str
        .parse::<Cid>()
        .map_err(|e| anyhow::anyhow!("Invalid CID: {}", e))?;

    let pb = progress::spinner(&format!("Retrieving tensor {}", cid));

    // Initialize storage
    let config = BlockStoreConfig::default();
    let store = SledBlockStore::new(config)?;

    // Retrieve block
    match store.get(&cid).await? {
        Some(block) => {
            progress::finish_spinner_success(&pb, "Tensor retrieved");

            let output_path = output.unwrap_or("tensor.safetensors");
            fs::write(output_path, block.data()).await?;

            success(&format!("Saved tensor to: {}", output_path));
            print_kv("CID", &cid.to_string());
            print_kv("Size", &format_bytes(block.size()));
            Ok(())
        }
        None => {
            progress::finish_spinner_error(&pb, "Tensor not found");
            error(&format!("Tensor not found: {}", cid));
            std::process::exit(1);
        }
    }
}

/// Show tensor metadata
pub async fn tensor_info(cid_str: &str, format: &str) -> Result<()> {
    use ipfrs_core::Cid;
    use ipfrs_storage::{BlockStoreConfig, BlockStoreTrait, SledBlockStore};

    // Parse CID
    let cid = cid_str
        .parse::<Cid>()
        .map_err(|e| anyhow::anyhow!("Invalid CID: {}", e))?;

    let pb = progress::spinner(&format!("Retrieving tensor metadata {}", cid));

    // Initialize storage
    let config = BlockStoreConfig::default();
    let store = SledBlockStore::new(config)?;

    // Retrieve block
    match store.get(&cid).await? {
        Some(block) => {
            progress::finish_spinner_success(&pb, "Tensor metadata retrieved");

            let tensor_info = extract_safetensors_metadata(block.data());

            match format {
                "json" => {
                    println!("{{");
                    println!("  \"cid\": \"{}\",", cid);
                    println!("  \"size\": {},", block.size());
                    if let Some(info) = tensor_info {
                        println!("  \"format\": \"safetensors\",");
                        println!("  \"num_tensors\": {},", info.num_tensors);
                        println!("  \"tensors\": [");
                        for (i, (name, shape, dtype)) in info.tensors.iter().enumerate() {
                            print!("    {{");
                            print!("\"name\": \"{}\", ", name);
                            print!("\"shape\": {:?}, ", shape);
                            print!("\"dtype\": \"{}\"", dtype);
                            print!("}}");
                            if i < info.tensors.len() - 1 {
                                println!(",");
                            } else {
                                println!();
                            }
                        }
                        println!("  ]");
                    } else {
                        println!("  \"format\": \"unknown\"");
                    }
                    println!("}}");
                }
                _ => {
                    print_header(&format!("Tensor: {}", cid));
                    print_kv("Size", &format_bytes(block.size()));
                    if let Some(info) = tensor_info {
                        print_kv("Format", "safetensors");
                        print_kv("Number of tensors", &info.num_tensors.to_string());
                        println!("\nTensors:");
                        for (name, shape, dtype) in &info.tensors {
                            println!("  {} {:?} ({})", name, shape, dtype);
                        }
                    } else {
                        print_kv("Format", "unknown (raw binary)");
                    }
                }
            }
            Ok(())
        }
        None => {
            progress::finish_spinner_error(&pb, "Tensor not found");
            error(&format!("Tensor not found: {}", cid));
            std::process::exit(1);
        }
    }
}

/// Export tensor to different format
pub async fn tensor_export(cid_str: &str, output_path: &str, target_format: &str) -> Result<()> {
    use ipfrs_core::Cid;
    use ipfrs_storage::{BlockStoreConfig, BlockStoreTrait, SledBlockStore};
    use tokio::fs;

    // Parse CID
    let cid = cid_str
        .parse::<Cid>()
        .map_err(|e| anyhow::anyhow!("Invalid CID: {}", e))?;

    let pb = progress::spinner(&format!("Exporting tensor to {}", target_format));

    // Initialize storage
    let config = BlockStoreConfig::default();
    let store = SledBlockStore::new(config)?;

    // Retrieve block
    match store.get(&cid).await? {
        Some(block) => {
            // For now, we just copy the data as-is
            // In a real implementation, we would convert between formats
            match target_format {
                "safetensors" | "numpy" | "pytorch" => {
                    fs::write(output_path, block.data()).await?;
                    progress::finish_spinner_success(&pb, "Tensor exported");

                    success(&format!("Exported tensor to {}", output_path));
                    print_kv("Format", target_format);
                    print_kv("Size", &format_bytes(block.size()));
                }
                _ => {
                    progress::finish_spinner_error(&pb, "Unsupported format");
                    error(&format!("Unsupported format: {}", target_format));
                    output::info("Supported formats: safetensors, numpy, pytorch");
                    std::process::exit(1);
                }
            }
            Ok(())
        }
        None => {
            progress::finish_spinner_error(&pb, "Tensor not found");
            error(&format!("Tensor not found: {}", cid));
            std::process::exit(1);
        }
    }
}
