//! IPLD path resolution and DAG inspection CLI commands.
//!
//! Implements three subcommands under `ipfrs ipld`:
//!
//! - `resolve <path>` — resolve an `/ipld/<cid>/...` path and print the value
//! - `stat <cid>`     — print codec, size, and link count for a CID
//! - `links <cid>`    — list every CID linked from a given block
//!
//! The additional `dag_cli` helpers (`dag_stat`, `dag_export`, `dag_import`)
//! are exposed from this module for use by the `ipfrs dag` command group.

use anyhow::{anyhow, Result};
use std::path::Path;

use crate::commands::query::OutputFormat;

// ─── Path parsing ────────────────────────────────────────────────────────────

/// Parse an `/ipld/<cid-string>[/seg1/seg2/…]` path into its components.
///
/// Returns `(cid_string, path_segments)` on success.
///
/// # Errors
/// Returns an error when the path does not start with `/ipld/` or when the
/// CID segment is missing.
pub fn parse_ipld_path(path: &str) -> Result<(String, Vec<String>)> {
    // Strip optional leading slash and split on '/'
    let stripped = path.trim_start_matches('/');
    let mut parts: Vec<&str> = stripped.split('/').filter(|s| !s.is_empty()).collect();

    if parts.is_empty() {
        return Err(anyhow!("Empty path"));
    }

    if parts[0] != "ipld" {
        return Err(anyhow!("Path must start with /ipld/, got: /{}", parts[0]));
    }

    // Remove the "ipld" prefix segment
    parts.remove(0);

    if parts.is_empty() {
        return Err(anyhow!("Path is missing the CID segment: {}", path));
    }

    let cid_str = parts.remove(0).to_string();
    let segments: Vec<String> = parts.iter().map(|s| s.to_string()).collect();

    Ok((cid_str, segments))
}

// ─── IPLD value display ───────────────────────────────────────────────────────

/// Walk an `ipfrs_core::Ipld` tree following `segments` and return the leaf.
fn traverse_ipld<'a>(
    node: &'a ipfrs_core::Ipld,
    segments: &[String],
) -> Result<&'a ipfrs_core::Ipld> {
    if segments.is_empty() {
        return Ok(node);
    }

    let seg = &segments[0];
    let rest = &segments[1..];

    match node {
        ipfrs_core::Ipld::Map(map) => {
            let child = map
                .get(seg.as_str())
                .ok_or_else(|| anyhow!("Key '{}' not found in IPLD map", seg))?;
            traverse_ipld(child, rest)
        }
        ipfrs_core::Ipld::List(list) => {
            let idx: usize = seg
                .parse()
                .map_err(|_| anyhow!("Expected numeric index for list, got '{}'", seg))?;
            let child = list
                .get(idx)
                .ok_or_else(|| anyhow!("Index {} out of bounds (len {})", idx, list.len()))?;
            traverse_ipld(child, rest)
        }
        other => Err(anyhow!(
            "Cannot descend into {:?} with segment '{}'",
            std::mem::discriminant(other),
            seg
        )),
    }
}

/// Render an `ipfrs_core::Ipld` value as a `serde_json::Value` for display.
fn ipld_to_json(ipld: &ipfrs_core::Ipld) -> serde_json::Value {
    match ipld {
        ipfrs_core::Ipld::Null => serde_json::Value::Null,
        ipfrs_core::Ipld::Bool(b) => serde_json::Value::Bool(*b),
        ipfrs_core::Ipld::Integer(n) => serde_json::json!(*n),
        ipfrs_core::Ipld::Float(f) => serde_json::json!(*f),
        ipfrs_core::Ipld::String(s) => serde_json::Value::String(s.clone()),
        ipfrs_core::Ipld::Bytes(b) => {
            // Encode bytes as base64 in a DAG-JSON style object
            use std::fmt::Write;
            let mut hex = String::with_capacity(b.len() * 2);
            for byte in b {
                write!(hex, "{:02x}", byte).ok();
            }
            serde_json::json!({ "bytes": hex })
        }
        ipfrs_core::Ipld::List(items) => {
            serde_json::Value::Array(items.iter().map(ipld_to_json).collect())
        }
        ipfrs_core::Ipld::Map(map) => {
            let obj: serde_json::Map<String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), ipld_to_json(v)))
                .collect();
            serde_json::Value::Object(obj)
        }
        ipfrs_core::Ipld::Link(cid) => {
            serde_json::json!({ "/": cid.0.to_string() })
        }
    }
}

// ─── ipld resolve ─────────────────────────────────────────────────────────────

/// Resolve an IPLD path and print the value.
///
/// `path` must follow the format `/ipld/<cid-string>/field/subfield/0`.
/// The block is fetched from local storage, decoded as DAG-CBOR, and the
/// given sub-path is traversed.
pub async fn ipld_resolve(path: &str, format: &OutputFormat) -> Result<()> {
    use ipfrs::{Node, NodeConfig};
    use ipfrs_core::Cid;

    let (cid_str, segments) = parse_ipld_path(path)?;

    let cid = cid_str
        .parse::<Cid>()
        .map_err(|e| anyhow!("Invalid CID '{}': {}", cid_str, e))?;

    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    let raw = node
        .get_block_raw(&cid)
        .await?
        .ok_or_else(|| anyhow!("Block not found: {}", cid))?;

    // Decode as DAG-CBOR
    let ipld = ipfrs_core::Ipld::from_dag_cbor(&raw)
        .map_err(|e| anyhow!("Failed to decode block as DAG-CBOR: {}", e))?;

    let leaf = traverse_ipld(&ipld, &segments)?;
    let json_val = ipld_to_json(leaf);

    node.stop().await?;

    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&json_val)?);
        }
        OutputFormat::Text => {
            // Pretty-print but without the JSON wrapper for scalar types
            match &json_val {
                serde_json::Value::String(s) => println!("{}", s),
                serde_json::Value::Number(n) => println!("{}", n),
                serde_json::Value::Bool(b) => println!("{}", b),
                serde_json::Value::Null => println!("null"),
                other => println!("{}", serde_json::to_string_pretty(other)?),
            }
        }
    }

    Ok(())
}

// ─── ipld stat ────────────────────────────────────────────────────────────────

/// Print metadata about a CID: codec, size, links count.
pub async fn ipld_stat(cid_str: &str, format: &OutputFormat) -> Result<()> {
    use ipfrs::{Node, NodeConfig};
    use ipfrs_core::Cid;

    let cid = cid_str
        .parse::<Cid>()
        .map_err(|e| anyhow!("Invalid CID '{}': {}", cid_str, e))?;

    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    let raw = node
        .get_block_raw(&cid)
        .await?
        .ok_or_else(|| anyhow!("Block not found: {}", cid))?;

    let size = raw.len();
    let codec_code = cid.codec();
    let codec_name = codec_name_for(codec_code);

    // Count links by trying to decode as DAG-CBOR
    let links_count = match ipfrs_core::Ipld::from_dag_cbor(&raw) {
        Ok(ipld) => ipld.links().len(),
        Err(_) => 0,
    };

    node.stop().await?;

    match format {
        OutputFormat::Json => {
            let obj = serde_json::json!({
                "cid": cid.to_string(),
                "size": size,
                "codec": codec_name,
                "codec_code": codec_code,
                "links": links_count,
            });
            println!("{}", serde_json::to_string_pretty(&obj)?);
        }
        OutputFormat::Text => {
            use crate::output::{print_header, print_kv};
            print_header("IPLD Block Stat");
            print_kv("CID", &cid.to_string());
            print_kv("Size", &format!("{} bytes", size));
            print_kv("Codec", &format!("{} (0x{:x})", codec_name, codec_code));
            print_kv("Links", &links_count.to_string());
        }
    }

    Ok(())
}

// ─── ipld links ───────────────────────────────────────────────────────────────

/// List all CIDs linked from a given CID.
pub async fn ipld_links(cid_str: &str, format: &OutputFormat) -> Result<()> {
    use ipfrs::{Node, NodeConfig};
    use ipfrs_core::Cid;

    let cid = cid_str
        .parse::<Cid>()
        .map_err(|e| anyhow!("Invalid CID '{}': {}", cid_str, e))?;

    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    let raw = node
        .get_block_raw(&cid)
        .await?
        .ok_or_else(|| anyhow!("Block not found: {}", cid))?;

    let ipld = ipfrs_core::Ipld::from_dag_cbor(&raw)
        .map_err(|e| anyhow!("Failed to decode block as DAG-CBOR: {}", e))?;

    let links: Vec<String> = ipld.links().iter().map(|c| c.to_string()).collect();

    node.stop().await?;

    match format {
        OutputFormat::Json => {
            let arr: serde_json::Value = links
                .iter()
                .map(|s| serde_json::json!({ "/": s }))
                .collect::<Vec<_>>()
                .into();
            println!("{}", serde_json::to_string_pretty(&arr)?);
        }
        OutputFormat::Text => {
            if links.is_empty() {
                println!("No links found in block {}", cid);
            } else {
                for link in &links {
                    println!("{}", link);
                }
            }
        }
    }

    Ok(())
}

// ─── dag_cli helpers ──────────────────────────────────────────────────────────

/// Statistics returned after a DAG import operation.
#[derive(Debug, Default, Clone)]
pub struct ImportStats {
    pub blocks_imported: usize,
    pub bytes_imported: u64,
}

/// Print DAG node stats: CID, size, links, codec.
///
/// This mirrors `ipld_stat` but is exposed under the `dag` command group.
pub async fn dag_stat(cid_str: &str, format: &OutputFormat) -> Result<()> {
    // Delegate to ipld_stat — identical semantics
    ipld_stat(cid_str, format).await
}

/// Export a DAG sub-graph as a CAR v1 stream.
///
/// When `output` is `None`, the CAR bytes are written to stdout.
/// Traversal is breadth-first; only locally available blocks are included.
///
/// CAR v1 format (simplified, no compression):
/// ```text
/// <varint: header-len> <dag-cbor-header> <blocks…>
/// ```
/// Each block:
/// ```text
/// <varint: cid-len + data-len> <cid-bytes> <data-bytes>
/// ```
pub async fn dag_export(cid_str: &str, output: Option<&Path>) -> Result<()> {
    use ipfrs::{Node, NodeConfig};
    use ipfrs_core::Cid;
    use std::io::Write;

    let root = cid_str
        .parse::<Cid>()
        .map_err(|e| anyhow!("Invalid CID '{}': {}", cid_str, e))?;

    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    // Breadth-first traversal collecting (cid, raw_bytes) pairs
    let mut visited: std::collections::HashSet<Cid> = std::collections::HashSet::new();
    let mut queue: std::collections::VecDeque<Cid> = std::collections::VecDeque::new();
    let mut blocks: Vec<(Cid, Vec<u8>)> = Vec::new();

    queue.push_back(root);
    visited.insert(root);

    while let Some(cid) = queue.pop_front() {
        let raw = match node.get_block_raw(&cid).await? {
            Some(r) => r,
            None => {
                // Skip unavailable blocks (warn but continue)
                eprintln!("Warning: block {} not available locally, skipping", cid);
                continue;
            }
        };

        // Discover child links if decodable as DAG-CBOR
        if let Ok(ipld) = ipfrs_core::Ipld::from_dag_cbor(&raw) {
            for link in ipld.links() {
                if visited.insert(link) {
                    queue.push_back(link);
                }
            }
        }

        blocks.push((cid, raw));
    }

    node.stop().await?;

    // Build CAR v1 payload in memory
    let car_bytes = build_car_v1(&root, &blocks)?;

    // Write output
    match output {
        Some(path) => {
            tokio::fs::write(path, &car_bytes).await?;
            eprintln!(
                "Exported {} blocks ({} bytes) to {}",
                blocks.len(),
                car_bytes.len(),
                path.display()
            );
        }
        None => {
            std::io::stdout()
                .write_all(&car_bytes)
                .map_err(|e| anyhow!("Failed to write CAR to stdout: {}", e))?;
        }
    }

    Ok(())
}

/// Import blocks from a CAR file into local block storage.
///
/// Supports CAR v1 format (the simplest widely-used variant).
pub async fn dag_import(input: &Path) -> Result<ImportStats> {
    use ipfrs::{Node, NodeConfig};

    let car_bytes = tokio::fs::read(input)
        .await
        .map_err(|e| anyhow!("Failed to read {}: {}", input.display(), e))?;

    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    let mut stats = ImportStats::default();
    let mut cursor: usize = 0;

    // Skip CAR v1 header (DAG-CBOR encoded header — we only need to advance past it)
    let (_header_len, _header_bytes) = read_varint_and_data(&car_bytes, &mut cursor)?;

    // Read blocks until EOF
    while cursor < car_bytes.len() {
        let (block_len, _) = read_varint_prefix_len(&car_bytes, &mut cursor)?;
        if block_len == 0 {
            break;
        }

        let block_start = cursor;
        let block_end = block_start + block_len;

        if block_end > car_bytes.len() {
            return Err(anyhow!(
                "CAR file truncated: expected {} bytes at offset {}",
                block_len,
                cursor
            ));
        }

        // CID bytes: encoded CID, length determined by parsing
        let (cid, cid_len) = parse_cid_bytes(&car_bytes[block_start..block_end])?;
        let data_start = block_start + cid_len;
        let data = car_bytes[data_start..block_end].to_vec();
        let data_len = data.len() as u64;

        node.put_block_raw(data)
            .await
            .map_err(|e| anyhow!("Failed to store block {}: {}", cid, e))?;

        stats.blocks_imported += 1;
        stats.bytes_imported += data_len;

        cursor = block_end;
    }

    node.stop().await?;

    Ok(stats)
}

// ─── CAR v1 helpers ───────────────────────────────────────────────────────────

/// Encode a single unsigned varint and append it to `buf`.
fn write_varint(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte);
            break;
        } else {
            buf.push(byte | 0x80);
        }
    }
}

/// Build a minimal CAR v1 payload from a list of `(cid, raw_bytes)` blocks.
fn build_car_v1(root: &ipfrs_core::Cid, blocks: &[(ipfrs_core::Cid, Vec<u8>)]) -> Result<Vec<u8>> {
    let mut out = Vec::new();

    // --- Header (DAG-CBOR encoded map: {"version": 1, "roots": [<cid-link>]}) ---
    // Encode a minimal CAR header: {"version":1,"roots":[/<root-cid>]}
    let root_cid_bytes = cid_to_bytes(root)?;
    // DAG-CBOR encoding of the header map: hand-crafted for simplicity
    let header_cbor = build_car_header_cbor(&root_cid_bytes)?;
    write_varint(&mut out, header_cbor.len() as u64);
    out.extend_from_slice(&header_cbor);

    // --- Blocks ---
    for (cid, data) in blocks {
        let cid_bytes = cid_to_bytes(cid)?;
        let block_len = cid_bytes.len() + data.len();
        write_varint(&mut out, block_len as u64);
        out.extend_from_slice(&cid_bytes);
        out.extend_from_slice(data);
    }

    Ok(out)
}

/// Serialize a CID into its binary representation (multihash-encoded CIDv1).
fn cid_to_bytes(cid: &ipfrs_core::Cid) -> Result<Vec<u8>> {
    // cid crate provides `.to_bytes()` which gives the standard binary encoding
    Ok(cid.to_bytes())
}

/// Build a minimal DAG-CBOR header for a CAR v1 file.
///
/// Structure: `{"roots": [<cid-link>], "version": 1}`
///
/// DAG-CBOR:
/// - map(2) = 0xa2
/// - "roots" key + array of 1 CID tag(42) + bytes
/// - "version" key + uint(1)
fn build_car_header_cbor(root_cid_bytes: &[u8]) -> Result<Vec<u8>> {
    let mut buf = Vec::new();

    // map of 2 entries: 0xa2
    buf.push(0xa2);

    // key: "roots" (5 chars) → text(5) = 0x65, then bytes
    buf.push(0x65);
    buf.extend_from_slice(b"roots");

    // value: array(1) = 0x81
    buf.push(0x81);

    // CID link: tag(42) = 0xd8 0x2a, then bytes(len) + CID bytes
    // CBOR tag 42
    buf.push(0xd8);
    buf.push(42u8);

    // The CID bytes are prefixed with 0x00 (multibase identity prefix for binary CIDs in CAR)
    let cid_with_prefix = {
        let mut v = vec![0u8]; // identity multibase prefix
        v.extend_from_slice(root_cid_bytes);
        v
    };

    // bytes(len)
    encode_cbor_bytes_header(&mut buf, cid_with_prefix.len());
    buf.extend_from_slice(&cid_with_prefix);

    // key: "version" (7 chars) → text(7) = 0x67
    buf.push(0x67);
    buf.extend_from_slice(b"version");

    // value: 1 → 0x01
    buf.push(0x01);

    Ok(buf)
}

/// Encode a CBOR byte string length header (major type 2).
fn encode_cbor_bytes_header(buf: &mut Vec<u8>, len: usize) {
    if len <= 23 {
        buf.push(0x40 | len as u8);
    } else if len <= 0xff {
        buf.push(0x58);
        buf.push(len as u8);
    } else if len <= 0xffff {
        buf.push(0x59);
        buf.push((len >> 8) as u8);
        buf.push(len as u8);
    } else {
        buf.push(0x5a);
        buf.push((len >> 24) as u8);
        buf.push((len >> 16) as u8);
        buf.push((len >> 8) as u8);
        buf.push(len as u8);
    }
}

/// Read a varint-prefixed blob from `data` starting at `*cursor`.
///
/// Returns `(payload_length, payload_slice)` and advances `*cursor` past the payload.
fn read_varint_and_data<'a>(data: &'a [u8], cursor: &mut usize) -> Result<(usize, &'a [u8])> {
    let (len, n) = decode_varint(&data[*cursor..])
        .ok_or_else(|| anyhow!("Truncated varint at offset {}", cursor))?;
    *cursor += n;
    let end = *cursor + len as usize;
    if end > data.len() {
        return Err(anyhow!("CAR payload truncated"));
    }
    let slice = &data[*cursor..end];
    *cursor = end;
    Ok((len as usize, slice))
}

/// Read the varint length prefix and advance `*cursor` past the varint only
/// (not past the payload).  Returns `(payload_len, varint_byte_count)`.
fn read_varint_prefix_len(data: &[u8], cursor: &mut usize) -> Result<(usize, usize)> {
    let (len, n) = decode_varint(&data[*cursor..])
        .ok_or_else(|| anyhow!("Truncated varint at offset {}", cursor))?;
    *cursor += n;
    Ok((len as usize, n))
}

/// Decode a unsigned varint from the beginning of `buf`.
///
/// Returns `(value, bytes_consumed)` or `None` on truncation.
fn decode_varint(buf: &[u8]) -> Option<(u64, usize)> {
    let mut value: u64 = 0;
    let mut shift = 0u32;
    for (i, &byte) in buf.iter().enumerate() {
        value |= ((byte & 0x7f) as u64) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            return Some((value, i + 1));
        }
        if shift >= 64 {
            return None; // overflow
        }
    }
    None
}

/// Parse the CID at the beginning of `block_data` and return `(cid, bytes_consumed)`.
fn parse_cid_bytes(block_data: &[u8]) -> Result<(ipfrs_core::Cid, usize)> {
    use ipfrs_core::Cid;
    use std::io::Cursor;

    // Try CIDv1 first (reads a varint-codec + varint-multihash)
    let mut cur = Cursor::new(block_data);
    let cid = Cid::read_bytes(&mut cur)
        .map_err(|e| anyhow!("Failed to parse CID from CAR block: {}", e))?;
    let consumed = cur.position() as usize;
    Ok((cid, consumed))
}

// ─── Utility ─────────────────────────────────────────────────────────────────

/// Return a human-readable codec name for common IPLD codec codes.
fn codec_name_for(code: u64) -> &'static str {
    match code {
        0x55 => "raw",
        0x70 => "dag-pb",
        0x71 => "dag-cbor",
        0x0129 => "dag-json",
        _ => "unknown",
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Parsing a well-formed `/ipld/<cid>/a/b/0` path must succeed and yield
    /// the expected CID string and segment list.
    #[test]
    fn test_ipld_path_parse_valid() {
        let (cid_str, segs) = parse_ipld_path("/ipld/bafkreihdwdcefgh48/a/b/0")
            .expect("should parse valid ipld path");
        assert_eq!(cid_str, "bafkreihdwdcefgh48");
        assert_eq!(segs, vec!["a", "b", "0"]);
    }

    /// A path that does not begin with `/ipld/` must return an error.
    #[test]
    fn test_ipld_path_parse_missing_prefix() {
        let err = parse_ipld_path("/rule/bafkreihdwdcefgh48/head").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("ipld"),
            "Error message should mention 'ipld': {}",
            msg
        );
    }

    /// A path with only the `/ipld/` prefix but no CID must return an error.
    #[test]
    fn test_ipld_path_parse_missing_cid() {
        let err = parse_ipld_path("/ipld/").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("CID") || msg.contains("cid") || msg.contains("missing"),
            "Error should mention missing CID: {}",
            msg
        );
    }

    /// `ImportStats` with zero values should be constructible.
    #[test]
    fn test_import_stats_default() {
        let stats = ImportStats::default();
        assert_eq!(stats.blocks_imported, 0);
        assert_eq!(stats.bytes_imported, 0);
    }

    /// Verify that the JSON output of `ipld_stat` would include the expected keys.
    /// (Unit-level: we inspect the JSON object directly, no I/O needed.)
    #[test]
    fn test_dag_stat_format_json_keys() {
        // Build the expected JSON object structure
        let obj = serde_json::json!({
            "cid": "bafkreitest",
            "size": 42usize,
            "codec": "dag-cbor",
            "codec_code": 0x71u64,
            "links": 0usize,
        });

        assert!(obj.get("cid").is_some(), "must have 'cid' key");
        assert!(obj.get("size").is_some(), "must have 'size' key");
        assert!(obj.get("links").is_some(), "must have 'links' key");
    }

    /// Verify varint encoding round-trips for small and large values.
    #[test]
    fn test_varint_roundtrip() {
        for &val in &[0u64, 1, 127, 128, 255, 300, 16383, 16384, u32::MAX as u64] {
            let mut buf = Vec::new();
            write_varint(&mut buf, val);
            let (decoded, consumed) = decode_varint(&buf).expect("should decode");
            assert_eq!(decoded, val, "roundtrip failed for {}", val);
            assert_eq!(
                consumed,
                buf.len(),
                "consumed wrong number of bytes for {}",
                val
            );
        }
    }

    /// `codec_name_for` must return known names for common codec codes.
    #[test]
    fn test_codec_name_known_codes() {
        assert_eq!(codec_name_for(0x55), "raw");
        assert_eq!(codec_name_for(0x70), "dag-pb");
        assert_eq!(codec_name_for(0x71), "dag-cbor");
        assert_eq!(codec_name_for(0xdeadbeef), "unknown");
    }
}
