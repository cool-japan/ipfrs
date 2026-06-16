//! Semantic search commands
//!
//! This module provides semantic search operations:
//! - `semantic_search` - Vector search
//! - `semantic_index` - Manual indexing
//! - `semantic_similar` - Find similar content
//! - `semantic_stats` - Index statistics
//! - `semantic_save` - Save semantic index
//! - `semantic_load` - Load semantic index

use anyhow::Result;

use crate::output::{self, print_cid, print_header, print_kv};
use crate::progress;

/// Generate a deterministic hash-based embedding vector for query text.
///
/// Uses `DefaultHasher` with (text, dimension_index) as inputs so the same
/// text always produces the same normalised unit vector.  This is intentionally
/// a stand-in until a real embedding model is wired in.
fn text_to_embedding(text: &str, dim: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut embedding = vec![0.0f32; dim];
    for (i, slot) in embedding.iter_mut().enumerate() {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        (i as u64).hash(&mut hasher);
        let hash_val = hasher.finish();
        *slot = (hash_val as f32 / u64::MAX as f32) * 2.0 - 1.0;
    }
    // L2-normalise
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut embedding {
            *x /= norm;
        }
    }
    embedding
}

/// Internal helper: run semantic search and return a `(cids, results_printed)`
/// pair.  The caller decides whether to also print the results.
///
/// Returns the list of CID strings that were found so the hybrid query can
/// optionally pipe them through the logic filter.
async fn semantic_query_inner(
    text: &str,
    top_k: usize,
    threshold: f32,
    json_output: bool,
    print_results: bool,
) -> Result<Vec<String>> {
    use ipfrs::{Node, NodeConfig, QueryFilter};
    use ipfrs_semantic::RouterConfig;

    let mut node = Node::new(NodeConfig::default().with_semantic(RouterConfig::default()))?;
    node.start().await?;

    // Default RouterConfig uses dimension 128
    let embedding = text_to_embedding(text, 128);

    let filter = QueryFilter {
        min_score: if threshold > 0.0 {
            Some(threshold)
        } else {
            None
        },
        max_score: None,
        max_results: Some(top_k),
        cid_prefix: None,
    };

    let results = match node.search_hybrid(&embedding, top_k, filter).await {
        Ok(r) => r,
        Err(_) => {
            output::warning(
                "Semantic index not initialized. Use 'ipfrs semantic index <cid>' to index content first.",
            );
            node.stop().await?;
            if json_output && print_results {
                println!("[]");
            }
            return Ok(Vec::new());
        }
    };

    node.stop().await?;

    if print_results {
        if json_output {
            println!("[");
            for (idx, result) in results.iter().enumerate() {
                let comma = if idx + 1 < results.len() { "," } else { "" };
                println!(
                    "  {{\"cid\": \"{}\", \"score\": {:.4}}}{}",
                    result.cid, result.score, comma
                );
            }
            println!("]");
        } else {
            print_header(&format!("Semantic search: \"{}\"", text));
            if threshold > 0.0 {
                println!(
                    "Found {} results (threshold: {:.2})",
                    results.len(),
                    threshold
                );
            } else {
                println!("Found {} results", results.len());
            }
            println!();
            for result in &results {
                println!("  CID: {} (score: {:.2})", result.cid, result.score);
            }
        }
    }

    let cids: Vec<String> = results.into_iter().map(|r| r.cid.to_string()).collect();
    Ok(cids)
}

/// Semantic similarity search: `ipfrs semantic query "<text>" --top-k 10`
///
/// Prints results and returns `()`.
pub async fn semantic_query(
    text: &str,
    top_k: usize,
    threshold: f32,
    json_output: bool,
) -> Result<()> {
    semantic_query_inner(text, top_k, threshold, json_output, true).await?;
    Ok(())
}

/// Semantic similarity search that also returns the matching CIDs.
///
/// Used by the hybrid query pipeline so that logic post-filtering can be
/// applied to the semantic result set without re-running the search.
pub async fn semantic_query_with_cids(
    text: &str,
    top_k: usize,
    threshold: f32,
    json_output: bool,
) -> Result<Vec<String>> {
    semantic_query_inner(text, top_k, threshold, json_output, true).await
}

/// Vector search
#[allow(dead_code)]
pub async fn semantic_search(query: &str, top_k: usize, format: &str) -> Result<()> {
    let pb = progress::spinner("Searching for similar content...");
    progress::finish_spinner_success(&pb, "Search initialization complete");

    // Note: Full implementation requires an embedding model to convert query text to vectors
    output::warning("Semantic search requires an embedding model (not yet configured)");

    match format {
        "json" => {
            println!("{{");
            println!("  \"query\": \"{}\",", query);
            println!("  \"top_k\": {},", top_k);
            println!("  \"status\": \"not_implemented\",");
            println!("  \"message\": \"Semantic search requires embedding model configuration\"");
            println!("}}");
        }
        _ => {
            print_header(&format!("Semantic Search: {}", query));
            println!("Query: {}", query);
            println!("Top K: {}", top_k);
            println!();
            println!("To enable semantic search:");
            println!("  1. Configure an embedding model in config.toml");
            println!("  2. Index your content with 'ipfrs semantic index <cid>'");
            println!("  3. Run your query again");
        }
    }

    Ok(())
}

/// Manual indexing
#[allow(dead_code)]
pub async fn semantic_index(cid: &str, metadata: Option<&str>) -> Result<()> {
    let pb = progress::spinner("Preparing to index content...");
    progress::finish_spinner_success(&pb, "Index preparation complete");

    // Note: Full implementation requires embedding extraction from content
    output::warning("Semantic indexing requires an embedding model (not yet configured)");

    print_cid("CID", cid);
    if let Some(meta) = metadata {
        println!("  Metadata: {}", meta);
    }

    println!();
    println!("To enable semantic indexing:");
    println!("  1. Configure an embedding model in config.toml");
    println!("  2. Ensure the content exists in IPFRS");
    println!("  3. Run indexing again to extract and store embeddings");

    Ok(())
}

/// Find similar content
#[allow(dead_code)]
pub async fn semantic_similar(cid: &str, top_k: usize, format: &str) -> Result<()> {
    let pb = progress::spinner("Preparing similarity search...");
    progress::finish_spinner_success(&pb, "Search preparation complete");

    // Note: Full implementation requires content retrieval and embedding extraction
    output::warning("Similarity search requires an embedding model (not yet configured)");

    match format {
        "json" => {
            println!("{{");
            println!("  \"cid\": \"{}\",", cid);
            println!("  \"top_k\": {},", top_k);
            println!("  \"status\": \"not_implemented\",");
            println!("  \"message\": \"Similarity search requires embedding model configuration\"");
            println!("}}");
        }
        _ => {
            print_header("Similarity Search");
            print_cid("Query CID", cid);
            println!("  Top K: {}", top_k);
            println!();
            println!("To enable similarity search:");
            println!("  1. Configure an embedding model in config.toml");
            println!("  2. Index your content with 'ipfrs semantic index'");
            println!("  3. Run similarity search again");
        }
    }

    Ok(())
}

/// Index statistics
#[allow(dead_code)]
pub async fn semantic_stats(format: &str) -> Result<()> {
    let pb = progress::spinner("Retrieving semantic index statistics...");
    progress::finish_spinner_success(&pb, "Statistics retrieved");

    output::warning("Semantic index not yet initialized");

    match format {
        "json" => {
            println!("{{");
            println!("  \"total_vectors\": 0,");
            println!("  \"index_size_bytes\": 0,");
            println!("  \"num_dimensions\": 0,");
            println!("  \"status\": \"not_initialized\"");
            println!("}}");
        }
        _ => {
            print_header("Semantic Index Statistics");
            print_kv("Total Vectors", "0");
            print_kv("Index Size", "0 B");
            print_kv("Status", "Not initialized");
            println!();
            println!("To initialize the semantic index:");
            println!("  1. Configure an embedding model");
            println!("  2. Index content with 'ipfrs semantic index <cid>'");
        }
    }

    Ok(())
}

/// Save semantic index
pub async fn semantic_save(path: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    println!("Saving semantic index to {}...", path);
    node.save_semantic_index(path).await?;
    println!("Semantic index saved successfully");

    node.stop().await?;
    Ok(())
}

/// Load semantic index
pub async fn semantic_load(path: &str) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    println!("Loading semantic index from {}...", path);
    node.load_semantic_index(path).await?;
    println!("Semantic index loaded successfully");

    node.stop().await?;
    Ok(())
}
