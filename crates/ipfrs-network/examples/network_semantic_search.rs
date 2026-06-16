//! Semantic DHT example
//!
//! This example demonstrates how to use the semantic DHT for vector-based content discovery.
//!
//! Run with:
//! ```bash
//! cargo run --example semantic_search
//! ```

use ipfrs_network::{
    DistanceMetric, LshConfig, NamespaceId, SemanticDht, SemanticDhtConfig, SemanticNamespace,
    SemanticQuery,
};
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== IPFRS Semantic DHT Example ===\n");

    // 1. Create semantic DHT with configuration
    println!("1. Creating semantic DHT...");
    let config = SemanticDhtConfig {
        lsh_hash_functions: 8,
        lsh_hash_tables: 4,
        lsh_bucket_width: 4.0,
        max_query_peers: 20,
        query_timeout: Duration::from_secs(10),
        enable_caching: true,
        cache_ttl: Duration::from_secs(300),
        max_cache_size: 1000,
        top_k: 10,
        ..Default::default()
    };

    let semantic_dht = SemanticDht::new(config);
    println!("   ✓ Semantic DHT created\n");

    // 2. Register namespaces for different embedding types
    println!("2. Registering namespaces...");

    // Text embedding namespace (768 dimensions, typical for BERT)
    let text_namespace = SemanticNamespace {
        id: NamespaceId::text(),
        dimension: 768,
        distance_metric: DistanceMetric::Cosine,
        lsh_config: LshConfig::default(),
    };
    semantic_dht.register_namespace(text_namespace)?;
    println!("   ✓ Registered 'text' namespace (768-dim, Cosine)");

    // Image embedding namespace (512 dimensions, typical for ResNet)
    let image_namespace = SemanticNamespace {
        id: NamespaceId::image(),
        dimension: 512,
        distance_metric: DistanceMetric::Euclidean,
        lsh_config: LshConfig::default(),
    };
    semantic_dht.register_namespace(image_namespace)?;
    println!("   ✓ Registered 'image' namespace (512-dim, Euclidean)");

    // Custom namespace for specialized embeddings
    let custom_namespace = SemanticNamespace {
        id: NamespaceId::new("scientific-papers"),
        dimension: 1024,
        distance_metric: DistanceMetric::DotProduct,
        lsh_config: LshConfig {
            hash_functions: 12,
            num_tables: 6,
            bucket_width: 2.0,
        },
    };
    semantic_dht.register_namespace(custom_namespace)?;
    println!("   ✓ Registered 'scientific-papers' namespace (1024-dim, DotProduct)\n");

    // 3. Index content with embeddings
    println!("3. Indexing content...");

    // Simulate text document embeddings
    for i in 0..10 {
        let embedding = generate_mock_embedding(768, i as f32 * 0.1);
        let cid = cid::Cid::default(); // In real use, would be actual content CID

        semantic_dht.index_content(cid, embedding, NamespaceId::text())?;
    }
    println!("   ✓ Indexed 10 text documents");

    // Simulate image embeddings
    for i in 0..5 {
        let embedding = generate_mock_embedding(512, i as f32 * 0.2);
        let cid = cid::Cid::default();

        semantic_dht.index_content(cid, embedding, NamespaceId::image())?;
    }
    println!("   ✓ Indexed 5 images\n");

    // 4. Compute LSH hashes
    println!("4. Computing LSH hashes for query embedding...");
    let query_embedding = generate_mock_embedding(768, 0.25);
    let hashes = semantic_dht.compute_lsh_hashes(&query_embedding, &NamespaceId::text())?;

    println!("   Generated {} LSH hashes:", hashes.len());
    for (i, hash) in hashes.iter().enumerate() {
        let cid = hash.to_cid();
        println!("   Table {}: {:?} → {}", i, &hash.bucket[..3], cid);
    }
    println!();

    // 5. Execute semantic queries
    println!("5. Executing semantic queries...");

    // Query 1: Find similar text documents
    let text_query = SemanticQuery {
        embedding: query_embedding.clone(),
        namespace: NamespaceId::text(),
        top_k: 5,
        metadata_filter: None,
        timeout: Duration::from_secs(5),
    };

    let results = semantic_dht.query(text_query)?;
    println!("   Text query results (top {}):", results.len());
    for (i, result) in results.iter().enumerate() {
        println!(
            "   {}. CID: {} | Score: {:.4} | Peer: {}",
            i + 1,
            result.cid,
            result.score,
            result.peer
        );
    }
    println!();

    // Query 2: Find similar images
    let image_query_embedding = generate_mock_embedding(512, 0.3);
    let image_query = SemanticQuery {
        embedding: image_query_embedding,
        namespace: NamespaceId::image(),
        top_k: 3,
        metadata_filter: None,
        timeout: Duration::from_secs(5),
    };

    let results = semantic_dht.query(image_query)?;
    println!("   Image query results (top {}):", results.len());
    for (i, result) in results.iter().enumerate() {
        println!(
            "   {}. CID: {} | Score: {:.4}",
            i + 1,
            result.cid,
            result.score
        );
    }
    println!();

    // 6. Demonstrate caching
    println!("6. Testing query caching...");

    // First query (cache miss)
    let query1 = SemanticQuery {
        embedding: query_embedding.clone(),
        namespace: NamespaceId::text(),
        top_k: 5,
        metadata_filter: None,
        timeout: Duration::from_secs(5),
    };
    let _ = semantic_dht.query(query1)?;

    // Second query (cache hit)
    let query2 = SemanticQuery {
        embedding: query_embedding.clone(),
        namespace: NamespaceId::text(),
        top_k: 5,
        metadata_filter: None,
        timeout: Duration::from_secs(5),
    };
    let _ = semantic_dht.query(query2)?;

    let stats = semantic_dht.stats();
    println!("   Cache hits: {}", stats.cache_hits);
    println!("   Cache misses: {}", stats.cache_misses);
    println!(
        "   Cache hit rate: {:.1}%\n",
        (stats.cache_hits as f64 / stats.total_queries as f64) * 100.0
    );

    // 7. Display statistics
    println!("7. Semantic DHT Statistics:");
    println!("   Total queries: {}", stats.total_queries);
    println!("   Successful: {}", stats.successful_queries);
    println!("   Failed: {}", stats.failed_queries);
    println!("   Indexed content: {}", stats.indexed_content);
    println!("   Avg query latency: {:.2}ms", stats.avg_query_latency_ms);

    println!("\n   Queries per namespace:");
    for (namespace, count) in &stats.queries_per_namespace {
        println!("   - {}: {}", namespace, count);
    }
    println!();

    // 8. List all namespaces
    println!("8. Registered namespaces:");
    let namespaces = semantic_dht.list_namespaces();
    for ns_id in namespaces {
        if let Some(ns) = semantic_dht.get_namespace(&ns_id) {
            println!(
                "   - {} ({}-dim, {:?})",
                ns.id.0, ns.dimension, ns.distance_metric
            );
        }
    }

    println!("\n=== Example Complete ===");

    Ok(())
}

/// Generate a mock embedding for demonstration purposes
/// In real applications, embeddings would come from ML models (BERT, ResNet, etc.)
fn generate_mock_embedding(dim: usize, seed: f32) -> Vec<f32> {
    (0..dim).map(|i| ((i as f32 + seed) * 0.1).sin()).collect()
}

/// Example: Using different distance metrics
#[allow(dead_code)]
fn distance_metric_examples() -> Result<(), Box<dyn std::error::Error>> {
    let semantic_dht = SemanticDht::new(SemanticDhtConfig::default());

    // Euclidean distance - good for spatial data
    let euclidean_ns = SemanticNamespace {
        id: NamespaceId::new("spatial"),
        dimension: 128,
        distance_metric: DistanceMetric::Euclidean,
        lsh_config: LshConfig::default(),
    };
    semantic_dht.register_namespace(euclidean_ns)?;

    // Cosine distance - good for text embeddings
    let cosine_ns = SemanticNamespace {
        id: NamespaceId::text(),
        dimension: 768,
        distance_metric: DistanceMetric::Cosine,
        lsh_config: LshConfig::default(),
    };
    semantic_dht.register_namespace(cosine_ns)?;

    // Manhattan distance - robust to outliers
    let manhattan_ns = SemanticNamespace {
        id: NamespaceId::new("robust"),
        dimension: 256,
        distance_metric: DistanceMetric::Manhattan,
        lsh_config: LshConfig::default(),
    };
    semantic_dht.register_namespace(manhattan_ns)?;

    // Dot product - for normalized vectors
    let dotprod_ns = SemanticNamespace {
        id: NamespaceId::new("normalized"),
        dimension: 512,
        distance_metric: DistanceMetric::DotProduct,
        lsh_config: LshConfig::default(),
    };
    semantic_dht.register_namespace(dotprod_ns)?;

    Ok(())
}

/// Example: Custom LSH configuration for different use cases
#[allow(dead_code)]
fn lsh_configuration_examples() -> Result<(), Box<dyn std::error::Error>> {
    let semantic_dht = SemanticDht::new(SemanticDhtConfig::default());

    // High precision (more hash functions)
    let high_precision = SemanticNamespace {
        id: NamespaceId::new("high-precision"),
        dimension: 512,
        distance_metric: DistanceMetric::Cosine,
        lsh_config: LshConfig {
            hash_functions: 16, // More functions = better precision
            num_tables: 4,
            bucket_width: 2.0, // Smaller width = finer granularity
        },
    };
    semantic_dht.register_namespace(high_precision)?;

    // High recall (more hash tables)
    let high_recall = SemanticNamespace {
        id: NamespaceId::new("high-recall"),
        dimension: 512,
        distance_metric: DistanceMetric::Cosine,
        lsh_config: LshConfig {
            hash_functions: 8,
            num_tables: 8, // More tables = better recall
            bucket_width: 4.0,
        },
    };
    semantic_dht.register_namespace(high_recall)?;

    // Fast search (fewer hash functions and tables)
    let fast_search = SemanticNamespace {
        id: NamespaceId::new("fast"),
        dimension: 512,
        distance_metric: DistanceMetric::Cosine,
        lsh_config: LshConfig {
            hash_functions: 4, // Fewer = faster
            num_tables: 2,
            bucket_width: 8.0, // Coarser = fewer buckets
        },
    };
    semantic_dht.register_namespace(fast_search)?;

    Ok(())
}
