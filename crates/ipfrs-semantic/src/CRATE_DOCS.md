# IPFRS Semantic - Vector Search and Semantic Routing

This crate provides high-performance semantic search and routing capabilities for IPFRS,
enabling content discovery based on vector embeddings and semantic similarity.

## Features

- **HNSW-based Vector Search** - Fast approximate nearest neighbor search
- **Semantic Routing** - Content discovery based on embeddings
- **Hybrid Search** - Combine vector search with metadata filtering
- **Vector Quantization** - Memory-efficient index compression (PQ, OPQ, Scalar)
- **Learned Indices** - ML-based indexing with Recursive Model Index (RMI)
- **Logic Integration** - TensorLogic reasoning with embeddings
- **DiskANN** - Disk-based indexing for 100M+ vectors
- **SIMD Optimization** - ARM NEON and x86 SSE/AVX acceleration
- **Caching** - Hot embedding cache with LRU eviction
- **Batch Query Processing** - Parallel batch queries for high throughput
- **Query Re-ranking** - Multi-criteria result re-ranking with weighted scoring
- **Query Analytics** - Performance tracking and query pattern analysis
- **Multi-Modal Search** - Unified search across text, image, audio, video, and code
- **Differential Privacy** - Privacy-preserving embeddings with configurable privacy budgets
- **Dynamic Updates** - Online embedding updates and version migration
- **Vector Quality Analysis** - Data validation, anomaly detection, and quality metrics (NEW!)
- **Index Diagnostics** - Health monitoring, performance profiling, and issue detection (NEW!)
- **Index Optimization** - Automatic parameter tuning and resource management (NEW!)
- **Auto-Scaling Advisor** - Intelligent scaling recommendations for production deployments (NEW!)

## Quick Start

### Basic Semantic Search

```rust
use ipfrs_semantic::{SemanticRouter, RouterConfig};
use ipfrs_core::Cid;

# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
// Create a semantic router with default configuration
let router = SemanticRouter::with_defaults()?;

// Index content with embeddings (typically from a model like BERT, CLIP, etc.)
let cid1: Cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi".parse()?;
let embedding1 = vec![0.1, 0.2, 0.3]; // 768-dim embedding in real use

// Add to index
router.add(&cid1, &vec![0.5; 768])?;

// Query for similar content
let query_embedding = vec![0.5; 768];
let results = router.query(&query_embedding, 10).await?;

for result in results {
    println!("CID: {}, Score: {}", result.cid, result.score);
}
# Ok(())
# }
```

### Batch Query for High Throughput

```rust
use ipfrs_semantic::{SemanticRouter, RouterConfig};
use ipfrs_core::Cid;

# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
// Create a semantic router
let router = SemanticRouter::with_defaults()?;

// Index multiple items
let items = vec![
    ("bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi".parse::<Cid>()?, vec![0.1; 768]),
    ("bafybeihpjhkeuiq3k6nqa3fkgeigeri7iebtrsuyuey5y6vy36n345xmbi".parse::<Cid>()?, vec![0.2; 768]),
    ("bafybeif2pall7dybz7vecqka3zo24irdwabwdi4wc55jznaq75q7eaavvu".parse::<Cid>()?, vec![0.3; 768]),
];

router.add_batch(&items)?;

// Batch query - process multiple queries in parallel
let query_embeddings = vec![
    vec![0.15; 768],
    vec![0.25; 768],
    vec![0.35; 768],
];

// More efficient than querying one by one
let batch_results = router.query_batch(&query_embeddings, 10).await?;

for (i, results) in batch_results.iter().enumerate() {
    println!("Query {} found {} results", i, results.len());
}

// Get batch statistics
let stats = router.batch_stats(&batch_results);
println!("Total queries: {}", stats.total_queries);
println!("Avg results per query: {:.2}", stats.avg_results_per_query);
println!("Avg score: {:.4}", stats.avg_score);
# Ok(())
# }
```

### Hybrid Search with Metadata Filtering

```rust
use ipfrs_semantic::{HybridIndex, HybridConfig, HybridQuery, Metadata, MetadataValue, MetadataFilter};
use ipfrs_core::Cid;

# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
// Create hybrid index
let config = HybridConfig::default();
let index = HybridIndex::new(config)?;

// Index content with metadata
let cid: Cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi".parse()?;
let embedding = vec![0.5; 768];

let mut metadata = Metadata::new();
metadata.set("type", MetadataValue::String("image".to_string()));
metadata.set("size", MetadataValue::Integer(1024));

index.insert(&cid, &embedding, Some(metadata))?;

// Query with filters using builder pattern
let filter = MetadataFilter::eq("type", MetadataValue::String("image".to_string()));
let query = HybridQuery::knn(vec![0.5; 768], 10)
    .with_filter(filter);

let response = index.search(query).await?;
println!("Found {} results", response.results.len());
# Ok(())
# }
```

### Vector Quantization for Memory Efficiency

```rust
use ipfrs_semantic::{ProductQuantizer, ScalarQuantizer};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
// Create Product Quantizer (8-32x compression)
let dimension = 768;
let num_subspaces = 8;
let bits_per_subspace = 8;

let mut pq = ProductQuantizer::new(dimension, num_subspaces, bits_per_subspace)?;

// Train on representative data (1000 training samples, max 10 iterations)
let training_data: Vec<Vec<f32>> = vec![vec![0.5; 768]; 1000];
pq.train(&training_data, 10)?;

// Quantize embeddings
let embedding = vec![0.5; 768];
let quantized = pq.quantize(&embedding)?;

println!("Original size: {} bytes", dimension * 4);
println!("Quantized size: {} bytes", quantized.codes.len());
# Ok(())
# }
```

### DiskANN for Large-Scale Indexing

```rust,no_run
use ipfrs_semantic::{DiskANNIndex, DiskANNConfig};
use ipfrs_core::Cid;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
// Create DiskANN index for 100M+ vectors
let config = DiskANNConfig {
    dimension: 768,
    max_degree: 32,
    ..Default::default()
};

let mut index = DiskANNIndex::new(config);
index.create("/path/to/diskann_index")?;

// Insert vectors (stored on disk, not in RAM)
let cid: Cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi".parse()?;
let embedding = vec![0.5; 768];
index.insert(&cid, &embedding)?;

// Search with constant memory usage
let results = index.search(&embedding, 10)?;
println!("Found {} results from disk", results.len());
# Ok(())
# }
```

### Learned Index Structures

```rust
use ipfrs_semantic::{LearnedIndex, RMIConfig, ModelType};
use ipfrs_core::Cid;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
// Create a learned index with Recursive Model Index (RMI)
let config = RMIConfig {
    num_models: 10,              // Number of second-stage models
    model_type: ModelType::Linear, // Linear, Polynomial, or NeuralNetwork
    training_iterations: 100,
    learning_rate: 0.01,
    error_threshold: 0.05,
};

let mut index = LearnedIndex::new(config);

// Add embeddings - the index learns data distribution
for i in 0..1000 {
    let cid = Cid::default();
    let embedding = vec![i as f32 / 1000.0; 768];
    index.add(cid, embedding)?;
}

// The index automatically rebuilds and trains models
let query = vec![0.5; 768];
let results = index.search(&query, 10)?;

// Check statistics
let stats = index.stats();
println!("Indexed {} points using {} models",
         stats.data_points, stats.num_models);
# Ok(())
# }
```

### TensorLogic Integration

```rust,no_run
use ipfrs_semantic::{LogicSolver, SolverConfig};
use ipfrs_tensorlogic::{Predicate, Term, Constant};
use ipfrs_core::Cid;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
// Create a logic solver with semantic similarity
let config = SolverConfig {
    max_depth: 100,
    similarity_threshold: 0.8,
    top_k_similar: 10,
    embedding_dim: 384,
    detect_cycles: true,
};

let mut solver = LogicSolver::new(config)?;

// Add facts to the knowledge base
let cid1: Cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi".parse()?;
let fact1 = Predicate::new("likes".to_string(), vec![
    Term::Const(Constant::String("alice".to_string())),
    Term::Const(Constant::String("rust".to_string())),
]);
solver.add_fact(fact1, cid1)?;

let cid2: Cid = "bafybeihpjhkeuiq3k6nqa3fkgeigeri7iebtrsuyuey5y6vy36n345xmbi".parse()?;
let fact2 = Predicate::new("likes".to_string(), vec![
    Term::Const(Constant::String("bob".to_string())),
    Term::Const(Constant::String("python".to_string())),
]);
solver.add_fact(fact2, cid2)?;

// Add a rule for similarity-based matching
// Rule: similar(X, Y) :- likes(X, Lang), likes(Y, Lang)
let head = Predicate::new("similar".to_string(), vec![
    Term::Var("X".to_string()),
    Term::Var("Y".to_string())
]);
let body = vec![
    Predicate::new("likes".to_string(), vec![
        Term::Var("X".to_string()),
        Term::Var("Lang".to_string())
    ]),
    Predicate::new("likes".to_string(), vec![
        Term::Var("Y".to_string()),
        Term::Var("Lang".to_string())
    ]),
];
solver.add_rule(head, body)?;

// Query using semantic similarity
let query = Predicate::new("likes".to_string(), vec![
    Term::Var("Who".to_string()),
    Term::Const(Constant::String("rust".to_string())),
]);

let results = solver.query(&query)?;
for substitution in results {
    println!("Found substitution: {:?}", substitution);
}

// Get solver statistics
let stats = solver.stats();
println!("Total facts: {}", stats.num_facts);
println!("Total rules: {}", stats.num_rules);
println!("Indexed predicates: {}", stats.num_indexed_predicates);
# Ok(())
# }
```

### Custom Embedding Model Integration

```rust,no_run
use ipfrs_semantic::{SemanticRouter, RouterConfig, DistanceMetric};
use ipfrs_core::Cid;

# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
// Configure router for your embedding model
let config = RouterConfig {
    dimension: 768,  // BERT-base dimension
    metric: DistanceMetric::Cosine,
    max_connections: 16,
    ef_construction: 200,
    ef_search: 50,
    cache_size: 1000,
    ..RouterConfig::default()
};

let router = SemanticRouter::new(config)?;

// Function to generate embeddings from your model
// This is a placeholder - replace with your actual model
fn generate_embedding(text: &str) -> Vec<f32> {
    // Example: Use sentence-transformers, Hugging Face transformers, etc.
    // let model = SentenceTransformer::new("all-MiniLM-L6-v2");
    // model.encode(text)

    // For this example, just return a dummy embedding
    vec![0.5; 768]
}

// Index documents with embeddings
let documents = vec![
    ("Rust programming language", "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"),
    ("Python machine learning", "bafybeihpjhkeuiq3k6nqa3fkgeigeri7iebtrsuyuey5y6vy36n345xmbi"),
    ("Distributed systems", "bafybeif2pall7dybz7vecqka3zo24irdwabwdi4wc55jznaq75q7eaavvu"),
];

for (text, cid_str) in documents {
    let cid: Cid = cid_str.parse()?;
    let embedding = generate_embedding(text);
    router.add(&cid, &embedding)?;
}

// Query with natural language
let query_text = "rust systems programming";
let query_embedding = generate_embedding(query_text);
let results = router.query(&query_embedding, 5).await?;

println!("Top results for '{}':", query_text);
for result in results {
    println!("  CID: {}, Score: {:.3}", result.cid, result.score);
}
# Ok(())
# }
```

### Distributed Semantic Search

For large-scale deployments across multiple nodes:

```rust,no_run
use ipfrs_semantic::{SemanticDHTNode, SemanticDHTConfig, VectorIndex, DistanceMetric};
use ipfrs_network::libp2p::PeerId;
use ipfrs_core::Cid;

# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
// Configure distributed semantic DHT
let config = SemanticDHTConfig {
    embedding_dim: 768,
    replication_factor: 3,     // Replicate to 3 peers
    routing_table_size: 20,    // Top 20 nearest peers
    distance_metric: DistanceMetric::Cosine,
    max_hops: 5,               // Maximum query propagation hops
    query_timeout_ms: 5000,    // 5 second timeout
};

// Create local vector index
let local_index = VectorIndex::new(768, DistanceMetric::Cosine, 16, 200)?;

// Create DHT node
let local_peer_id = PeerId::random();
let dht_node = SemanticDHTNode::new(config, local_peer_id, local_index);

// Add peer to routing table
use ipfrs_semantic::SemanticPeer;
let peer_id = PeerId::random();
let peer_embedding = vec![0.5; 768];  // Aggregate embedding of peer's data
let peer = SemanticPeer::new(peer_id, peer_embedding);
dht_node.routing_table().add_peer(peer)?;

// Insert data (automatically replicated to nearest peers)
let cid: Cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi".parse()?;
let embedding = vec![0.7; 768];
dht_node.insert(&cid, &embedding).await?;

// Distributed k-NN search across the network
let query_embedding = vec![0.6; 768];
let results = dht_node.search_distributed(&query_embedding, 10).await?;

println!("Found {} results from distributed search", results.len());
for result in results {
    println!("  CID: {}, Score: {:.3}", result.cid, result.score);
}

// Update peer clusters for locality optimization
dht_node.routing_table().update_clusters(3)?;

// Get DHT statistics
let stats = dht_node.get_stats();
println!("DHT Stats:");
println!("  Peers: {}", stats.num_peers);
println!("  Clusters: {}", stats.num_clusters);
println!("  Local entries: {}", stats.num_local_entries);
println!("  Queries processed: {}", stats.queries_processed);
println!("  Avg latency: {:.2}ms", stats.avg_query_latency_ms);
# Ok(())
# }
```

### Federated Multi-Index Search

Query multiple indices simultaneously with heterogeneous distance metrics:

```rust
use ipfrs_semantic::{
    FederatedQueryExecutor, FederatedConfig, AggregationStrategy,
    LocalIndexAdapter, VectorIndex, DistanceMetric
};
use ipfrs_core::Cid;
use parking_lot::RwLock;
use std::sync::Arc;

# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
// Configure federated queries
let mut config = FederatedConfig::default();
config.aggregation_strategy = AggregationStrategy::RankFusion; // Best for heterogeneous metrics
config.privacy_preserving = true;  // Enable differential privacy
config.privacy_noise_level = 0.01; // Small noise for privacy

let executor = FederatedQueryExecutor::new(config);

// Create multiple indices with different metrics
let index1 = VectorIndex::new(768, DistanceMetric::Cosine, 16, 200)?;
let index2 = VectorIndex::new(768, DistanceMetric::L2, 16, 200)?;
let index3 = VectorIndex::new(768, DistanceMetric::DotProduct, 16, 200)?;

// Populate indices with data
// (In practice, these might be from different organizations or data sources)
let cid1: Cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi".parse()?;
let embedding1 = vec![0.5; 768];
Arc::new(RwLock::new(index1)).write().insert(&cid1, &embedding1)?;

// Register indices for federated queries
let adapter1 = LocalIndexAdapter::new(
    Arc::new(RwLock::new(VectorIndex::new(768, DistanceMetric::Cosine, 16, 200)?)),
    "org1_index".to_string()
);
let adapter2 = LocalIndexAdapter::new(
    Arc::new(RwLock::new(VectorIndex::new(768, DistanceMetric::L2, 16, 200)?)),
    "org2_index".to_string()
);

executor.register_index(Arc::new(adapter1))?;
executor.register_index(Arc::new(adapter2))?;

// Query all registered indices simultaneously
let query_embedding = vec![0.6; 768];
let results = executor.query(&query_embedding, 10).await?;

println!("Federated search found {} results", results.len());
for result in results {
    println!(
        "  CID: {}, Score: {:.3}, Source: {}, Metric: {:?}",
        result.cid, result.score, result.source_index_id, result.source_metric
    );
}

// Query specific indices only
let specific_results = executor.query_indices(
    &query_embedding,
    10,
    &["org1_index".to_string()]
).await?;

// Get federated query statistics
let stats = executor.stats();
println!("Federated Query Stats:");
println!("  Total queries: {}", stats.total_queries);
println!("  Indices queried: {}", stats.total_indices_queried);
println!("  Avg latency: {:.2}ms", stats.avg_latency_ms);
# Ok(())
# }
```

### Multi-Modal Semantic Search

Search across different data types (text, images, audio, etc.) in a unified embedding space:

```rust
use ipfrs_semantic::{MultiModalIndex, MultiModalConfig, MultiModalEmbedding, Modality};
use ipfrs_core::Cid;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
// Create multi-modal index
let mut config = MultiModalConfig::default();
config.project_to_unified = true;  // Enable unified embedding space
config.unified_dim = 512;

let mut index = MultiModalIndex::new(config);

// Register different modalities with their native dimensions
index.register_modality(Modality::Text, 768)?;  // BERT embeddings
index.register_modality(Modality::Image, 512)?;  // ResNet embeddings
index.register_modality(Modality::Audio, 768)?;  // Wav2Vec embeddings

// Add text content
let text_cid: Cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi".parse()?;
let text_embedding = MultiModalEmbedding::new(
    vec![0.1; 768],  // Text embedding from BERT
    Modality::Text
);
index.add(text_cid, text_embedding)?;

// Add image content
let image_cid: Cid = "bafybeigvgzoolh3cxsculpsjkz3hxfpg37pszqx3j7i5fwzgjmrmtv5wmi".parse()?;
let image_embedding = MultiModalEmbedding::new(
    vec![0.2; 512],  // Image embedding from ResNet
    Modality::Image
);
index.add(image_cid, image_embedding)?;

// Search within a specific modality
let text_query = MultiModalEmbedding::new(vec![0.15; 768], Modality::Text);
let text_results = index.search_modality(&text_query, 5, None)?;

// Cross-modal search: find similar content across all modalities
let cross_modal_results = index.search_cross_modal(&text_query, 10, None)?;
for (cid, score, modality) in cross_modal_results {
    println!("Found {:?} content: {} (score: {:.3})", modality, cid, score);
}

// Get statistics
let stats = index.stats();
for (modality, stat) in stats {
    println!("{:?}: {} embeddings, {} dims", modality, stat.num_embeddings, stat.dimension);
}
# Ok(())
# }
```

### Privacy-Preserving Search with Differential Privacy

Protect embedding privacy while maintaining search utility:

```rust
use ipfrs_semantic::{PrivacyMechanism, PrivacyBudget, PrivateEmbedding, TradeoffAnalyzer};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
// Create a privacy mechanism (epsilon-differential privacy)
let epsilon = 1.0;  // Privacy budget
let sensitivity = 1.0;  // L2 sensitivity of embeddings
let mechanism = PrivacyMechanism::laplacian(epsilon, sensitivity)?;

// Create a private embedding
let original_embedding = vec![0.5; 768];
let private_emb = PrivateEmbedding::new(original_embedding, mechanism);

// Use the noisy embedding for public release
let public_embedding = private_emb.public_embedding();
let (epsilon, delta) = private_emb.privacy_params();
println!("Privacy: ε={}, δ={}", epsilon, delta);
println!("Expected utility loss: {:.3}", private_emb.utility_loss());

// Track privacy budget across multiple queries
let budget = PrivacyBudget::new(10.0, 0.001)?;  // Total budget

// Consume budget for each query
budget.consume(0.5, 0.0001)?;
budget.consume(0.5, 0.0001)?;

println!("Remaining budget: {:.2}", budget.remaining());

// Analyze privacy-utility trade-offs
let analyzer = TradeoffAnalyzer::new(sensitivity);
let tradeoffs = analyzer.analyze(768);
for point in tradeoffs {
    println!("ε={:.1}: utility loss={:.2}", point.epsilon, point.utility_loss);
}

// Find best epsilon for target utility
if let Some(best_epsilon) = analyzer.find_epsilon_for_utility(768, 15.0) {
    println!("Best ε for utility loss <15.0: {:.2}", best_epsilon);
}
# Ok(())
# }
```

### Dynamic Embedding Updates and Version Migration

Manage evolving embeddings with version control and online updates:

```rust
use ipfrs_semantic::{DynamicIndex, ModelVersion, OnlineUpdater, EmbeddingTransform};
use ipfrs_core::Cid;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
// Create a dynamic index with version tracking
let v1 = ModelVersion::new(1, 0, 0);
let index = DynamicIndex::new(v1.clone(), 768)?;

// Add embeddings to version 1.0.0
let cid: Cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi".parse()?;
let embedding_v1 = vec![0.5; 768];
index.insert(&cid, &embedding_v1, None)?;

// Add a new model version with transformation
let v2 = ModelVersion::new(1, 1, 0);
let transform = EmbeddingTransform::identity(v1.clone());
index.add_version(v2.clone(), Some(transform))?;

// Set the new version as active
index.set_active_version(v2.clone())?;

// Add new embeddings to v2
let cid2: Cid = "bafybeigvgzoolh3cxsculpsjkz3hxfpg37pszqx3j7i5fwzgjmrmtv5wmi".parse()?;
let embedding_v2 = vec![0.6; 768];
index.insert(&cid2, &embedding_v2, Some(v2))?;

// Online fine-tuning with momentum
let updater = OnlineUpdater::new(0.01, 0.9);  // learning_rate, momentum

// Apply gradient updates
let gradient = vec![0.001; 768];
let updated_embedding = updater.update(&cid, &embedding_v2, &gradient);

// Track versions
let stats = index.version_stats();
for (version, stat) in stats {
    println!("Version {}: {} embeddings (active: {})",
        version, stat.num_embeddings, stat.is_active);
}

// Online updater statistics
let updater_stats = updater.stats();
println!("Online updater: lr={}, momentum={}, tracking {} embeddings",
    updater_stats.learning_rate, updater_stats.momentum, updater_stats.num_tracked);
# Ok(())
# }
```

## Performance

### SIMD Acceleration

The crate includes SIMD-optimized distance computations:

```rust
use ipfrs_semantic::{l2_distance, cosine_distance, dot_product};

let vec1 = vec![1.0, 2.0, 3.0, 4.0];
let vec2 = vec![0.5, 1.5, 2.5, 3.5];

// Uses ARM NEON or x86 SSE/AVX when available
let l2_dist = l2_distance(&vec1, &vec2);
let cos_dist = cosine_distance(&vec1, &vec2);
let dot_prod = dot_product(&vec1, &vec2);
```

### Performance Targets

- **Query latency**: < 1ms for 1M vectors (cached)
- **Query latency**: < 5ms for 1M vectors (uncached)
- **Index build time**: < 10min for 1M vectors
- **Memory usage**: < 2GB for 1M × 768-dim vectors
- **Recall@10**: > 95% for k-NN search

## Architecture

### Core Components

- **[`VectorIndex`]** - HNSW-based vector search index
- **[`SemanticRouter`]** - High-level routing with caching
- **[`HybridIndex`]** - Hybrid search with metadata filtering
- **[`DiskANNIndex`]** - Disk-based indexing for massive scale

### Optimization Layers

- **Quantization** - [`ProductQuantizer`], [`OptimizedProductQuantizer`], [`ScalarQuantizer`]
- **Caching** - [`HotEmbeddingCache`], [`AlignedVector`]
- **SIMD** - [`l2_distance`], [`cosine_distance`], [`dot_product`]

### Logic Integration

- **[`LogicSolver`]** - TensorLogic reasoning with embeddings
- **[`QueryExecutor`]** - SPARQL-like query language
- **[`ProvenanceTracker`]** - Audit trails and provenance

## Use Cases

### Semantic Content Discovery

Find similar content based on embeddings from models like:
- Text: BERT, RoBERTa, Sentence Transformers
- Images: CLIP, ResNet, ViT
- Multi-modal: CLIP, ALIGN

### Recommendation Systems

Build recommendation engines that find similar:
- Documents based on text embeddings
- Images based on visual features
- Users based on behavior embeddings

### Distributed AI Model Routing

Route AI inference requests to:
- Find similar cached results
- Locate relevant model weights
- Discover related training data

## Configuration

### Index Tuning

```rust
use ipfrs_semantic::{VectorIndex, DistanceMetric, ParameterTuner, UseCase};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
// Get recommended parameters for your use case
let rec = ParameterTuner::recommend(
    100_000,              // number of vectors
    768,                  // dimension
    UseCase::HighRecall   // optimize for recall
);

// Create index with recommended parameters
let index = VectorIndex::new(
    768,
    DistanceMetric::Cosine,
    rec.m,
    rec.ef_construction
)?;

println!("M: {}, efConstruction: {}", rec.m, rec.ef_construction);
println!("Estimated recall@10: {:.2}%", rec.estimated_recall * 100.0);
# Ok(())
# }
```

## Query Language

The crate provides a SPARQL-like query language for complex knowledge base queries:

```rust,no_run
use ipfrs_semantic::{Query, QueryPattern, QueryExecutor, FilterExpr, TermPattern};
use ipfrs_tensorlogic::{KnowledgeBase, Predicate, Term, Constant};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
// Create knowledge base
let mut kb = KnowledgeBase::new();

// Add some facts
let fact1 = Predicate::new("person".to_string(), vec![
    Term::Const(Constant::String("alice".to_string())),
]);
kb.add_fact(fact1);

let fact2 = Predicate::new("age".to_string(), vec![
    Term::Const(Constant::String("alice".to_string())),
    Term::Const(Constant::Int(30)),
]);
kb.add_fact(fact2);

// Create query executor
let executor = QueryExecutor::new(kb);

// Build a query using the builder pattern
let query = Query::new()
    .select("name")
    .select("age_val")
    .where_pattern(QueryPattern::Pattern {
        name: Some("person".to_string()),
        args: vec![TermPattern::Variable("name".to_string())],
    })
    .where_pattern(QueryPattern::Pattern {
        name: Some("age".to_string()),
        args: vec![
            TermPattern::Variable("name".to_string()),
            TermPattern::Variable("age_val".to_string()),
        ],
    })
    .limit(10);

// Execute the query
let result = executor.execute(query)?;

println!("Found {} results", result.bindings.len());
for binding in result.bindings {
    println!("  Name: {:?}, Age: {:?}", binding.get("name"), binding.get("age_val"));
}

// Query statistics
println!("Patterns evaluated: {}", result.stats.patterns_evaluated);
println!("Execution time: {} ms", result.stats.execution_time_ms);
# Ok(())
# }
```

### Query Features

- **SELECT clause**: Specify variables to return
- **WHERE patterns**: Pattern matching with wildcards and variables
- **FILTER expressions**: Filter results with boolean logic
- **LIMIT/OFFSET**: Pagination support
- **Query optimization**: Automatic join order optimization and filter pushdown

### Boolean Queries

```rust,no_run
use ipfrs_semantic::{BooleanQuery, Query, FilterExpr};

# fn main() {
// AND query: match both conditions
let and_query = BooleanQuery::And(vec![
    Query::new().select("x"),
    Query::new().select("y"),
]);

// OR query: match either condition
let or_query = BooleanQuery::Or(vec![
    Query::new().select("x"),
    Query::new().select("y"),
]);

// NOT query: negate a query
let not_query = BooleanQuery::Not(Box::new(
    Query::new().select("x")
));
# }
```

## Error Handling

All operations return `Result<T, ipfrs_core::Error>`:

```rust
use ipfrs_semantic::SemanticRouter;
use ipfrs_core::Error;

# #[tokio::main]
# async fn main() {
match SemanticRouter::with_defaults() {
    Ok(router) => println!("Router created successfully"),
    Err(Error::InvalidInput(msg)) => eprintln!("Invalid input: {}", msg),
    Err(e) => eprintln!("Error: {}", e),
}
# }
```

## Advanced Features

### Vector Quality Analysis

Validate embeddings and detect anomalies:

```rust
use ipfrs_semantic::{analyze_quality, detect_anomaly, compute_batch_stats};

# fn main() {
// Analyze a single vector
let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];
let quality = analyze_quality(&embedding);

println!("Quality score: {:.2}", quality.quality_score);
println!("Is valid: {}", quality.is_valid);
println!("Is normalized: {}", quality.is_normalized);
println!("Sparsity: {:.1}%", quality.sparsity * 100.0);

// Detect anomalies
let report = detect_anomaly(
    &embedding,
    0.3,   // expected mean
    0.15,  // expected std dev
    1.0,   // expected L2 norm
    0.1,   // mean tolerance
    0.1,   // std dev tolerance
    0.2,   // norm tolerance
);

if report.is_anomaly {
    println!("Anomaly detected: {}", report.description);
    println!("Confidence: {:.1}%", report.confidence * 100.0);
}

// Analyze batch of vectors
let vectors = vec![
    vec![0.1, 0.2, 0.3],
    vec![0.4, 0.5, 0.6],
    vec![0.7, 0.8, 0.9],
];
let batch_stats = compute_batch_stats(&vectors);

println!("Average quality: {:.2}", batch_stats.avg_quality);
println!("Valid vectors: {}/{}", batch_stats.valid_count, batch_stats.count);
# }
```

### Index Diagnostics and Health Monitoring

Monitor index health and performance:

```rust
use ipfrs_semantic::{VectorIndex, diagnose_index, HealthMonitor, SearchProfiler};
use std::time::Duration;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut index = VectorIndex::with_defaults(128)?;

// Run diagnostics
let report = diagnose_index(&index);

println!("Health status: {:?}", report.status);
println!("Index size: {} vectors", report.size);
println!("Memory usage: ~{:.2} MB", report.memory_usage as f64 / 1e6);

for issue in &report.issues {
    println!("Issue ({:?}): {}", issue.severity, issue.description);
    if let Some(fix) = &issue.suggested_fix {
        println!("  Suggested fix: {}", fix);
    }
}

for rec in &report.recommendations {
    println!("Recommendation: {}", rec);
}

// Set up periodic health monitoring
let mut monitor = HealthMonitor::new(Duration::from_secs(60));

if monitor.should_check() {
    let report = monitor.check(&index);
    println!("Health check: {:?}", report.status);
}

// Profile search performance
let mut profiler = SearchProfiler::new();

// Simulate queries
profiler.record_query(Duration::from_millis(5));
profiler.record_query(Duration::from_millis(3));
profiler.record_query(Duration::from_millis(4));

let stats = profiler.stats();
println!("Total queries: {}", stats.total_queries);
println!("Average latency: {:?}", stats.avg_latency);
println!("QPS: {:.2}", stats.qps);
# Ok(())
# }
```

### Index Optimization

Automatically tune index parameters:

```rust
use ipfrs_semantic::{analyze_optimization, OptimizationGoal, QueryOptimizer, MemoryOptimizer};
use std::time::Duration;

# fn main() {
// Analyze and get optimization recommendations
let result = analyze_optimization(
    50_000,  // index size
    768,     // dimension
    16,      // current M
    200,     // current ef_construction
    OptimizationGoal::Balanced,
);

println!("Current quality score: {:.2}", result.current_score);
println!("Recommended M: {}", result.recommended_m);
println!("Recommended ef_construction: {}", result.recommended_ef_construction);
println!("Recommended ef_search: {}", result.recommended_ef_search);
println!("Estimated improvement: {:.1}%", result.estimated_improvement * 100.0);

for reason in &result.reasoning {
    println!("  - {}", reason);
}

// Adaptive query optimization
let mut query_optimizer = QueryOptimizer::new(
    50,                            // initial ef_search
    Duration::from_millis(10),     // target latency
);

// The optimizer adjusts ef_search based on observed latency
for _ in 0..20 {
    query_optimizer.record_query(Duration::from_millis(15));
}

println!("Optimized ef_search: {}", query_optimizer.get_ef_search());

// Memory budget optimization
let mut memory_optimizer = MemoryOptimizer::new(1024 * 1024 * 1024); // 1GB

let (m, ef_c, max_vectors) = memory_optimizer.recommend_config(768);
println!("For 1GB budget:");
println!("  Recommended M: {}", m);
println!("  Recommended ef_construction: {}", ef_c);
println!("  Max vectors: {}", max_vectors);
# }
```
