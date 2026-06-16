//! IPFRS - Inter-Planet File RUST System
//!
//! Version: 0.2.0 "Network Release"
//!
//! A next-generation distributed file system built in Rust, combining:
//! - High-performance zero-copy data transport
//! - Semantic routing and vector search
//! - TensorLogic integration for distributed reasoning
//! - ARM-optimized implementation
//! - Content-addressed storage with IPLD
//! - Distributed networking via libp2p
//! - Pin management and garbage collection
//! - Comprehensive observability (metrics, logging, tracing)
//!
//! # Architecture
//!
//! IPFRS follows a bi-layer architecture:
//!
//! ## Logical Layer (The Brain)
//! - **Semantic Router**: Vector search and logic solving for content discovery
//! - **TensorLogic**: Distributed reasoning and knowledge representation
//! - **Differentiable Storage**: Gradient tracking for learning
//!
//! ## Physical Layer (The Body)
//! - **TensorSwap**: Optimized tensor streaming protocol
//! - **Block Storage**: Sled-backed content-addressed storage
//! - **Network Stack**: libp2p with QUIC, Bitswap, and DHT
//! - **Pin Management**: Protect important content from garbage collection
//!
//! # Features
//!
//! ## Content-Addressed Storage
//! - Store and retrieve data by content hash (CID)
//! - IPLD support for structured data with links
//! - DAG operations for hierarchical data structures
//!
//! ## Semantic Search
//! - Vector similarity search using HNSW index
//! - Filter-based queries
//! - Persistent index with save/load support
//!
//! ## Logic Programming
//! - TensorLogic integration for facts, rules, and inference
//! - Backward chaining reasoning
//! - Proof generation and verification
//! - Knowledge base persistence
//!
//! ## Networking
//! - Peer-to-peer content distribution
//! - DHT-based content discovery
//! - Bitswap protocol for block exchange
//! - NAT traversal and relay support
//!
//! ## Pin Management
//! - Direct and recursive pinning
//! - Automatic DAG traversal for recursive pins
//! - Protect content from garbage collection
//!
//! ## Observability
//! - Prometheus metrics
//! - Structured logging with tracing
//! - OpenTelemetry integration
//! - Health checks and status monitoring
//!
//! # Quick Start
//!
//! ## Basic Node Setup
//!
//! ```rust,no_run
//! use ipfrs::{Node, NodeConfig};
//!
//! #[tokio::main]
//! async fn main() -> ipfrs::Result<()> {
//!     // Create and start a node with default configuration
//!     let config = NodeConfig::default();
//!     let mut node = Node::new(config)?;
//!     node.start().await?;
//!
//!     // Node is now running and ready to use
//!     println!("Node started successfully!");
//!
//!     // Stop the node when done
//!     node.stop().await?;
//!     Ok(())
//! }
//! ```
//!
//! ## Storing and Retrieving Content
//!
//! ```rust,no_run
//! # use ipfrs::{Node, NodeConfig, Block};
//! # use bytes::Bytes;
//! #
//! # #[tokio::main]
//! # async fn main() -> ipfrs::Result<()> {
//! # let mut node = Node::new(NodeConfig::default())?;
//! # node.start().await?;
//! // Store content
//! let data = Bytes::from("Hello, IPFRS!");
//! let block = Block::new(data.clone())?;
//! let cid = *block.cid();
//! node.put_block(&block).await?;
//! println!("Stored content with CID: {}", cid);
//!
//! // Retrieve content
//! let retrieved = node.get_block(&cid).await?;
//! assert!(retrieved.is_some());
//! # Ok(())
//! # }
//! ```
//!
//! ## Pin Management
//!
//! ```rust,no_run
//! # use ipfrs::{Node, NodeConfig, Block};
//! # use bytes::Bytes;
//! #
//! # #[tokio::main]
//! # async fn main() -> ipfrs::Result<()> {
//! # let mut node = Node::new(NodeConfig::default())?;
//! # node.start().await?;
//! # let block = Block::new(Bytes::from("data"))?;
//! # let cid = *block.cid();
//! # node.put_block(&block).await?;
//! // Pin content to prevent garbage collection
//! node.pin_add(&cid, false, Some("important-data".to_string())).await?;
//!
//! // List all pins
//! let pins = node.pin_ls()?;
//! for pin in pins {
//!     println!("Pinned: {} ({:?})", pin.cid, pin.pin_type);
//! }
//!
//! // Unpin when no longer needed
//! node.pin_rm(&cid, false).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Semantic Search
//!
//! ```rust,no_run
//! # use ipfrs::{Node, NodeConfig, Block};
//! # use bytes::Bytes;
//! #
//! # #[tokio::main]
//! # async fn main() -> ipfrs::Result<()> {
//! # let mut node = Node::new(NodeConfig::default().with_semantic(Default::default()))?;
//! # node.start().await?;
//! # let block = Block::new(Bytes::from("content"))?;
//! # let cid = *block.cid();
//! # node.put_block(&block).await?;
//! // Index content for semantic search
//! let embedding = vec![0.1; 768]; // Your embedding vector
//! node.index_content(&cid, &embedding).await?;
//!
//! // Perform similarity search
//! let query = vec![0.2; 768];
//! let results = node.search_similar(&query, 10).await?;
//! for result in results {
//!     println!("Found: {} (score: {})", result.cid, result.score);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Logic Programming
//!
//! ```rust,no_run
//! # use ipfrs::{Node, NodeConfig, Predicate, Term, Constant};
//! #
//! # #[tokio::main]
//! # async fn main() -> ipfrs::Result<()> {
//! # let mut node = Node::new(NodeConfig::default().with_tensorlogic())?;
//! # node.start().await?;
//! // Add facts
//! let fact = Predicate::new(
//!     "parent".to_string(),
//!     vec![
//!         Term::Const(Constant::String("alice".to_string())),
//!         Term::Const(Constant::String("bob".to_string())),
//!     ],
//! );
//! node.add_fact(fact)?;
//!
//! // Perform inference
//! let query = Predicate::new(
//!     "parent".to_string(),
//!     vec![
//!         Term::Var("X".to_string()),
//!         Term::Const(Constant::String("bob".to_string())),
//!     ],
//! );
//! let results = node.infer(&query)?;
//! println!("Found {} results", results.len());
//! # Ok(())
//! # }
//! ```

// Re-export core types
pub use ipfrs_core::{Block, Cid, Error, Ipld, Result};

// Re-export network types
pub use ipfrs_network::{NetworkConfig, NetworkNode};

// Re-export storage types
pub use ipfrs_storage::{BlockStoreConfig, BlockStoreTrait, SledBlockStore};

// Re-export transport types
pub use ipfrs_transport::{
    BitswapConfig, BitswapExchange, BitswapStats, Message, PeerId, TensorMetadata, TensorSwap,
    TensorSwapConfig, TensorSwapStats, WantEntry,
};

// Re-export semantic types
pub use ipfrs_semantic::{
    DistanceMetric, QueryFilter, RouterConfig, RouterStats, SearchResult, SemanticRouter,
    VectorIndex,
};

// Re-export interface types
pub use ipfrs_interface::{
    Gateway, GatewayConfig, RealtimeEvent, SubscriptionManager, WsMessage, WsState, ZeroCopyBuffer,
};

// Re-export TensorLogic types
pub use ipfrs_tensorlogic::{
    Constant, KnowledgeBase, KnowledgeBaseStats, Predicate, Proof, Rule, Substitution,
    TensorLogicStore, Term, TermRef,
};

pub mod auth;
pub mod diagnostics;
pub mod fsck;
pub mod gc;
pub mod health;
pub mod metrics;
pub mod node;
pub mod pin;
pub mod profiler;
pub mod recovery;
pub mod repo;
pub mod shutdown;
pub mod tls;
pub mod tracing_setup;

pub use node::{
    BlockStat, DagExportStats, DagImportStats, DistributedInferResult, FsckResult, GcResult, Node,
    NodeConfig, NodeStatus, SemanticStats, StorageStats, TensorLogicStats,
};

pub use auth::{AuthManager, AuthToken, OAuth2Config, Permission, Role, TokenType, User};

pub use tls::{
    CertificateInfo, SelfSignedCertGenerator, TlsConfig, TlsError, TlsManager, TlsVersion,
};

pub use diagnostics::{
    DiagnosticAnalyzer, DiagnosticRecommendation, HealthStatus, NetworkDiagnostics,
    NodeDiagnostics, RecommendationSeverity, ResourceUsage, SemanticDiagnostics,
    StorageDiagnostics, TensorLogicDiagnostics,
};
pub use fsck::{FilesystemChecker, FsckConfig, FsckResult as FsckResultDetailed};
pub use gc::{GarbageCollector, GcConfig, GcStats};
pub use pin::{PinInfo, PinManager, PinType};
pub use profiler::{OperationStats, Profiler};
pub use repo::{format_bytes, BlockDistribution, RepoAnalyzer, RepoStats};
