//! Semantic Knowledge Graph Builder
//!
//! Full production-quality implementation of a semantic knowledge graph builder
//! that constructs and manipulates graphs from text and embeddings.
//!
//! # Name Aliasing
//! Because `GraphNode`, `GraphEdge`, and `GraphQuery` are already exported from
//! `graph_linker` and `knowledge_graph`, the local equivalents are prefixed `Sgb`:
//!
//! - [`SgbGraphNode`] — a node in the semantic graph
//! - [`SgbGraphEdge`] — a directed, weighted edge
//! - [`SgbGraphQuery`] — query parameters for subgraph traversal
//!
//! All other public types (`NodeType`, `EdgeRelation`, `BuilderConfig`,
//! `GraphStats`, `BuilderError`, `SemanticGraphBuilder`) have no conflicts and
//! are exported under their own names.

pub mod builderconfig_traits;
pub mod buildererror_traits;
pub mod functions;
pub mod types;

// Re-export all types
pub use types::*;

#[cfg(test)]
mod tests;
