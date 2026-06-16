//! Multi-Modal Content Indexer — unified index for text, vector, and structured data.
//!
//! [`MultiModalIndexer`] is the primary entry point.  It maintains three
//! sub-indexes:
//!
//! * **Inverted text index** — tokenised BM25 scoring over `ModalityData::Text` payloads.
//! * **Brute-force vector index** — cosine similarity over `ModalityData::Vector` payloads.
//! * **Exact-match structured index** — field+value pairs from
//!   `ModalityData::Structured` (and `ModalityData::Binary` / `ModalityData::Numeric`
//!   stored as string representations for filtering).
//!
//! Scores from active modalities are blended with configurable weights (default
//! 0.4 / 0.4 / 0.2 for text / vector / structured), and results filtered by
//! `min_score` before the top-k are returned.
//!
//! # Name-Collision Notes
//!
//! Several type names collide with existing exports in this crate.  Prefixed
//! aliases are provided for the ones that clash:
//!
//! | New name               | Canonical name      | Alias exported           |
//! |------------------------|---------------------|--------------------------|
//! | `MmiSearchQuery`       | `SearchQuery`       | existing: multimodal_search |
//! | `MmiSearchResult`      | `SearchResult`      | existing: various        |
//! | `MmiIndexedDocument`   | `IndexedDocument`   | existing: corpus_indexer |
//! | `MmiIndexError`        | `IndexError`        | existing: corpus_indexer |
//! | `MmiIndexStats`        | `IndexStats`        | existing: stats          |
//! | `MmiIndexConfig`       | `IndexConfig`       | existing: benchmark_comparison |

pub mod constants;
pub mod functions;
pub mod mmiindexconfig_traits;
pub mod mmiindexerror_traits;
pub mod type_aliases;
pub mod types;

// Re-export all types
pub use functions::*;
pub use type_aliases::*;
pub use types::*;

#[cfg(test)]
mod tests;
