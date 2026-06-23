//! Contextual Embedding Search — context-aware vector search with query expansion,
//! negative example suppression, diversity-aware re-ranking, and rich result explanations.
//!
//! # Overview
//!
//! [`ContextualEmbeddingSearch`] maintains an in-memory flat index of [`SearchDoc`]s and
//! exposes a single [`ContextualEmbeddingSearch::search`] method that:
//!
//! 1. Expands the raw query embedding using recent query history and positive examples.
//! 2. Optionally suppresses directions associated with negative examples.
//! 3. Retrieves the top-`rerank_top_n` candidates via brute-force cosine similarity.
//! 4. Re-ranks those candidates with one of four [`DiversityStrategy`] variants.
//! 5. Returns up to `top_k` [`ContextualResult`]s with per-feature score explanations.

pub mod contextualembeddingsearch_traits;
pub mod functions;
pub mod searchconfig_traits;
pub mod searcherror_traits;
pub mod types;

// Re-export all types
pub use functions::*;
pub use types::*;

#[cfg(test)]
mod tests;
