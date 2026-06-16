//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    MmiIndexConfig, MmiIndexError, MmiIndexStats, MmiIndexedDocument, MmiSearchQuery,
    MmiSearchResult,
};

/// Public type alias so callers may use the canonical name.
pub type IndexedDocument = MmiIndexedDocument;
/// Public type alias — `MmiSearchQuery` is also exported under this name.
///
/// Note: the pre-existing `SearchQuery` from `multimodal_search` differs in
/// structure; use the `Mmi`-prefixed name in contexts where both are imported.
pub type MmiSearchQueryAlias = MmiSearchQuery;
/// Public type alias.
pub type MmiSearchResultAlias = MmiSearchResult;
/// Public type alias — `MmiIndexConfig` is also exported under this name.
///
/// Note: the pre-existing `IndexConfig` from `benchmark_comparison` differs in
/// structure.
pub type MmiIndexConfigAlias = MmiIndexConfig;
/// Public type alias.
pub type MmiIndexStatsAlias = MmiIndexStats;
/// Public type alias.
pub type MmiIndexErrorAlias = MmiIndexError;
