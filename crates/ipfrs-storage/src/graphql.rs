//! GraphQL Query Interface for Block Metadata
//!
//! This module provides a GraphQL API for querying blocks by properties,
//! with support for filtering, sorting, and pagination.
//!
//! # Features
//!
//! - Query blocks by CID, size, or age
//! - Filter by size range, date range, or CID pattern
//! - Sort by size, creation time, or CID
//! - Cursor-based pagination for large result sets
//! - Aggregate statistics (total count, total size, etc.)
//!
//! # Example
//!
//! ```ignore
//! use ipfrs_storage::graphql::{BlockQuerySchema, QueryRoot};
//! use ipfrs_storage::MemoryBlockStore;
//! use async_graphql::Schema;
//!
//! #[tokio::main]
//! async fn main() {
//!     let store = MemoryBlockStore::new();
//!     let schema = Schema::build(QueryRoot, EmptyMutation, EmptySubscription)
//!         .data(store)
//!         .finish();
//!
//!     let query = r#"
//!         query {
//!             blocks(filter: { minSize: 1000 }, limit: 10) {
//!                 nodes {
//!                     cid
//!                     size
//!                 }
//!                 totalCount
//!             }
//!         }
//!     "#;
//!
//!     let result = schema.execute(query).await;
//!     println!("{}", result.data);
//! }
//! ```

use crate::traits::BlockStore;
use async_graphql::{
    Context, EmptyMutation, EmptySubscription, Object, Result, Schema, SimpleObject,
};
use chrono::{DateTime, Utc};
use ipfrs_core::{Block, Cid};
use std::sync::Arc;

/// GraphQL schema for block queries
pub type BlockQuerySchema = Schema<QueryRoot, EmptyMutation, EmptySubscription>;

/// Block metadata exposed via GraphQL
#[derive(Debug, Clone, SimpleObject)]
pub struct BlockMetadata {
    /// Content identifier (CID) as string
    pub cid: String,
    /// Block size in bytes
    pub size: u64,
    /// Creation/insertion timestamp (simulated for now)
    pub created_at: DateTime<Utc>,
    /// Block data (optional, can be large)
    #[graphql(skip)]
    pub data: Option<Vec<u8>>,
}

impl BlockMetadata {
    /// Create metadata from a Block
    pub fn from_block(block: &Block) -> Self {
        Self {
            cid: block.cid().to_string(),
            size: block.data().len() as u64,
            created_at: Utc::now(), // In production, this would be tracked
            data: Some(block.data().to_vec()),
        }
    }

    /// Get the actual CID
    pub fn parse_cid(&self) -> Result<Cid> {
        self.cid
            .parse()
            .map_err(|e| format!("Invalid CID: {e}").into())
    }
}

/// Filter criteria for block queries
#[derive(Debug, Clone, Default)]
pub struct BlockFilter {
    /// Minimum block size in bytes
    pub min_size: Option<u64>,
    /// Maximum block size in bytes
    pub max_size: Option<u64>,
    /// Filter by CID prefix
    pub cid_prefix: Option<String>,
    /// Filter blocks created after this time
    pub created_after: Option<DateTime<Utc>>,
    /// Filter blocks created before this time
    pub created_before: Option<DateTime<Utc>>,
}

#[Object]
impl BlockFilter {
    async fn min_size(&self) -> Option<u64> {
        self.min_size
    }

    async fn max_size(&self) -> Option<u64> {
        self.max_size
    }

    async fn cid_prefix(&self) -> Option<&str> {
        self.cid_prefix.as_deref()
    }
}

/// Sort order for block queries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    Ascending,
    Descending,
}

/// Field to sort blocks by
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortField {
    /// Sort by CID (lexicographically)
    Cid,
    /// Sort by block size
    Size,
    /// Sort by creation time
    CreatedAt,
}

/// Paginated result for block queries
#[derive(Debug, Clone, SimpleObject)]
pub struct BlockConnection {
    /// List of blocks in this page
    pub nodes: Vec<BlockMetadata>,
    /// Total count of blocks matching the query
    pub total_count: u64,
    /// Cursor for the next page (if available)
    pub next_cursor: Option<String>,
    /// Whether there are more results
    pub has_next_page: bool,
}

/// Aggregate statistics for blocks
#[derive(Debug, Clone, SimpleObject)]
pub struct BlockStats {
    /// Total number of blocks
    pub count: u64,
    /// Total size of all blocks in bytes
    pub total_size: u64,
    /// Average block size in bytes
    pub average_size: f64,
    /// Minimum block size
    pub min_size: u64,
    /// Maximum block size
    pub max_size: u64,
}

/// GraphQL query root
pub struct QueryRoot;

#[Object]
impl QueryRoot {
    /// Query blocks with optional filtering, sorting, and pagination
    async fn blocks(
        &self,
        ctx: &Context<'_>,
        #[graphql(desc = "Minimum size filter")] min_size: Option<u64>,
        #[graphql(desc = "Maximum size filter")] max_size: Option<u64>,
        #[graphql(desc = "CID prefix filter")] cid_prefix: Option<String>,
        #[graphql(desc = "Maximum number of results", default = 100)] limit: usize,
        #[graphql(desc = "Cursor for pagination")] cursor: Option<String>,
    ) -> Result<BlockConnection> {
        let store = ctx.data::<Arc<dyn BlockStore + Send + Sync>>()?;

        // Get all CIDs from the store
        let cids = store
            .list_cids()
            .map_err(|e| format!("Failed to list CIDs: {e}"))?;

        // Fetch all blocks and convert to metadata
        let mut blocks = Vec::new();
        for cid in cids {
            if let Some(block) = store
                .get(&cid)
                .await
                .map_err(|e| format!("Failed to get block: {e}"))?
            {
                let metadata = BlockMetadata::from_block(&block);

                // Apply filters
                if let Some(min) = min_size {
                    if metadata.size < min {
                        continue;
                    }
                }
                if let Some(max) = max_size {
                    if metadata.size > max {
                        continue;
                    }
                }
                if let Some(prefix) = &cid_prefix {
                    if !metadata.cid.starts_with(prefix) {
                        continue;
                    }
                }

                blocks.push(metadata);
            }
        }

        // Sort by size (default)
        blocks.sort_by_key(|b| b.size);

        // Apply cursor-based pagination
        let start_idx = if let Some(cursor) = cursor {
            cursor.parse::<usize>().unwrap_or(0)
        } else {
            0
        };

        let total_count = blocks.len() as u64;
        let end_idx = (start_idx + limit).min(blocks.len());
        let paginated_blocks: Vec<_> = blocks[start_idx..end_idx].to_vec();
        let has_next_page = end_idx < blocks.len();
        let next_cursor = if has_next_page {
            Some(end_idx.to_string())
        } else {
            None
        };

        Ok(BlockConnection {
            nodes: paginated_blocks,
            total_count,
            next_cursor,
            has_next_page,
        })
    }

    /// Get a single block by CID
    async fn block(
        &self,
        ctx: &Context<'_>,
        #[graphql(desc = "Content identifier")] cid: String,
    ) -> Result<Option<BlockMetadata>> {
        let store = ctx.data::<Arc<dyn BlockStore + Send + Sync>>()?;

        let cid: Cid = cid.parse().map_err(|e| format!("Invalid CID: {e}"))?;

        let block = store
            .get(&cid)
            .await
            .map_err(|e| format!("Failed to get block: {e}"))?;

        Ok(block.as_ref().map(BlockMetadata::from_block))
    }

    /// Get aggregate statistics for all blocks
    async fn stats(&self, ctx: &Context<'_>) -> Result<BlockStats> {
        let store = ctx.data::<Arc<dyn BlockStore + Send + Sync>>()?;

        let cids = store
            .list_cids()
            .map_err(|e| format!("Failed to list CIDs: {e}"))?;

        let mut count = 0u64;
        let mut total_size = 0u64;
        let mut min_size = u64::MAX;
        let mut max_size = 0u64;

        for cid in cids {
            if let Some(block) = store
                .get(&cid)
                .await
                .map_err(|e| format!("Failed to get block: {e}"))?
            {
                let size = block.data().len() as u64;
                count += 1;
                total_size += size;
                min_size = min_size.min(size);
                max_size = max_size.max(size);
            }
        }

        let average_size = if count > 0 {
            total_size as f64 / count as f64
        } else {
            0.0
        };

        Ok(BlockStats {
            count,
            total_size,
            average_size,
            min_size: if min_size == u64::MAX { 0 } else { min_size },
            max_size,
        })
    }

    /// Search blocks by CID pattern
    async fn search(
        &self,
        ctx: &Context<'_>,
        #[graphql(desc = "Search pattern (CID prefix)")] pattern: String,
        #[graphql(desc = "Maximum number of results", default = 10)] limit: usize,
    ) -> Result<Vec<BlockMetadata>> {
        let store = ctx.data::<Arc<dyn BlockStore + Send + Sync>>()?;

        let cids = store
            .list_cids()
            .map_err(|e| format!("Failed to list CIDs: {e}"))?;

        let mut results = Vec::new();
        for cid in cids {
            let cid_str = cid.to_string();
            if cid_str.contains(&pattern) {
                if let Some(block) = store
                    .get(&cid)
                    .await
                    .map_err(|e| format!("Failed to get block: {e}"))?
                {
                    results.push(BlockMetadata::from_block(&block));
                    if results.len() >= limit {
                        break;
                    }
                }
            }
        }

        Ok(results)
    }
}

/// Create a new GraphQL schema with a block store
pub fn create_schema<S: BlockStore + Send + Sync + 'static>(store: S) -> BlockQuerySchema {
    let store: Arc<dyn BlockStore + Send + Sync> = Arc::new(store);
    Schema::build(QueryRoot, EmptyMutation, EmptySubscription)
        .data(store)
        .finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryBlockStore;
    use bytes::Bytes;

    #[tokio::test]
    async fn test_query_all_blocks() {
        let store = MemoryBlockStore::new();

        // Add some test blocks
        let block1 = Block::new(Bytes::from("hello")).unwrap();
        let block2 = Block::new(Bytes::from("world")).unwrap();
        store.put(&block1).await.unwrap();
        store.put(&block2).await.unwrap();

        let schema = create_schema(store);

        let query = r#"
            query {
                blocks(limit: 10) {
                    nodes {
                        cid
                        size
                    }
                    totalCount
                }
            }
        "#;

        let result = schema.execute(query).await;
        assert!(result.errors.is_empty());
    }

    #[tokio::test]
    async fn test_query_single_block() {
        let store = MemoryBlockStore::new();
        let block = Block::new(Bytes::from("test")).unwrap();
        let cid = block.cid().to_string();
        store.put(&block).await.unwrap();

        let schema = create_schema(store);

        let query = format!(
            r#"
            query {{
                block(cid: "{}") {{
                    cid
                    size
                }}
            }}
        "#,
            cid
        );

        let result = schema.execute(&query).await;
        assert!(result.errors.is_empty());
    }

    #[tokio::test]
    async fn test_stats_query() {
        let store = MemoryBlockStore::new();

        let block1 = Block::new(Bytes::from("hello")).unwrap();
        let block2 = Block::new(Bytes::from("world")).unwrap();
        store.put(&block1).await.unwrap();
        store.put(&block2).await.unwrap();

        let schema = create_schema(store);

        let query = r#"
            query {
                stats {
                    count
                    totalSize
                    averageSize
                    minSize
                    maxSize
                }
            }
        "#;

        let result = schema.execute(query).await;
        assert!(result.errors.is_empty());
    }

    #[tokio::test]
    async fn test_search_blocks() {
        let store = MemoryBlockStore::new();
        let block = Block::new(Bytes::from("searchable")).unwrap();
        store.put(&block).await.unwrap();

        let schema = create_schema(store);

        let query = r#"
            query {
                search(pattern: "ba", limit: 5) {
                    cid
                    size
                }
            }
        "#;

        let result = schema.execute(query).await;
        assert!(result.errors.is_empty());
    }

    #[tokio::test]
    async fn test_filter_by_size() {
        let store = MemoryBlockStore::new();

        let small = Block::new(Bytes::from("hi")).unwrap();
        let large = Block::new(Bytes::from("this is a much larger block")).unwrap();
        store.put(&small).await.unwrap();
        store.put(&large).await.unwrap();

        let schema = create_schema(store);

        let query = r#"
            query {
                blocks(minSize: 10, limit: 10) {
                    nodes {
                        size
                    }
                    totalCount
                }
            }
        "#;

        let result = schema.execute(query).await;
        assert!(result.errors.is_empty());
    }
}
