//! Storage traits

use async_trait::async_trait;
use ipfrs_core::{Block, Cid, Result};
use std::sync::Arc;

/// Trait for block storage backends
#[async_trait]
pub trait BlockStore: Send + Sync {
    /// Store a single block
    async fn put(&self, block: &Block) -> Result<()>;

    /// Store multiple blocks atomically
    async fn put_many(&self, blocks: &[Block]) -> Result<()> {
        // Default implementation: sequential puts
        for block in blocks {
            self.put(block).await?;
        }
        Ok(())
    }

    /// Retrieve a block by CID
    async fn get(&self, cid: &Cid) -> Result<Option<Block>>;

    /// Retrieve multiple blocks
    async fn get_many(&self, cids: &[Cid]) -> Result<Vec<Option<Block>>> {
        // Default implementation: sequential gets
        let mut results = Vec::with_capacity(cids.len());
        for cid in cids {
            results.push(self.get(cid).await?);
        }
        Ok(results)
    }

    /// Check if a block exists
    async fn has(&self, cid: &Cid) -> Result<bool>;

    /// Check if multiple blocks exist
    async fn has_many(&self, cids: &[Cid]) -> Result<Vec<bool>> {
        // Default implementation: sequential checks
        let mut results = Vec::with_capacity(cids.len());
        for cid in cids {
            results.push(self.has(cid).await?);
        }
        Ok(results)
    }

    /// Delete a block
    async fn delete(&self, cid: &Cid) -> Result<()>;

    /// Delete multiple blocks
    async fn delete_many(&self, cids: &[Cid]) -> Result<()> {
        // Default implementation: sequential deletes
        for cid in cids {
            self.delete(cid).await?;
        }
        Ok(())
    }

    /// List all CIDs in the store
    fn list_cids(&self) -> Result<Vec<Cid>>;

    /// Get number of blocks
    fn len(&self) -> usize;

    /// Check if store is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Flush any pending writes
    async fn flush(&self) -> Result<()> {
        // Default: no-op
        Ok(())
    }

    /// Close the store and release resources
    async fn close(&self) -> Result<()> {
        // Default: flush then no-op
        self.flush().await
    }
}

/// Blanket implementation for `Arc<S>` where `S: BlockStore`
/// This allows Arc-wrapped stores to be used transparently
#[async_trait]
impl<S: BlockStore> BlockStore for Arc<S> {
    async fn put(&self, block: &Block) -> Result<()> {
        (**self).put(block).await
    }

    async fn put_many(&self, blocks: &[Block]) -> Result<()> {
        (**self).put_many(blocks).await
    }

    async fn get(&self, cid: &Cid) -> Result<Option<Block>> {
        (**self).get(cid).await
    }

    async fn get_many(&self, cids: &[Cid]) -> Result<Vec<Option<Block>>> {
        (**self).get_many(cids).await
    }

    async fn has(&self, cid: &Cid) -> Result<bool> {
        (**self).has(cid).await
    }

    async fn has_many(&self, cids: &[Cid]) -> Result<Vec<bool>> {
        (**self).has_many(cids).await
    }

    async fn delete(&self, cid: &Cid) -> Result<()> {
        (**self).delete(cid).await
    }

    async fn delete_many(&self, cids: &[Cid]) -> Result<()> {
        (**self).delete_many(cids).await
    }

    fn list_cids(&self) -> Result<Vec<Cid>> {
        (**self).list_cids()
    }

    fn len(&self) -> usize {
        (**self).len()
    }

    fn is_empty(&self) -> bool {
        (**self).is_empty()
    }

    async fn flush(&self) -> Result<()> {
        (**self).flush().await
    }

    async fn close(&self) -> Result<()> {
        (**self).close().await
    }
}
