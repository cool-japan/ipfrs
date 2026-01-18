//! In-memory block store for testing
//!
//! This module provides a simple in-memory implementation of BlockStore
//! that is useful for testing and benchmarking.

use crate::traits::BlockStore;
use async_trait::async_trait;
use dashmap::DashMap;
use ipfrs_core::{Block, Cid, Result};
use std::sync::Arc;

/// In-memory block store using a concurrent hash map
#[derive(Clone)]
pub struct MemoryBlockStore {
    blocks: Arc<DashMap<Cid, Block>>,
}

impl MemoryBlockStore {
    /// Create a new in-memory block store
    pub fn new() -> Self {
        Self {
            blocks: Arc::new(DashMap::new()),
        }
    }

    /// Clear all blocks
    pub fn clear(&self) {
        self.blocks.clear();
    }
}

impl Default for MemoryBlockStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BlockStore for MemoryBlockStore {
    async fn put(&self, block: &Block) -> Result<()> {
        self.blocks.insert(*block.cid(), block.clone());
        Ok(())
    }

    async fn get(&self, cid: &Cid) -> Result<Option<Block>> {
        Ok(self.blocks.get(cid).map(|entry| entry.value().clone()))
    }

    async fn has(&self, cid: &Cid) -> Result<bool> {
        Ok(self.blocks.contains_key(cid))
    }

    async fn delete(&self, cid: &Cid) -> Result<()> {
        self.blocks.remove(cid);
        Ok(())
    }

    fn list_cids(&self) -> Result<Vec<Cid>> {
        Ok(self.blocks.iter().map(|entry| *entry.key()).collect())
    }

    fn len(&self) -> usize {
        self.blocks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    #[tokio::test]
    async fn test_put_get() {
        let store = MemoryBlockStore::new();
        let data = Bytes::from("hello");
        let block = Block::new(data.clone()).unwrap();

        store.put(&block).await.unwrap();
        let retrieved = store.get(block.cid()).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().data(), &data);
    }

    #[tokio::test]
    async fn test_has() {
        let store = MemoryBlockStore::new();
        let data = Bytes::from("hello");
        let block = Block::new(data).unwrap();

        assert!(!store.has(block.cid()).await.unwrap());
        store.put(&block).await.unwrap();
        assert!(store.has(block.cid()).await.unwrap());
    }

    #[tokio::test]
    async fn test_delete() {
        let store = MemoryBlockStore::new();
        let data = Bytes::from("hello");
        let block = Block::new(data).unwrap();

        store.put(&block).await.unwrap();
        assert!(store.has(block.cid()).await.unwrap());

        store.delete(block.cid()).await.unwrap();
        assert!(!store.has(block.cid()).await.unwrap());
    }

    #[tokio::test]
    async fn test_batch_operations() {
        let store = MemoryBlockStore::new();
        let block1 = Block::new(Bytes::from("hello")).unwrap();
        let block2 = Block::new(Bytes::from("world")).unwrap();

        let blocks = vec![block1.clone(), block2.clone()];
        store.put_many(&blocks).await.unwrap();

        let results = store
            .get_many(&[*block1.cid(), *block2.cid()])
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].is_some());
        assert!(results[1].is_some());
    }

    #[tokio::test]
    async fn test_len() {
        let store = MemoryBlockStore::new();
        assert_eq!(store.len(), 0);

        let block = Block::new(Bytes::from("test")).unwrap();
        store.put(&block).await.unwrap();
        assert_eq!(store.len(), 1);
    }

    #[tokio::test]
    async fn test_list_cids() {
        let store = MemoryBlockStore::new();
        let block1 = Block::new(Bytes::from("hello")).unwrap();
        let block2 = Block::new(Bytes::from("world")).unwrap();

        store.put(&block1).await.unwrap();
        store.put(&block2).await.unwrap();

        let cids = store.list_cids().unwrap();
        assert_eq!(cids.len(), 2);
        assert!(cids.contains(block1.cid()));
        assert!(cids.contains(block2.cid()));
    }
}
