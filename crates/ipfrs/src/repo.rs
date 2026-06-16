//! Repository Statistics and Utilities
//!
//! Provides comprehensive repository information and management utilities.

use ipfrs_core::{Cid, Result};
use ipfrs_storage::BlockStoreTrait;
use std::collections::HashMap;
use std::sync::Arc;

use crate::pin::PinManager;

/// Comprehensive repository statistics
#[derive(Debug, Clone)]
pub struct RepoStats {
    /// Total number of blocks
    pub num_blocks: usize,
    /// Total repository size in bytes
    pub repo_size: u64,
    /// Number of pinned blocks
    pub num_pinned: usize,
    /// Number of unpinned blocks (eligible for GC)
    pub num_unpinned: usize,
    /// Total size of pinned blocks
    pub pinned_size: u64,
    /// Total size of unpinned blocks
    pub unpinned_size: u64,
    /// Average block size
    pub avg_block_size: u64,
    /// Largest block size
    pub max_block_size: u64,
    /// Smallest block size
    pub min_block_size: u64,
    /// Storage path
    pub storage_path: String,
    /// Version information
    pub version: String,
}

/// Block size distribution
#[derive(Debug, Clone)]
pub struct BlockDistribution {
    /// Number of blocks < 1KB
    pub tiny: usize,
    /// Number of blocks 1KB - 10KB
    pub small: usize,
    /// Number of blocks 10KB - 100KB
    pub medium: usize,
    /// Number of blocks 100KB - 1MB
    pub large: usize,
    /// Number of blocks > 1MB
    pub huge: usize,
}

/// Repository analyzer
pub struct RepoAnalyzer<S: BlockStoreTrait> {
    storage: Arc<S>,
    pin_manager: Arc<PinManager>,
}

impl<S: BlockStoreTrait> RepoAnalyzer<S> {
    /// Create a new repository analyzer
    pub fn new(storage: Arc<S>, pin_manager: Arc<PinManager>) -> Self {
        Self {
            storage,
            pin_manager,
        }
    }

    /// Get comprehensive repository statistics
    pub async fn analyze(&self) -> Result<RepoStats> {
        let all_cids = self.storage.list_cids()?;
        let num_blocks = all_cids.len();

        if num_blocks == 0 {
            return Ok(RepoStats {
                num_blocks: 0,
                repo_size: 0,
                num_pinned: 0,
                num_unpinned: 0,
                pinned_size: 0,
                unpinned_size: 0,
                avg_block_size: 0,
                max_block_size: 0,
                min_block_size: 0,
                storage_path: "".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            });
        }

        let mut total_size = 0u64;
        let mut pinned_size = 0u64;
        let mut unpinned_size = 0u64;
        let mut num_pinned = 0usize;
        let mut num_unpinned = 0usize;
        let mut max_size = 0u64;
        let mut min_size = u64::MAX;

        for cid in &all_cids {
            if let Some(block) = self.storage.get(cid).await? {
                let size = block.data().len() as u64;
                total_size += size;

                if size > max_size {
                    max_size = size;
                }
                if size < min_size {
                    min_size = size;
                }

                if self.pin_manager.is_pinned(cid) {
                    num_pinned += 1;
                    pinned_size += size;
                } else {
                    num_unpinned += 1;
                    unpinned_size += size;
                }
            }
        }

        let avg_size = if num_blocks > 0 {
            total_size / num_blocks as u64
        } else {
            0
        };

        Ok(RepoStats {
            num_blocks,
            repo_size: total_size,
            num_pinned,
            num_unpinned,
            pinned_size,
            unpinned_size,
            avg_block_size: avg_size,
            max_block_size: max_size,
            min_block_size: if min_size == u64::MAX { 0 } else { min_size },
            storage_path: "".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        })
    }

    /// Get block size distribution
    pub async fn block_distribution(&self) -> Result<BlockDistribution> {
        let all_cids = self.storage.list_cids()?;
        let mut dist = BlockDistribution {
            tiny: 0,
            small: 0,
            medium: 0,
            large: 0,
            huge: 0,
        };

        for cid in all_cids {
            if let Some(block) = self.storage.get(&cid).await? {
                let size = block.data().len();
                match size {
                    0..=1024 => dist.tiny += 1,
                    1025..=10240 => dist.small += 1,
                    10241..=102400 => dist.medium += 1,
                    102401..=1048576 => dist.large += 1,
                    _ => dist.huge += 1,
                }
            }
        }

        Ok(dist)
    }

    /// Find duplicate blocks (same content, different CIDs - shouldn't happen but useful for debugging)
    pub async fn find_duplicates(&self) -> Result<Vec<Vec<Cid>>> {
        let all_cids = self.storage.list_cids()?;
        let mut content_map: HashMap<Vec<u8>, Vec<Cid>> = HashMap::new();

        for cid in all_cids {
            if let Some(block) = self.storage.get(&cid).await? {
                content_map
                    .entry(block.data().to_vec())
                    .or_default()
                    .push(cid);
            }
        }

        // Filter to only groups with more than one CID
        let duplicates: Vec<Vec<Cid>> = content_map
            .into_values()
            .filter(|cids| cids.len() > 1)
            .collect();

        Ok(duplicates)
    }

    /// Get largest blocks
    pub async fn largest_blocks(&self, limit: usize) -> Result<Vec<(Cid, u64)>> {
        let all_cids = self.storage.list_cids()?;
        let mut blocks_with_sizes = Vec::new();

        for cid in all_cids {
            if let Some(block) = self.storage.get(&cid).await? {
                blocks_with_sizes.push((cid, block.data().len() as u64));
            }
        }

        // Sort by size descending
        blocks_with_sizes.sort_by_key(|a| std::cmp::Reverse(a.1));

        // Take top N
        blocks_with_sizes.truncate(limit);

        Ok(blocks_with_sizes)
    }

    /// Find orphaned blocks (not pinned and not referenced)
    pub async fn find_orphaned_blocks(&self) -> Result<Vec<Cid>> {
        let all_cids = self.storage.list_cids()?;
        let mut orphaned = Vec::new();

        for cid in all_cids {
            if !self.pin_manager.is_pinned(&cid) {
                orphaned.push(cid);
            }
        }

        Ok(orphaned)
    }
}

/// Format bytes to human-readable string
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    if unit_idx == 0 {
        format!("{} {}", bytes, UNITS[unit_idx])
    } else {
        format!("{:.2} {}", size, UNITS[unit_idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pin::PinType;
    use bytes::Bytes;
    use ipfrs_core::Block;
    use ipfrs_storage::{BlockStoreConfig, SledBlockStore};

    #[tokio::test]
    async fn test_repo_analyzer_empty() {
        let path =
            std::env::temp_dir().join(format!("ipfrs_repo_test_empty_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        let config = BlockStoreConfig {
            path: path.clone(),
            ..Default::default()
        };
        let storage = Arc::new(
            SledBlockStore::new(config).expect("test: block store creation should succeed"),
        );
        let pin_manager = Arc::new(PinManager::new());
        let analyzer = RepoAnalyzer::new(storage.clone(), pin_manager.clone());

        let stats = analyzer
            .analyze()
            .await
            .expect("test: analyze should succeed");
        assert_eq!(stats.num_blocks, 0);
        assert_eq!(stats.repo_size, 0);

        // Cleanup
        let _ = std::fs::remove_dir_all(&path);
    }

    #[tokio::test]
    async fn test_repo_analyzer_basic() {
        let path =
            std::env::temp_dir().join(format!("ipfrs_repo_test_basic_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        let config = BlockStoreConfig {
            path: path.clone(),
            ..Default::default()
        };
        let storage = Arc::new(
            SledBlockStore::new(config).expect("test: block store creation should succeed"),
        );
        let pin_manager = Arc::new(PinManager::new());
        let analyzer = RepoAnalyzer::new(storage.clone(), pin_manager.clone());

        // Add some blocks
        let block1 =
            Block::new(Bytes::from(vec![0u8; 100])).expect("test: block creation should succeed");
        let block2 =
            Block::new(Bytes::from(vec![1u8; 200])).expect("test: block creation should succeed");
        let cid1 = *block1.cid();

        storage
            .put(&block1)
            .await
            .expect("test: put block1 should succeed");
        storage
            .put(&block2)
            .await
            .expect("test: put block2 should succeed");

        // Pin one block
        pin_manager
            .pin(cid1, PinType::Direct, None)
            .expect("test: pin should succeed");

        let stats = analyzer
            .analyze()
            .await
            .expect("test: analyze should succeed");
        assert_eq!(stats.num_blocks, 2);
        assert_eq!(stats.num_pinned, 1);
        assert_eq!(stats.num_unpinned, 1);
        assert_eq!(stats.repo_size, 300);

        // Cleanup
        let _ = std::fs::remove_dir_all(&path);
    }

    #[tokio::test]
    async fn test_block_distribution() {
        let path =
            std::env::temp_dir().join(format!("ipfrs_repo_test_dist_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        let config = BlockStoreConfig {
            path: path.clone(),
            ..Default::default()
        };
        let storage = Arc::new(
            SledBlockStore::new(config).expect("test: block store creation should succeed"),
        );
        let pin_manager = Arc::new(PinManager::new());
        let analyzer = RepoAnalyzer::new(storage.clone(), pin_manager.clone());

        // Add blocks of different sizes
        let tiny =
            Block::new(Bytes::from(vec![0u8; 100])).expect("test: block creation should succeed");
        let small =
            Block::new(Bytes::from(vec![1u8; 5000])).expect("test: block creation should succeed");
        let medium =
            Block::new(Bytes::from(vec![2u8; 50000])).expect("test: block creation should succeed");

        storage
            .put(&tiny)
            .await
            .expect("test: put tiny should succeed");
        storage
            .put(&small)
            .await
            .expect("test: put small should succeed");
        storage
            .put(&medium)
            .await
            .expect("test: put medium should succeed");

        let dist = analyzer
            .block_distribution()
            .await
            .expect("test: block_distribution should succeed");
        assert_eq!(dist.tiny, 1);
        assert_eq!(dist.small, 1);
        assert_eq!(dist.medium, 1);

        // Cleanup
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
    }
}
