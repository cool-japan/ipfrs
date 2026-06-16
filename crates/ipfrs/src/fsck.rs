//! Filesystem Check (fsck) - Repository Integrity Verification
//!
//! Verifies the integrity of the IPFRS repository by checking:
//! - Block CID matches content
//! - Referenced blocks exist
//! - IPLD structures are valid

use ipfrs_core::{Block, Cid, Error, Ipld, Result};
use ipfrs_storage::BlockStoreTrait;
use std::collections::HashSet;
use std::sync::Arc;

/// Result of filesystem check
#[derive(Debug, Clone)]
pub struct FsckResult {
    /// Number of blocks checked
    pub blocks_checked: u64,
    /// Number of valid blocks
    pub blocks_valid: u64,
    /// List of corrupt blocks (CID doesn't match content)
    pub blocks_corrupt: Vec<Cid>,
    /// List of missing blocks (referenced but not found)
    pub blocks_missing: Vec<Cid>,
    /// List of invalid IPLD structures
    pub ipld_invalid: Vec<Cid>,
}

impl FsckResult {
    /// Check if the repository is healthy (no corrupt or missing blocks)
    pub fn is_healthy(&self) -> bool {
        self.blocks_corrupt.is_empty() && self.blocks_missing.is_empty()
    }

    /// Get total number of issues found
    pub fn total_issues(&self) -> usize {
        self.blocks_corrupt.len() + self.blocks_missing.len() + self.ipld_invalid.len()
    }
}

/// Filesystem check configuration
#[derive(Debug, Clone)]
pub struct FsckConfig {
    /// Whether to check IPLD structure validity
    pub check_ipld: bool,
    /// Whether to verify all references exist
    pub verify_references: bool,
    /// Maximum depth to traverse for reference checking
    pub max_depth: Option<usize>,
}

impl Default for FsckConfig {
    fn default() -> Self {
        Self {
            check_ipld: true,
            verify_references: true,
            max_depth: Some(100), // Prevent infinite loops
        }
    }
}

/// Filesystem checker
pub struct FilesystemChecker<S: BlockStoreTrait> {
    storage: Arc<S>,
}

impl<S: BlockStoreTrait> FilesystemChecker<S> {
    /// Create a new filesystem checker
    pub fn new(storage: Arc<S>) -> Self {
        Self { storage }
    }

    /// Run filesystem check
    pub async fn check(&self, config: FsckConfig) -> Result<FsckResult> {
        let mut result = FsckResult {
            blocks_checked: 0,
            blocks_valid: 0,
            blocks_corrupt: Vec::new(),
            blocks_missing: Vec::new(),
            ipld_invalid: Vec::new(),
        };

        // Get all CIDs in storage
        let all_cids = self.storage.list_cids()?;
        result.blocks_checked = all_cids.len() as u64;

        let mut checked_refs = HashSet::new();

        for cid in all_cids {
            // Check block integrity
            match self.check_block(&cid).await {
                Ok(true) => {
                    result.blocks_valid += 1;

                    // If configured, check IPLD and references
                    if config.check_ipld || config.verify_references {
                        if let Some(block) = self.storage.get(&cid).await? {
                            // Try to parse as IPLD
                            match Ipld::from_dag_cbor(block.data()) {
                                Ok(ipld) => {
                                    // Check references if enabled
                                    if config.verify_references {
                                        let missing = self
                                            .check_references(
                                                &cid,
                                                &ipld,
                                                &mut checked_refs,
                                                config.max_depth,
                                            )
                                            .await?;
                                        result.blocks_missing.extend(missing);
                                    }
                                }
                                Err(_) => {
                                    // Not valid IPLD, but might be raw data
                                    // Only mark as invalid if it looks like it should be IPLD
                                    if config.check_ipld && self.looks_like_ipld(block.data()) {
                                        result.ipld_invalid.push(cid);
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(false) => {
                    result.blocks_corrupt.push(cid);
                }
                Err(_) => {
                    // Block exists but can't be read
                    result.blocks_corrupt.push(cid);
                }
            }
        }

        Ok(result)
    }

    /// Check if a single block is valid (CID matches content)
    async fn check_block(&self, cid: &Cid) -> Result<bool> {
        match self.storage.get(cid).await? {
            Some(block) => {
                // Recompute CID from data
                let recomputed_block = Block::new(block.data().clone())?;
                Ok(recomputed_block.cid() == block.cid())
            }
            None => Err(Error::NotFound(format!("Block not found: {}", cid))),
        }
    }

    /// Check if all references in an IPLD structure exist
    fn check_references<'a>(
        &'a self,
        _parent: &'a Cid,
        ipld: &'a Ipld,
        checked: &'a mut HashSet<Cid>,
        max_depth: Option<usize>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Cid>>> + 'a>> {
        Box::pin(async move {
            let mut missing = Vec::new();

            // Check depth limit
            if let Some(max) = max_depth {
                if checked.len() >= max {
                    return Ok(missing);
                }
            }

            // Extract all links
            for link_cid in ipld.links() {
                // Skip if already checked
                if checked.contains(&link_cid) {
                    continue;
                }

                checked.insert(link_cid);

                // Check if block exists
                match self.storage.has(&link_cid).await {
                    Ok(true) => {
                        // Block exists, recursively check its references
                        if let Ok(Some(block)) = self.storage.get(&link_cid).await {
                            if let Ok(child_ipld) = Ipld::from_dag_cbor(block.data()) {
                                let child_missing = self
                                    .check_references(&link_cid, &child_ipld, checked, max_depth)
                                    .await?;
                                missing.extend(child_missing);
                            }
                        }
                    }
                    Ok(false) => {
                        // Block is missing
                        missing.push(link_cid);
                    }
                    Err(_) => {
                        // Error checking, mark as missing
                        missing.push(link_cid);
                    }
                }
            }

            Ok(missing)
        })
    }

    /// Heuristic to check if data looks like IPLD
    fn looks_like_ipld(&self, data: &[u8]) -> bool {
        // Check if it looks like CBOR (starts with appropriate markers)
        if data.is_empty() {
            return false;
        }

        // CBOR major types
        let first_byte = data[0];
        let major_type = first_byte >> 5;

        // IPLD typically uses maps (5) or arrays (4)
        matches!(major_type, 4 | 5)
    }

    /// Quick check - just verify CIDs match content
    pub async fn quick_check(&self) -> Result<FsckResult> {
        let mut result = FsckResult {
            blocks_checked: 0,
            blocks_valid: 0,
            blocks_corrupt: Vec::new(),
            blocks_missing: Vec::new(),
            ipld_invalid: Vec::new(),
        };

        let all_cids = self.storage.list_cids()?;
        result.blocks_checked = all_cids.len() as u64;

        for cid in all_cids {
            match self.check_block(&cid).await {
                Ok(true) => result.blocks_valid += 1,
                Ok(false) => result.blocks_corrupt.push(cid),
                Err(_) => result.blocks_corrupt.push(cid),
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use ipfrs_storage::{BlockStoreConfig, SledBlockStore};
    use std::collections::BTreeMap;

    fn unique_fsck_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("ipfrs_fsck_{}_{}", tag, std::process::id()))
    }

    #[tokio::test]
    async fn test_fsck_healthy_repo() {
        let path = unique_fsck_path("healthy");
        let _ = std::fs::remove_dir_all(&path);
        let config = BlockStoreConfig {
            path: path.clone(),
            ..Default::default()
        };
        let storage = Arc::new(
            SledBlockStore::new(config).expect("test: block store creation should succeed"),
        );
        let fsck = FilesystemChecker::new(storage.clone());

        // Add valid blocks
        let block1 =
            Block::new(Bytes::from("test data 1")).expect("test: block creation should succeed");
        let block2 =
            Block::new(Bytes::from("test data 2")).expect("test: block creation should succeed");
        storage
            .put(&block1)
            .await
            .expect("test: put block1 should succeed");
        storage
            .put(&block2)
            .await
            .expect("test: put block2 should succeed");

        // Run check
        let result = fsck
            .check(FsckConfig::default())
            .await
            .expect("test: fsck check should succeed");

        // Should be healthy
        assert!(result.is_healthy());
        assert_eq!(result.blocks_checked, 2);
        assert_eq!(result.blocks_valid, 2);
        assert_eq!(result.total_issues(), 0);

        // Cleanup
        let _ = std::fs::remove_dir_all(&path);
    }

    #[tokio::test]
    async fn test_fsck_quick_check() {
        let path = unique_fsck_path("quick");
        let _ = std::fs::remove_dir_all(&path);
        let config = BlockStoreConfig {
            path: path.clone(),
            ..Default::default()
        };
        let storage = Arc::new(
            SledBlockStore::new(config).expect("test: block store creation should succeed"),
        );
        let fsck = FilesystemChecker::new(storage.clone());

        // Add valid block
        let block =
            Block::new(Bytes::from("test data")).expect("test: block creation should succeed");
        storage.put(&block).await.expect("test: put should succeed");

        // Run quick check
        let result = fsck
            .quick_check()
            .await
            .expect("test: quick_check should succeed");

        assert!(result.is_healthy());
        assert_eq!(result.blocks_valid, 1);

        // Cleanup
        let _ = std::fs::remove_dir_all(&path);
    }

    #[tokio::test]
    async fn test_fsck_with_links() {
        let path = unique_fsck_path("links");
        let _ = std::fs::remove_dir_all(&path);
        let config = BlockStoreConfig {
            path: path.clone(),
            ..Default::default()
        };
        let storage = Arc::new(
            SledBlockStore::new(config).expect("test: block store creation should succeed"),
        );
        let fsck = FilesystemChecker::new(storage.clone());

        // Create a block
        let block1 = Block::new(Bytes::from("referenced data"))
            .expect("test: block creation should succeed");
        let cid1 = *block1.cid();
        storage
            .put(&block1)
            .await
            .expect("test: put block1 should succeed");

        // Create an IPLD structure that references the block
        let mut map = BTreeMap::new();
        map.insert("link".to_string(), Ipld::link(cid1));
        let ipld = Ipld::Map(map);

        // Store the IPLD
        let ipld_bytes = ipld
            .to_dag_cbor()
            .expect("test: IPLD serialization should succeed");
        let block2 =
            Block::new(Bytes::from(ipld_bytes)).expect("test: block creation should succeed");
        storage
            .put(&block2)
            .await
            .expect("test: put block2 should succeed");

        // Run check
        let result = fsck
            .check(FsckConfig::default())
            .await
            .expect("test: fsck check should succeed");

        // Should be healthy (all references exist)
        assert!(result.is_healthy());
        assert_eq!(result.blocks_missing.len(), 0);

        // Cleanup
        let _ = std::fs::remove_dir_all(&path);
    }
}
