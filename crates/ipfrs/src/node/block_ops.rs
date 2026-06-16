//! Block and file operations for Node

use bytes::Bytes;
use ipfrs_core::{Block, Cid, Error, Result};
use ipfrs_storage::BlockStoreTrait;
use std::path::Path;
use std::time::Instant;
use tracing::{debug, warn};

use super::{BlockStat, Node, StorageStats};

impl Node {
    /// Add a file from the filesystem
    ///
    /// Reads the file, stores it as a block, and announces the CID to the DHT
    /// (best-effort — a network error will not cause this method to fail).
    pub async fn add_file(&mut self, path: impl AsRef<Path>) -> Result<Cid> {
        let storage = self.storage()?;

        let data = tokio::fs::read(path.as_ref()).await?;

        let block = Block::new(Bytes::from(data))?;
        let cid = *block.cid();

        storage.put(&block).await?;

        // Announce to DHT (best-effort)
        if let Some(network) = &mut self.network {
            match network.provide(&cid).await {
                Ok(()) => debug!("Announced {} to DHT", cid),
                Err(e) => warn!("Failed to announce {} to DHT: {}", cid, e),
            }
        }

        Ok(cid)
    }

    /// Add bytes directly to storage
    ///
    /// Stores the data as a block and announces the CID to the DHT
    /// (best-effort — a network error will not cause this method to fail).
    ///
    /// Uses write-time deduplication: if the block already exists the write
    /// is skipped and a debug message is emitted, saving I/O and disk space.
    pub async fn add_bytes(&mut self, data: impl Into<Bytes>) -> Result<Cid> {
        let data_bytes = data.into();
        let byte_len = data_bytes.len() as f64;
        let storage = self.storage()?;

        let block = Block::new(data_bytes)?;
        let cid = *block.cid();

        let written = storage.inner().put_if_absent(&block).await?;
        if !written {
            debug!("Duplicate block skipped (dedup): {}", cid);
        } else {
            // Instrument: count bytes written and blocks added
            self.metrics.blocks_added.inc();
            self.metrics.block_add_bytes.inc_by(byte_len);
        }

        // Announce to DHT (best-effort)
        if let Some(network) = &mut self.network {
            self.metrics.dht_provide_calls.inc();
            match network.provide(&cid).await {
                Ok(()) => debug!("Announced {} to DHT", cid),
                Err(e) => warn!("Failed to announce {} to DHT: {}", cid, e),
            }
        }

        Ok(cid)
    }

    /// Add content from an async reader
    ///
    /// Reads all data from the provided reader, stores it as a block, and returns the CID.
    /// This is useful for streaming data from files, network streams, or other async sources.
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    /// use tokio::fs::File;
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// let file = File::open("data.bin").await?;
    /// let cid = node.add_reader(file).await?;
    /// println!("Stored with CID: {}", cid);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn add_reader<R>(&mut self, mut reader: R) -> Result<Cid>
    where
        R: tokio::io::AsyncRead + Unpin,
    {
        use tokio::io::AsyncReadExt;

        let storage = self.storage()?;

        // Read all data into a buffer
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer).await?;

        let block = Block::new(Bytes::from(buffer))?;
        let cid = *block.cid();

        storage.put(&block).await?;

        // Announce to DHT (best-effort)
        if let Some(network) = &mut self.network {
            match network.provide(&cid).await {
                Ok(()) => debug!("Announced {} to DHT", cid),
                Err(e) => warn!("Failed to announce {} to DHT: {}", cid, e),
            }
        }

        Ok(cid)
    }

    /// Get content by CID
    ///
    /// First checks local storage. If not found locally and the network is
    /// available, queries the DHT for providers and attempts to fetch the block
    /// from a remote peer (best-effort).
    pub async fn get(&mut self, cid: &Cid) -> Result<Option<Bytes>> {
        let fetch_start = Instant::now();
        let storage = self.storage()?;

        if let Some(block) = storage.get(cid).await? {
            let elapsed = fetch_start.elapsed().as_secs_f64();
            self.metrics.blocks_fetched.inc();
            self.metrics.block_fetch_latency.observe(elapsed);
            return Ok(Some(block.data().clone()));
        }

        // Not found locally — attempt DHT provider discovery
        if self.network.is_some() {
            let timeout = self.config.fetch_timeout_secs.unwrap_or(30);
            debug!(
                "Block {} not found locally; querying DHT for providers (timeout={}s)",
                cid, timeout
            );
            self.metrics.dht_find_providers_calls.inc();
            match self.find_providers(cid).await {
                Ok(providers) if !providers.is_empty() => {
                    debug!(
                        "Found {} DHT providers for {}; attempting remote fetch",
                        providers.len(),
                        cid
                    );
                    // Try up to 3 providers
                    for provider in providers.iter().take(3) {
                        if let Some(network) = &mut self.network {
                            match network.fetch_block_from_peer(provider, cid).await {
                                Ok(block) => {
                                    // Store locally for future access
                                    if let Some(storage) = &self.storage {
                                        if let Err(e) = storage.put(&block).await {
                                            warn!("Failed to cache fetched block {}: {}", cid, e);
                                        }
                                    }
                                    let elapsed = fetch_start.elapsed().as_secs_f64();
                                    self.metrics.blocks_fetched.inc();
                                    self.metrics.block_fetch_latency.observe(elapsed);
                                    return Ok(Some(block.data().clone()));
                                }
                                Err(e) => {
                                    debug!(
                                        "Failed to fetch {} from provider {}: {}",
                                        cid, provider, e
                                    );
                                }
                            }
                        }
                    }
                    warn!(
                        "Block {} found in DHT but could not be fetched from any provider",
                        cid
                    );
                }
                Ok(_) => {
                    debug!("No DHT providers found for {}", cid);
                }
                Err(e) => {
                    warn!("DHT provider query for {} failed: {}", cid, e);
                }
            }
        }

        Ok(None)
    }

    /// Get a byte range from content
    ///
    /// Retrieves a specific byte range from a block, similar to HTTP 206 Partial Content.
    /// This is useful for streaming large files or implementing range requests.
    ///
    /// # Parameters
    /// - `cid`: The content identifier
    /// - `offset`: Starting byte position (0-indexed)
    /// - `length`: Number of bytes to read (None for all remaining bytes)
    ///
    /// # Returns
    /// - `Ok(Some(bytes))` - The requested byte range
    /// - `Ok(None)` - Block not found
    /// - `Err(_)` - Invalid range or other error
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// # let cid = ipfrs_core::Cid::default();
    /// // Get bytes 100-199 (100 bytes starting at offset 100)
    /// if let Some(data) = node.get_range(&cid, 100, Some(100)).await? {
    ///     println!("Retrieved {} bytes", data.len());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_range(
        &self,
        cid: &Cid,
        offset: usize,
        length: Option<usize>,
    ) -> Result<Option<Bytes>> {
        let storage = self.storage()?;

        match storage.get(cid).await? {
            Some(block) => {
                let data = block.data();
                let total_len = data.len();

                // Validate offset
                if offset >= total_len {
                    return Err(Error::InvalidData(format!(
                        "Offset {} is beyond block size {}",
                        offset, total_len
                    )));
                }

                // Calculate end position
                let end = match length {
                    Some(len) => std::cmp::min(offset + len, total_len),
                    None => total_len,
                };

                // Extract range
                let range_data = data.slice(offset..end);
                Ok(Some(range_data))
            }
            None => Ok(None),
        }
    }

    /// Get content and write to file
    pub async fn get_to_file(&mut self, cid: &Cid, path: impl AsRef<Path>) -> Result<()> {
        match self.get(cid).await? {
            Some(data) => {
                tokio::fs::write(path.as_ref(), data).await?;
                Ok(())
            }
            None => Err(Error::NotFound(format!("Block not found: {}", cid))),
        }
    }

    /// Add a directory recursively
    ///
    /// Traverses a directory tree, stores all files as blocks, and creates
    /// a directory structure using IPLD. Returns the root CID.
    ///
    /// # Directory Structure
    /// Directories are stored as IPLD maps where:
    /// - Keys are file/directory names
    /// - Values are either:
    ///   - Links to file blocks (for files)
    ///   - Nested maps (for subdirectories)
    ///
    /// # Example
    /// ```rust,ignore
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// let root_cid = node.add_directory("/path/to/directory").await?;
    /// println!("Directory stored with root CID: {}", root_cid);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn add_directory(&mut self, dir_path: impl AsRef<Path>) -> Result<Cid> {
        use std::collections::BTreeMap;

        let dir_path = dir_path.as_ref().to_path_buf();

        if !dir_path.is_dir() {
            return Err(Error::InvalidData(format!(
                "Path is not a directory: {}",
                dir_path.display()
            )));
        }

        let mut entries = BTreeMap::new();
        let mut read_dir = tokio::fs::read_dir(&dir_path).await?;

        // Collect entries first to avoid borrowing issues in the loop
        let mut dir_entries = Vec::new();
        while let Some(entry) = read_dir.next_entry().await? {
            dir_entries.push(entry);
        }

        for entry in dir_entries {
            let file_name = entry
                .file_name()
                .to_str()
                .ok_or_else(|| {
                    Error::InvalidData(format!("Invalid filename: {:?}", entry.file_name()))
                })?
                .to_string();

            let file_path = entry.path();
            let metadata = entry.metadata().await?;

            if metadata.is_file() {
                // Store file as block and create link
                let cid = self.add_file(&file_path).await?;
                entries.insert(file_name, ipfrs_core::Ipld::link(cid));
            } else if metadata.is_dir() {
                // Recursively add subdirectory
                let subdir_cid = self.add_directory(&file_path).await?;
                entries.insert(file_name, ipfrs_core::Ipld::link(subdir_cid));
            }
            // Skip other file types (symlinks, etc.)
        }

        // Store directory as IPLD map
        let dir_ipld = ipfrs_core::Ipld::Map(entries);
        self.dag_put(dir_ipld).await
    }

    /// Get a directory and write all files to the filesystem
    ///
    /// Retrieves a directory DAG from storage and recreates the directory
    /// structure on the filesystem.
    ///
    /// # Example
    /// ```rust,ignore
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// # let dir_cid = ipfrs_core::Cid::default();
    /// node.get_directory(&dir_cid, "/path/to/output").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_directory(&mut self, cid: &Cid, output_path: impl AsRef<Path>) -> Result<()> {
        let output_path = output_path.as_ref().to_path_buf();

        // Create output directory
        tokio::fs::create_dir_all(&output_path).await?;

        // Get directory IPLD
        let dir_ipld = self
            .dag_get(cid)
            .await?
            .ok_or_else(|| Error::NotFound(format!("Directory not found: {}", cid)))?;

        match dir_ipld {
            ipfrs_core::Ipld::Map(entries) => {
                for (name, value) in entries {
                    let entry_path = output_path.join(&name);

                    match value {
                        ipfrs_core::Ipld::Link(link) => {
                            // Try to determine if it's a file or directory
                            // by checking if it's a map
                            if let Some(ipld) = self.dag_get(&link.0).await? {
                                match ipld {
                                    ipfrs_core::Ipld::Map(_) => {
                                        // It's a directory
                                        self.get_directory(&link.0, &entry_path).await?;
                                    }
                                    _ => {
                                        // It's a file - get the raw bytes
                                        if let Some(bytes) = self.get(&link.0).await? {
                                            tokio::fs::write(&entry_path, bytes).await?;
                                        }
                                    }
                                }
                            }
                        }
                        _ => {
                            // Unexpected value type in directory
                            return Err(Error::InvalidData(format!(
                                "Unexpected value type in directory for entry: {}",
                                name
                            )));
                        }
                    }
                }
                Ok(())
            }
            _ => Err(Error::InvalidData(format!(
                "CID does not point to a directory structure: {}",
                cid
            ))),
        }
    }

    // ==================================================================
    // Raw Block Operations
    // ==================================================================

    /// Store a raw block
    pub async fn put_block(&self, block: &Block) -> Result<()> {
        let storage = self.storage()?;
        storage.put(block).await
    }

    /// Store multiple blocks atomically
    pub async fn put_blocks(&self, blocks: &[Block]) -> Result<()> {
        let storage = self.storage()?;
        storage.put_many(blocks).await
    }

    /// Retrieve a block by CID
    pub async fn get_block(&self, cid: &Cid) -> Result<Option<Block>> {
        let storage = self.storage()?;
        storage.get(cid).await
    }

    /// Retrieve multiple blocks
    pub async fn get_blocks(&self, cids: &[Cid]) -> Result<Vec<Option<Block>>> {
        let storage = self.storage()?;
        storage.get_many(cids).await
    }

    /// Check if a block exists
    pub async fn has_block(&self, cid: &Cid) -> Result<bool> {
        let storage = self.storage()?;
        storage.has(cid).await
    }

    /// Check if multiple blocks exist
    pub async fn has_blocks(&self, cids: &[Cid]) -> Result<Vec<bool>> {
        let storage = self.storage()?;
        storage.has_many(cids).await
    }

    /// Delete a block
    pub async fn delete_block(&self, cid: &Cid) -> Result<()> {
        let storage = self.storage()?;
        storage.delete(cid).await
    }

    /// Delete multiple blocks
    pub async fn delete_blocks(&self, cids: &[Cid]) -> Result<()> {
        let storage = self.storage()?;
        storage.delete_many(cids).await
    }

    /// Get detailed statistics about a block
    ///
    /// Returns comprehensive information about a block including its size,
    /// CID details, and storage metadata.
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// # let cid = ipfrs_core::Cid::default();
    /// if let Some(stat) = node.block_stat(&cid).await? {
    ///     println!("Block size: {} bytes", stat.size);
    ///     println!("CID: {}", stat.cid);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn block_stat(&self, cid: &Cid) -> Result<Option<BlockStat>> {
        let storage = self.storage()?;

        match storage.get(cid).await? {
            Some(block) => Ok(Some(BlockStat {
                cid: *cid,
                size: block.data().len(),
            })),
            None => Ok(None),
        }
    }

    /// Remove a block from storage
    ///
    /// Removes a block if it's safe to do so. This method checks pinning status
    /// and refuses to remove pinned blocks to prevent accidental data loss.
    ///
    /// # Safety
    /// This operation is irreversible. The block will be permanently deleted
    /// from storage. Pinned blocks are protected and cannot be removed until unpinned.
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// # let cid = ipfrs_core::Cid::default();
    /// node.block_rm(&cid).await?;
    /// println!("Block removed");
    /// # Ok(())
    /// # }
    /// ```
    pub async fn block_rm(&self, cid: &Cid) -> Result<()> {
        // Check if block is pinned
        if self.pin_manager.is_pinned(cid) {
            return Err(Error::InvalidInput(format!(
                "Cannot remove pinned block: {}. Unpin it first.",
                cid
            )));
        }
        self.delete_block(cid).await
    }

    /// List all CIDs in storage
    pub fn list_blocks(&self) -> Result<Vec<Cid>> {
        let storage = self.storage()?;
        storage.list_cids()
    }

    // ==================================================================
    // Statistics & Management
    // ==================================================================

    /// Get storage statistics including deduplication counters
    pub fn storage_stats(&self) -> Result<StorageStats> {
        let storage = self.storage()?;
        let num_blocks = storage.len();

        // Sync storage gauges so the metrics endpoint reflects current reality.
        self.metrics.storage_blocks_total.set(num_blocks as f64);

        Ok(StorageStats {
            num_blocks,
            is_empty: storage.is_empty(),
            dedup: storage.inner().dedup_stats().snapshot(),
        })
    }

    /// Return a point-in-time snapshot of the L1 block cache statistics.
    ///
    /// This exposes hits, misses, evictions and the derived hit-rate for the
    /// in-process LRU cache that wraps the underlying Sled block store.
    pub fn cache_stats(&self) -> Result<ipfrs_storage::CacheStatsSnapshot> {
        let storage = self.storage()?;
        Ok(storage.stats())
    }

    /// Flush pending writes to disk
    pub async fn flush(&self) -> Result<()> {
        let storage = self.storage()?;
        storage.flush().await
    }

    /// Retrieve the raw bytes of a block by CID.
    ///
    /// Returns `None` when the block is not found locally (no network fallback).
    /// This is a low-level counterpart to `get()` that skips DHT lookup.
    pub async fn get_block_raw(&self, cid: &Cid) -> Result<Option<Vec<u8>>> {
        let storage = self.storage()?;
        match storage.get(cid).await? {
            Some(block) => Ok(Some(block.data().to_vec())),
            None => Ok(None),
        }
    }

    /// Store raw bytes as a block and return the resulting CID.
    ///
    /// The CID is derived from the content using the raw codec and SHA2-256 hash.
    pub async fn put_block_raw(&mut self, data: Vec<u8>) -> Result<Cid> {
        self.add_bytes(bytes::Bytes::from(data)).await
    }
}

#[cfg(test)]
mod tests {
    use crate::node::core::NodeConfig;
    use crate::node::Node;
    use ipfrs_storage::BlockStoreConfig;

    fn unique_test_dir(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "ipfrs-node-block-ops-{}-{}",
            tag,
            std::process::id()
        ))
    }

    /// `test_put_if_absent_dedup`:
    /// Put the same block twice via `add_bytes()`, verify it is stored once,
    /// and that the deduplication stats show 1 duplicate.
    #[tokio::test]
    async fn test_put_if_absent_dedup() {
        let storage_path = unique_test_dir("dedup");
        let _ = std::fs::remove_dir_all(&storage_path);

        let config = NodeConfig {
            storage: BlockStoreConfig {
                path: storage_path.clone(),
                cache_size: 1024 * 1024,
            },
            enable_semantic: false,
            enable_tensorlogic: false,
            ..NodeConfig::default()
        };

        let mut node = Node::new(config).expect("test: node creation should succeed");
        // Use storage directly without starting network.
        // Wrap SledBlockStore in CachedBlockStore to match NodeStore type.
        let sled_store = ipfrs_storage::SledBlockStore::new(ipfrs_storage::BlockStoreConfig {
            path: storage_path.clone(),
            cache_size: 1024 * 1024,
        })
        .expect("test: sled store creation should succeed");
        let storage = ipfrs_storage::CachedBlockStore::with_default_config(sled_store);
        node.storage = Some(std::sync::Arc::new(storage));

        let data = bytes::Bytes::from("hello dedup world");
        let block =
            ipfrs_core::Block::new(data.clone()).expect("test: block creation should succeed");

        // First write – new block (accessed via inner() to reach SledBlockStore).
        let written1 = node
            .storage()
            .expect("test: storage should be set")
            .inner()
            .put_if_absent(&block)
            .await
            .expect("test: put_if_absent should succeed");
        assert!(written1, "first write must be stored");

        // Second write – duplicate, must be skipped.
        let written2 = node
            .storage()
            .expect("test: storage should be set")
            .inner()
            .put_if_absent(&block)
            .await
            .expect("test: put_if_absent should succeed");
        assert!(!written2, "duplicate write must be skipped");

        // Stats must reflect exactly one dedup hit.
        let stats = node
            .storage()
            .expect("test: storage should be set")
            .inner()
            .dedup_stats()
            .snapshot();
        assert_eq!(stats.total_puts, 2);
        assert_eq!(stats.deduplicated, 1);
        assert!(stats.bytes_saved > 0);

        let _ = std::fs::remove_dir_all(&storage_path);
    }

    /// `test_node_config_fetch_timeout`:
    /// Verify that `NodeConfig { fetch_timeout_secs: Some(5), .. }` is
    /// constructed correctly and the value is accessible.
    #[test]
    fn test_node_config_fetch_timeout() {
        let config = NodeConfig {
            fetch_timeout_secs: Some(5),
            ..NodeConfig::default()
        };

        assert_eq!(
            config.fetch_timeout_secs,
            Some(5),
            "fetch_timeout_secs must propagate through NodeConfig"
        );

        // Default must be None (sentinel for 30 s)
        let default_config = NodeConfig::default();
        assert_eq!(
            default_config.fetch_timeout_secs, None,
            "default fetch_timeout_secs must be None"
        );

        // A Node constructed from such config must hold the value
        let node = Node::new(config).expect("test: node creation should succeed");
        assert_eq!(node.config.fetch_timeout_secs, Some(5));
    }
}
