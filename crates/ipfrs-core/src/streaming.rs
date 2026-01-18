//! Streaming support for reading and writing blocks
//!
//! This module provides async streaming capabilities for block data,
//! allowing efficient reading of chunked files and DAG structures.

use crate::block::Block;
use crate::chunking::{DagLink, DagNode};
use crate::cid::Cid;
use crate::error::{Error, Result};
use bytes::Bytes;
use futures::future::BoxFuture;
use std::collections::VecDeque;
use std::io::{self, Read};
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::io::{AsyncRead, ReadBuf};

/// Block fetcher trait for retrieving blocks by CID
pub trait BlockFetcher: Send + Sync {
    /// Fetch a block by its CID
    fn fetch(&self, cid: Cid) -> BoxFuture<'_, Result<Block>>;
}

/// A simple in-memory block fetcher for testing
pub struct MemoryBlockFetcher {
    blocks: std::collections::HashMap<Cid, Block>,
}

impl MemoryBlockFetcher {
    /// Create a new empty memory block fetcher
    pub fn new() -> Self {
        Self {
            blocks: std::collections::HashMap::new(),
        }
    }

    /// Add a block to the fetcher
    pub fn add_block(&mut self, block: Block) {
        self.blocks.insert(*block.cid(), block);
    }

    /// Add multiple blocks
    pub fn add_blocks(&mut self, blocks: impl IntoIterator<Item = Block>) {
        for block in blocks {
            self.add_block(block);
        }
    }
}

impl Default for MemoryBlockFetcher {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockFetcher for MemoryBlockFetcher {
    fn fetch(&self, cid: Cid) -> BoxFuture<'_, Result<Block>> {
        let result = self
            .blocks
            .get(&cid)
            .cloned()
            .ok_or_else(|| Error::BlockNotFound(cid.to_string()));
        Box::pin(async move { result })
    }
}

/// Synchronous block reader for reading raw block data
pub struct BlockReader {
    data: Bytes,
    position: usize,
}

impl BlockReader {
    /// Create a new block reader
    pub fn new(block: &Block) -> Self {
        Self {
            data: block.data().clone(),
            position: 0,
        }
    }

    /// Create from raw bytes
    pub fn from_bytes(data: Bytes) -> Self {
        Self { data, position: 0 }
    }

    /// Get the remaining bytes to read
    pub fn remaining(&self) -> usize {
        self.data.len() - self.position
    }

    /// Check if we've reached the end
    pub fn is_empty(&self) -> bool {
        self.position >= self.data.len()
    }

    /// Get total size
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl Read for BlockReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.position >= self.data.len() {
            return Ok(0);
        }

        let remaining = &self.data[self.position..];
        let to_read = std::cmp::min(buf.len(), remaining.len());
        buf[..to_read].copy_from_slice(&remaining[..to_read]);
        self.position += to_read;
        Ok(to_read)
    }
}

/// Async block reader implementing AsyncRead
pub struct AsyncBlockReader {
    data: Bytes,
    position: usize,
}

impl AsyncBlockReader {
    /// Create a new async block reader
    pub fn new(block: &Block) -> Self {
        Self {
            data: block.data().clone(),
            position: 0,
        }
    }

    /// Create from raw bytes
    pub fn from_bytes(data: Bytes) -> Self {
        Self { data, position: 0 }
    }

    /// Get the remaining bytes to read
    pub fn remaining(&self) -> usize {
        self.data.len() - self.position
    }

    /// Check if we've reached the end
    pub fn is_empty(&self) -> bool {
        self.position >= self.data.len()
    }

    /// Get total size
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl AsyncRead for AsyncBlockReader {
    fn poll_read(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        if self.position >= self.data.len() {
            return Poll::Ready(Ok(()));
        }

        let remaining = &self.data[self.position..];
        let to_read = std::cmp::min(buf.remaining(), remaining.len());
        buf.put_slice(&remaining[..to_read]);
        self.position += to_read;
        Poll::Ready(Ok(()))
    }
}

/// State for the DAG stream reader
#[allow(dead_code)]
enum DagReaderState {
    /// Reading from current chunk
    Reading { data: Bytes, position: usize },
    /// Need to fetch next chunk
    FetchingNext,
    /// Finished reading
    Done,
}

/// Async reader for DAG-structured data
///
/// Reads data from a DAG by traversing links and concatenating leaf data.
#[allow(dead_code)]
pub struct DagStreamReader<F: BlockFetcher> {
    fetcher: std::sync::Arc<F>,
    state: DagReaderState,
    pending_links: VecDeque<DagLink>,
    total_read: u64,
}

impl<F: BlockFetcher> DagStreamReader<F> {
    /// Create a new DAG stream reader
    pub fn new(fetcher: std::sync::Arc<F>, root_links: Vec<DagLink>) -> Self {
        Self {
            fetcher,
            state: DagReaderState::FetchingNext,
            pending_links: root_links.into(),
            total_read: 0,
        }
    }

    /// Create from a DAG node
    pub fn from_node(fetcher: std::sync::Arc<F>, node: &DagNode) -> Self {
        if let Some(data) = &node.data {
            // Leaf node - read directly
            Self {
                fetcher,
                state: DagReaderState::Reading {
                    data: Bytes::from(data.clone()),
                    position: 0,
                },
                pending_links: VecDeque::new(),
                total_read: 0,
            }
        } else {
            // Intermediate node - traverse links
            Self::new(fetcher, node.links.clone())
        }
    }

    /// Get the total bytes read so far
    pub fn bytes_read(&self) -> u64 {
        self.total_read
    }
}

/// Async stream that yields chunks of data from a DAG
pub struct DagChunkStream<F: BlockFetcher> {
    fetcher: std::sync::Arc<F>,
    pending_links: VecDeque<DagLink>,
}

impl<F: BlockFetcher> DagChunkStream<F> {
    /// Create a new chunk stream from links
    pub fn new(fetcher: std::sync::Arc<F>, links: Vec<DagLink>) -> Self {
        Self {
            fetcher,
            pending_links: links.into(),
        }
    }

    /// Fetch the next chunk (non-recursive implementation)
    pub async fn next_chunk(&mut self) -> Option<Result<Bytes>> {
        loop {
            let link = self.pending_links.pop_front()?;

            match self.fetcher.fetch(link.cid.0).await {
                Ok(block) => {
                    // Check if this is a leaf or intermediate node
                    // Raw codec (0x55) means leaf data
                    if block.cid().codec() == 0x55 {
                        return Some(Ok(block.data().clone()));
                    }

                    // DAG-CBOR - need to parse and expand links
                    match crate::ipld::Ipld::from_dag_cbor(block.data()) {
                        Ok(ipld) => {
                            if let crate::ipld::Ipld::Map(map) = ipld {
                                // Check for links to add
                                if let Some(crate::ipld::Ipld::List(links)) = map.get("links") {
                                    // Add child links to the front of the queue (in reverse order)
                                    let mut new_links = Vec::new();
                                    for link_ipld in links {
                                        if let crate::ipld::Ipld::Map(link_map) = link_ipld {
                                            if let (
                                                Some(crate::ipld::Ipld::Link(cid)),
                                                Some(crate::ipld::Ipld::Integer(size)),
                                            ) = (link_map.get("cid"), link_map.get("size"))
                                            {
                                                new_links.push(DagLink::new(cid.0, *size as u64));
                                            }
                                        }
                                    }
                                    // Prepend new links
                                    for new_link in new_links.into_iter().rev() {
                                        self.pending_links.push_front(new_link);
                                    }
                                }

                                // Check for data in this node
                                if let Some(crate::ipld::Ipld::Bytes(data)) = map.get("data") {
                                    return Some(Ok(Bytes::from(data.clone())));
                                }
                            }
                            // Continue to next link
                        }
                        Err(e) => return Some(Err(e)),
                    }
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }

    /// Check if there are more chunks to read
    pub fn has_more(&self) -> bool {
        !self.pending_links.is_empty()
    }
}

/// Read all data from a chunked file
pub async fn read_chunked_file<F: BlockFetcher>(fetcher: &F, root_cid: &Cid) -> Result<Vec<u8>> {
    let root_block = fetcher.fetch(*root_cid).await?;

    // Check if it's a single block or a DAG
    if root_block.cid().codec() == 0x55 {
        // Raw codec - single block
        return Ok(root_block.data().to_vec());
    }

    // Parse the DAG node and collect all data using queue for correct ordering
    let root_ipld = crate::ipld::Ipld::from_dag_cbor(root_block.data())?;

    let mut result = Vec::new();
    let mut queue: VecDeque<crate::ipld::Ipld> = VecDeque::new();
    queue.push_back(root_ipld);

    while let Some(ipld) = queue.pop_front() {
        if let crate::ipld::Ipld::Map(map) = ipld {
            // Check for data in this node first
            if let Some(crate::ipld::Ipld::Bytes(data)) = map.get("data") {
                result.extend_from_slice(data);
            }

            // Process links in order
            if let Some(crate::ipld::Ipld::List(links)) = map.get("links") {
                for link_ipld in links {
                    if let crate::ipld::Ipld::Map(link_map) = link_ipld {
                        if let Some(crate::ipld::Ipld::Link(cid)) = link_map.get("cid") {
                            let block = fetcher.fetch(cid.0).await?;

                            if block.cid().codec() == 0x55 {
                                // Raw codec - leaf data
                                result.extend_from_slice(block.data());
                            } else {
                                // DAG-CBOR - parse and add to queue
                                let child_ipld = crate::ipld::Ipld::from_dag_cbor(block.data())?;
                                queue.push_back(child_ipld);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunking::{Chunker, ChunkingConfig};
    use std::io::Read;

    #[test]
    fn test_block_reader() {
        let block = Block::new(Bytes::from_static(b"Hello, World!")).unwrap();
        let mut reader = BlockReader::new(&block);

        let mut buf = [0u8; 5];
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 5);
        assert_eq!(&buf, b"Hello");

        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 5);
        assert_eq!(&buf, b", Wor");

        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 3);
        assert_eq!(&buf[..3], b"ld!");
    }

    #[tokio::test]
    async fn test_async_block_reader() {
        use tokio::io::AsyncReadExt;

        let block = Block::new(Bytes::from_static(b"Hello, World!")).unwrap();
        let mut reader = AsyncBlockReader::new(&block);

        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).await.unwrap();
        assert_eq!(buf, b"Hello, World!");
    }

    #[tokio::test]
    async fn test_memory_block_fetcher() {
        let block = Block::new(Bytes::from_static(b"test data")).unwrap();
        let cid = *block.cid();

        let mut fetcher = MemoryBlockFetcher::new();
        fetcher.add_block(block.clone());

        let fetched = fetcher.fetch(cid).await.unwrap();
        assert_eq!(fetched.data(), block.data());
    }

    #[tokio::test]
    async fn test_read_single_block_file() {
        let data = b"Hello, IPFS!";
        let block = Block::new(Bytes::from_static(data)).unwrap();
        let cid = *block.cid();

        let mut fetcher = MemoryBlockFetcher::new();
        fetcher.add_block(block);

        let result = read_chunked_file(&fetcher, &cid).await.unwrap();
        assert_eq!(result, data);
    }

    #[tokio::test]
    async fn test_read_chunked_file() {
        // Create chunked data
        let config = ChunkingConfig::with_chunk_size(1024).unwrap();
        let chunker = Chunker::with_config(config);

        let data: Vec<u8> = (0..3000).map(|i| (i % 256) as u8).collect();
        let chunked = chunker.chunk(&data).unwrap();

        // Add all blocks to fetcher
        let mut fetcher = MemoryBlockFetcher::new();
        fetcher.add_blocks(chunked.blocks.clone());

        // Read back
        let result = read_chunked_file(&fetcher, &chunked.root_cid)
            .await
            .unwrap();
        assert_eq!(result, data);
    }

    #[tokio::test]
    async fn test_dag_chunk_stream() {
        // Create some test blocks
        let block1 = Block::new(Bytes::from_static(b"chunk1")).unwrap();
        let block2 = Block::new(Bytes::from_static(b"chunk2")).unwrap();

        let mut fetcher = MemoryBlockFetcher::new();
        fetcher.add_block(block1.clone());
        fetcher.add_block(block2.clone());

        let links = vec![
            DagLink::new(*block1.cid(), 6),
            DagLink::new(*block2.cid(), 6),
        ];

        let mut stream = DagChunkStream::new(std::sync::Arc::new(fetcher), links);

        let chunk1 = stream.next_chunk().await.unwrap().unwrap();
        assert_eq!(chunk1.as_ref(), b"chunk1");

        let chunk2 = stream.next_chunk().await.unwrap().unwrap();
        assert_eq!(chunk2.as_ref(), b"chunk2");

        assert!(stream.next_chunk().await.is_none());
    }
}
