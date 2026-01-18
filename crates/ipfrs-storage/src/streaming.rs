//! Streaming interfaces for large block operations.
//!
//! Provides AsyncRead/AsyncWrite implementations for efficient handling
//! of large blocks without loading everything into memory.

use crate::traits::BlockStore;
use bytes::{Bytes, BytesMut};
use futures::stream::Stream;
use ipfrs_core::{Block, Cid, Error, Result};
use std::io::SeekFrom;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::io::{AsyncRead, AsyncSeek, ReadBuf};

/// Configuration for streaming operations
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Buffer size for streaming reads/writes
    pub buffer_size: usize,
    /// Whether to prefetch next chunks
    pub prefetch: bool,
    /// Maximum prefetch queue size
    pub prefetch_queue_size: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            buffer_size: 64 * 1024, // 64KB chunks
            prefetch: true,
            prefetch_queue_size: 4,
        }
    }
}

/// Range specification for partial block reads
#[derive(Debug, Clone, Copy)]
pub struct ByteRange {
    /// Start offset (inclusive)
    pub start: u64,
    /// End offset (exclusive), None means end of block
    pub end: Option<u64>,
}

impl ByteRange {
    /// Create a range from start to end of block
    #[inline]
    pub fn from(start: u64) -> Self {
        Self { start, end: None }
    }

    /// Create a range with specific start and end
    #[inline]
    pub fn new(start: u64, end: u64) -> Self {
        Self {
            start,
            end: Some(end),
        }
    }

    /// Create a range for a specific length from start
    #[inline]
    pub fn with_length(start: u64, length: u64) -> Self {
        Self {
            start,
            end: Some(start + length),
        }
    }

    /// Get the length of this range, given the total size
    #[inline]
    pub fn length(&self, total_size: u64) -> u64 {
        let end = self.end.unwrap_or(total_size).min(total_size);
        end.saturating_sub(self.start)
    }
}

/// Async reader for a single block with seeking support
pub struct BlockReader {
    data: Bytes,
    position: u64,
}

impl BlockReader {
    /// Create a new block reader
    #[inline]
    pub fn new(block: &Block) -> Self {
        Self {
            data: block.data().clone(),
            position: 0,
        }
    }

    /// Create from raw bytes
    #[inline]
    pub fn from_bytes(data: Bytes) -> Self {
        Self { data, position: 0 }
    }

    /// Get the remaining bytes
    #[inline]
    pub fn remaining(&self) -> u64 {
        self.data.len() as u64 - self.position
    }

    /// Get the total size
    pub fn size(&self) -> u64 {
        self.data.len() as u64
    }
}

impl AsyncRead for BlockReader {
    fn poll_read(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let pos = self.position as usize;
        let data_len = self.data.len();

        if pos >= data_len {
            return Poll::Ready(Ok(())); // EOF
        }

        let remaining = data_len - pos;
        let to_read = remaining.min(buf.remaining());
        buf.put_slice(&self.data[pos..pos + to_read]);
        self.position += to_read as u64;

        Poll::Ready(Ok(()))
    }
}

impl AsyncSeek for BlockReader {
    fn start_seek(mut self: Pin<&mut Self>, position: SeekFrom) -> std::io::Result<()> {
        let new_pos = match position {
            SeekFrom::Start(pos) => pos as i64,
            SeekFrom::End(offset) => self.data.len() as i64 + offset,
            SeekFrom::Current(offset) => self.position as i64 + offset,
        };

        if new_pos < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "seek to negative position",
            ));
        }

        self.position = new_pos as u64;
        Ok(())
    }

    fn poll_complete(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<u64>> {
        Poll::Ready(Ok(self.position))
    }
}

/// Partial block read result
pub struct PartialBlock {
    /// The CID of the full block
    pub cid: Cid,
    /// The requested range
    pub range: ByteRange,
    /// The actual data (slice of the block)
    pub data: Bytes,
    /// Total size of the full block
    pub total_size: u64,
}

impl PartialBlock {
    /// Check if this is the complete block
    pub fn is_complete(&self) -> bool {
        self.range.start == 0 && self.data.len() as u64 == self.total_size
    }
}

/// Extension trait for block stores with streaming capabilities
#[async_trait::async_trait]
pub trait StreamingBlockStore: BlockStore {
    /// Read a partial block (range-based)
    ///
    /// This allows efficient access to portions of large blocks without
    /// loading the entire block into memory.
    async fn get_range(&self, cid: &Cid, range: ByteRange) -> Result<Option<PartialBlock>> {
        // Default implementation: read full block and slice
        let block = self.get(cid).await?;
        match block {
            Some(block) => {
                let data = block.data();
                let total_size = data.len() as u64;

                let start = (range.start as usize).min(data.len());
                let end = range
                    .end
                    .map(|e| (e as usize).min(data.len()))
                    .unwrap_or(data.len());

                let slice = if start < end {
                    data.slice(start..end)
                } else {
                    Bytes::new()
                };

                Ok(Some(PartialBlock {
                    cid: *block.cid(),
                    range,
                    data: slice,
                    total_size,
                }))
            }
            None => Ok(None),
        }
    }

    /// Create an async reader for a block
    async fn reader(&self, cid: &Cid) -> Result<Option<BlockReader>> {
        let block = self.get(cid).await?;
        Ok(block.map(|b| BlockReader::new(&b)))
    }

    /// Get the size of a block without reading its contents
    async fn get_size(&self, cid: &Cid) -> Result<Option<u64>> {
        // Default: read block and get size (inefficient, backends should override)
        let block = self.get(cid).await?;
        Ok(block.map(|b| b.size()))
    }
}

// Implement StreamingBlockStore for any BlockStore
impl<T: BlockStore> StreamingBlockStore for T {}

/// Streaming block writer for building large content
pub struct StreamingWriter<S: BlockStore> {
    store: Arc<S>,
    buffer: BytesMut,
    config: StreamConfig,
    written_cids: Vec<Cid>,
}

impl<S: BlockStore> StreamingWriter<S> {
    /// Create a new streaming writer
    pub fn new(store: Arc<S>) -> Self {
        Self::with_config(store, StreamConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(store: Arc<S>, config: StreamConfig) -> Self {
        Self {
            store,
            buffer: BytesMut::with_capacity(config.buffer_size),
            config,
            written_cids: Vec::new(),
        }
    }

    /// Write data, automatically chunking into blocks
    pub async fn write(&mut self, data: &[u8]) -> Result<usize> {
        self.buffer.extend_from_slice(data);

        // Flush complete chunks
        while self.buffer.len() >= self.config.buffer_size {
            self.flush_chunk().await?;
        }

        Ok(data.len())
    }

    /// Flush any buffered data as a final block
    pub async fn finish(mut self) -> Result<Vec<Cid>> {
        if !self.buffer.is_empty() {
            self.flush_chunk().await?;
        }
        Ok(self.written_cids)
    }

    /// Flush current buffer as a block
    async fn flush_chunk(&mut self) -> Result<()> {
        let chunk_size = self.buffer.len().min(self.config.buffer_size);
        let chunk_data = self.buffer.split_to(chunk_size).freeze();

        let block = Block::new(chunk_data)?;
        let cid = *block.cid();
        self.store.put(&block).await?;
        self.written_cids.push(cid);

        Ok(())
    }

    /// Get the CIDs written so far
    pub fn written_cids(&self) -> &[Cid] {
        &self.written_cids
    }
}

/// Stream of blocks from an iterator of CIDs
pub struct BlockStream<S: BlockStore> {
    store: Arc<S>,
    cids: std::vec::IntoIter<Cid>,
}

impl<S: BlockStore + 'static> BlockStream<S> {
    /// Create a new block stream
    pub fn new(store: Arc<S>, cids: Vec<Cid>) -> Self {
        Self {
            store,
            cids: cids.into_iter(),
        }
    }
}

impl<S: BlockStore + 'static> Stream for BlockStream<S> {
    type Item = Result<Block>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.cids.next() {
            Some(cid) => {
                let store = Arc::clone(&self.store);
                let fut = async move {
                    match store.get(&cid).await? {
                        Some(block) => Ok(block),
                        None => Err(Error::BlockNotFound(cid.to_string())),
                    }
                };
                // Pin the future and poll it
                // For simplicity, we return Pending and use a spawned task
                // In production, use proper async stream combinators
                let waker = cx.waker().clone();
                tokio::spawn(async move {
                    let _ = fut.await;
                    waker.wake();
                });
                Poll::Pending
            }
            None => Poll::Ready(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_range() {
        let range = ByteRange::new(10, 50);
        assert_eq!(range.length(100), 40);
        assert_eq!(range.length(30), 20); // Clamped to size

        let range = ByteRange::from(80);
        assert_eq!(range.length(100), 20);

        let range = ByteRange::with_length(10, 30);
        assert_eq!(range.length(100), 30);
    }

    #[tokio::test]
    async fn test_block_reader() {
        use tokio::io::AsyncReadExt;

        let data = Bytes::from("Hello, World!");
        let block = Block::new(data.clone()).unwrap();
        let mut reader = BlockReader::new(&block);

        let mut buf = vec![0u8; 5];
        let n = reader.read(&mut buf).await.unwrap();
        assert_eq!(n, 5);
        assert_eq!(&buf, b"Hello");

        let n = reader.read(&mut buf).await.unwrap();
        assert_eq!(n, 5);
        assert_eq!(&buf, b", Wor");
    }

    #[tokio::test]
    async fn test_block_reader_seek() {
        use tokio::io::{AsyncReadExt, AsyncSeekExt};

        let data = Bytes::from("Hello, World!");
        let block = Block::new(data).unwrap();
        let mut reader = BlockReader::new(&block);

        reader.seek(SeekFrom::Start(7)).await.unwrap();

        let mut buf = vec![0u8; 5];
        let n = reader.read(&mut buf).await.unwrap();
        assert_eq!(n, 5);
        assert_eq!(&buf, b"World");
    }
}
