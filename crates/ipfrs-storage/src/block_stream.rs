//! Block stream iterator with backpressure for large CAR/tensor exports.
//!
//! Provides [`BlockStreamIterator`] for memory-efficient iteration over large
//! block exports.  Backpressure is applied synchronously: when the internal
//! buffer is full, [`BlockStreamIterator::next_chunk`] returns `None` until
//! the consumer calls [`BlockStreamIterator::drain_one`].

use std::collections::VecDeque;

// ── BlockChunk ────────────────────────────────────────────────────────────────

/// A single unit of work produced by [`BlockStreamIterator`].
///
/// `cids` and `data` are parallel vectors: `cids[i]` is the CID of `data[i]`.
/// When [`StreamConfig::include_data`] is `false`, `data` contains only empty
/// `Vec`s (but `cids` is still populated).
#[derive(Debug, Clone)]
pub struct BlockChunk {
    /// CID strings in this chunk.
    pub cids: Vec<String>,
    /// Block data corresponding 1-to-1 with `cids`.
    pub data: Vec<Vec<u8>>,
    /// Zero-based index of this chunk in the stream.
    pub chunk_index: u64,
    /// `true` for the last chunk of the stream.
    pub is_final: bool,
}

impl BlockChunk {
    /// Total bytes across all data vectors in this chunk.
    #[inline]
    pub fn total_bytes(&self) -> usize {
        self.data.iter().map(|d| d.len()).sum()
    }

    /// Number of blocks in this chunk.
    #[inline]
    pub fn len(&self) -> usize {
        self.cids.len()
    }

    /// Returns `true` when the chunk contains no blocks.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cids.is_empty()
    }
}

// ── StreamConfig ──────────────────────────────────────────────────────────────

/// Configuration for a [`BlockStreamIterator`].
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Number of blocks per chunk (default: 64).
    pub chunk_size: usize,
    /// Maximum number of chunks held in the internal buffer before backpressure
    /// kicks in (default: 4).
    pub max_buffer_chunks: usize,
    /// When `false`, `data` vectors in every [`BlockChunk`] are left empty;
    /// only CIDs are streamed (default: `true`).
    pub include_data: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            chunk_size: 64,
            max_buffer_chunks: 4,
            include_data: true,
        }
    }
}

// ── BlockStreamState ──────────────────────────────────────────────────────────

/// Current state of a [`BlockStreamIterator`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockStreamState {
    /// Chunks are being produced normally.
    Ready,
    /// Source blocks are exhausted; buffer is being drained.
    Draining,
    /// Buffer is full; backpressure applied until a chunk is drained.
    Paused,
    /// All blocks have been produced and drained.
    Exhausted,
    /// An error occurred.
    Error(String),
}

// ── StreamStats ───────────────────────────────────────────────────────────────

/// Cumulative statistics for a [`BlockStreamIterator`].
#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    /// Total chunks pushed into the buffer.
    pub chunks_produced: u64,
    /// Total chunks removed from the buffer by the consumer.
    pub chunks_drained: u64,
    /// Total bytes that have left the buffer (sum of `total_bytes` of drained
    /// chunks).
    pub bytes_streamed: u64,
    /// Number of times `next_chunk` returned `None` due to a full buffer.
    pub backpressure_events: u64,
}

// ── BlockStreamIterator ───────────────────────────────────────────────────────

/// Synchronous, single-threaded streaming iterator over a block collection.
///
/// # Backpressure
///
/// If the internal buffer reaches [`StreamConfig::max_buffer_chunks`] chunks,
/// [`next_chunk`](BlockStreamIterator::next_chunk) returns `None` and
/// increments [`StreamStats::backpressure_events`].  The caller must call
/// [`drain_one`](BlockStreamIterator::drain_one) before production can resume.
///
/// # Example
///
/// ```rust
/// use ipfrs_storage::block_stream::{BlockStreamIterator, StreamConfig};
///
/// let blocks: Vec<(String, Vec<u8>)> = (0u8..10)
///     .map(|i| (format!("cid-{i}"), vec![i; 32]))
///     .collect();
/// let mut iter = BlockStreamIterator::new(blocks, StreamConfig::default());
///
/// while !iter.is_exhausted() {
///     if let Some(chunk) = iter.next_chunk() {
///         println!("chunk {}: {} blocks", chunk.chunk_index, chunk.len());
///     } else {
///         // backpressure — drain before continuing
///         iter.drain_one();
///     }
/// }
/// // drain any remaining buffered chunks
/// while let Some(_chunk) = iter.drain_one() {}
/// ```
#[derive(Debug)]
pub struct BlockStreamIterator {
    /// Source (CID, data) pairs.
    pub all_blocks: Vec<(String, Vec<u8>)>,
    /// Current read position in `all_blocks`.
    pub position: usize,
    /// Stream configuration.
    pub config: StreamConfig,
    /// Buffered but not yet consumed chunks.
    pub buffer: VecDeque<BlockChunk>,
    /// Current state of the iterator.
    pub state: BlockStreamState,
    /// Accumulated statistics.
    pub stats: StreamStats,
}

impl BlockStreamIterator {
    /// Construct a new iterator over `blocks` with the given `config`.
    pub fn new(blocks: Vec<(String, Vec<u8>)>, config: StreamConfig) -> Self {
        let state = if blocks.is_empty() {
            BlockStreamState::Exhausted
        } else {
            BlockStreamState::Ready
        };
        Self {
            all_blocks: blocks,
            position: 0,
            config,
            buffer: VecDeque::new(),
            state,
            stats: StreamStats::default(),
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Attempt to produce the next chunk.
    ///
    /// Returns `None` when:
    /// - The buffer is already at capacity (backpressure — call [`drain_one`](Self::drain_one)).
    /// - The iterator is exhausted.
    ///
    /// On success, advances `position`, pushes a [`BlockChunk`] onto the
    /// buffer, and returns a clone of that chunk.
    pub fn next_chunk(&mut self) -> Option<BlockChunk> {
        // Cannot produce if already exhausted.
        if matches!(self.state, BlockStreamState::Exhausted) {
            return None;
        }
        // Backpressure: buffer is full.
        if self.buffer.len() >= self.config.max_buffer_chunks {
            self.stats.backpressure_events += 1;
            self.state = BlockStreamState::Paused;
            return None;
        }
        // Nothing left to produce.
        if self.position >= self.all_blocks.len() {
            // Transition to Draining / Exhausted.
            if self.buffer.is_empty() {
                self.state = BlockStreamState::Exhausted;
            } else {
                self.state = BlockStreamState::Draining;
            }
            return None;
        }

        // Slice the next chunk out of all_blocks.
        let start = self.position;
        let end = (start + self.config.chunk_size).min(self.all_blocks.len());
        let is_final = end >= self.all_blocks.len();

        let mut cids = Vec::with_capacity(end - start);
        let mut data = Vec::with_capacity(end - start);

        for (cid, block_data) in &self.all_blocks[start..end] {
            cids.push(cid.clone());
            if self.config.include_data {
                data.push(block_data.clone());
            } else {
                data.push(Vec::new());
            }
        }

        self.position = end;

        let chunk_index = self.stats.chunks_produced;
        let chunk = BlockChunk {
            cids,
            data,
            chunk_index,
            is_final,
        };

        self.stats.chunks_produced += 1;
        self.buffer.push_back(chunk.clone());

        // Update state.
        if is_final {
            self.state = BlockStreamState::Draining;
        } else if self.buffer.len() >= self.config.max_buffer_chunks {
            self.state = BlockStreamState::Paused;
        } else {
            self.state = BlockStreamState::Ready;
        }

        Some(chunk)
    }

    /// Remove and return the oldest buffered chunk.
    ///
    /// Updates [`StreamStats::chunks_drained`] and [`StreamStats::bytes_streamed`].
    /// If the buffer becomes empty and all source blocks have been consumed,
    /// transitions to [`BlockStreamState::Exhausted`].
    pub fn drain_one(&mut self) -> Option<BlockChunk> {
        let chunk = self.buffer.pop_front()?;
        self.stats.chunks_drained += 1;
        self.stats.bytes_streamed += chunk.total_bytes() as u64;

        // After draining, transition out of Paused if we were blocked.
        let buffer_was_full = matches!(self.state, BlockStreamState::Paused);
        if buffer_was_full && self.buffer.len() < self.config.max_buffer_chunks {
            if self.position < self.all_blocks.len() {
                self.state = BlockStreamState::Ready;
            } else if self.buffer.is_empty() {
                self.state = BlockStreamState::Exhausted;
            } else {
                self.state = BlockStreamState::Draining;
            }
        } else if self.buffer.is_empty() && self.position >= self.all_blocks.len() {
            self.state = BlockStreamState::Exhausted;
        }

        Some(chunk)
    }

    /// Fill the buffer from source blocks up to `max_buffer_chunks`.
    ///
    /// Calls [`next_chunk`](Self::next_chunk) repeatedly until the buffer is
    /// full, backpressure fires, or the source is exhausted.
    pub fn fill_buffer(&mut self) {
        while self.buffer.len() < self.config.max_buffer_chunks
            && self.position < self.all_blocks.len()
            && !matches!(
                self.state,
                BlockStreamState::Exhausted | BlockStreamState::Error(_)
            )
        {
            if self.next_chunk().is_none() {
                break;
            }
        }
    }

    /// Returns `true` when all blocks have been produced **and** the buffer
    /// has been fully drained.
    #[inline]
    pub fn is_exhausted(&self) -> bool {
        matches!(self.state, BlockStreamState::Exhausted)
    }

    /// Number of source blocks not yet chunked.
    #[inline]
    pub fn remaining_blocks(&self) -> usize {
        self.all_blocks.len().saturating_sub(self.position)
    }

    /// Number of chunks currently sitting in the buffer.
    #[inline]
    pub fn buffered_chunks(&self) -> usize {
        self.buffer.len()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple block list of `n` entries.
    fn make_blocks(n: usize) -> Vec<(String, Vec<u8>)> {
        (0..n)
            .map(|i| (format!("cid-{i:04}"), vec![i as u8; 32]))
            .collect()
    }

    // ── 1. next_chunk produces correct chunk sizes ────────────────────────────

    #[test]
    fn test_chunk_size_exact_multiple() {
        let blocks = make_blocks(64);
        let config = StreamConfig {
            chunk_size: 16,
            max_buffer_chunks: 10,
            include_data: true,
        };
        let mut iter = BlockStreamIterator::new(blocks, config);

        let chunk = iter.next_chunk().expect("should produce chunk");
        assert_eq!(chunk.len(), 16);
        assert_eq!(chunk.cids.len(), chunk.data.len());
    }

    #[test]
    fn test_chunk_size_remainder() {
        // 70 blocks with chunk_size=32 → first two chunks of 32, last of 6.
        let blocks = make_blocks(70);
        let config = StreamConfig {
            chunk_size: 32,
            max_buffer_chunks: 10,
            include_data: true,
        };
        let mut iter = BlockStreamIterator::new(blocks, config);

        let c0 = iter.next_chunk().expect("chunk 0");
        assert_eq!(c0.len(), 32);
        assert!(!c0.is_final);

        let c1 = iter.next_chunk().expect("chunk 1");
        assert_eq!(c1.len(), 32);
        assert!(!c1.is_final);

        let c2 = iter.next_chunk().expect("chunk 2");
        assert_eq!(c2.len(), 6);
        assert!(c2.is_final);
    }

    // ── 2. Final chunk has is_final = true ───────────────────────────────────

    #[test]
    fn test_is_final_flag() {
        let blocks = make_blocks(10);
        let config = StreamConfig {
            chunk_size: 5,
            max_buffer_chunks: 10,
            include_data: true,
        };
        let mut iter = BlockStreamIterator::new(blocks, config);

        let c0 = iter.next_chunk().expect("chunk 0");
        assert!(!c0.is_final);

        let c1 = iter.next_chunk().expect("chunk 1");
        assert!(c1.is_final);
    }

    // ── 3. is_exhausted after all chunks produced and drained ────────────────

    #[test]
    fn test_is_exhausted_after_drain() {
        let blocks = make_blocks(5);
        let config = StreamConfig {
            chunk_size: 5,
            max_buffer_chunks: 4,
            include_data: true,
        };
        let mut iter = BlockStreamIterator::new(blocks, config);

        iter.next_chunk().expect("chunk");
        assert!(!iter.is_exhausted());

        iter.drain_one().expect("drain");
        assert!(iter.is_exhausted());
    }

    // ── 4. Backpressure: next_chunk returns None when buffer is full ─────────

    #[test]
    fn test_backpressure_returns_none() {
        // max_buffer_chunks = 2, chunk_size = 1 so we can fill fast
        let blocks = make_blocks(10);
        let config = StreamConfig {
            chunk_size: 1,
            max_buffer_chunks: 2,
            include_data: true,
        };
        let mut iter = BlockStreamIterator::new(blocks, config);

        // Fill buffer to capacity.
        iter.next_chunk().expect("chunk 0");
        iter.next_chunk().expect("chunk 1");
        assert_eq!(iter.buffered_chunks(), 2);

        // Now the buffer is full → backpressure.
        let result = iter.next_chunk();
        assert!(result.is_none());
        assert_eq!(iter.stats.backpressure_events, 1);
        assert!(matches!(iter.state, BlockStreamState::Paused));
    }

    // ── 5. drain_one releases backpressure ───────────────────────────────────

    #[test]
    fn test_drain_releases_backpressure() {
        let blocks = make_blocks(10);
        let config = StreamConfig {
            chunk_size: 1,
            max_buffer_chunks: 2,
            include_data: true,
        };
        let mut iter = BlockStreamIterator::new(blocks, config);

        iter.next_chunk();
        iter.next_chunk();
        // Trigger backpressure.
        assert!(iter.next_chunk().is_none());

        // Drain one → state should become Ready again.
        iter.drain_one().expect("should drain");
        assert!(matches!(iter.state, BlockStreamState::Ready));

        // Now we can produce again.
        let chunk = iter.next_chunk().expect("should produce after drain");
        assert_eq!(chunk.chunk_index, 2);
    }

    // ── 6. include_data = false produces empty data vecs ────────────────────

    #[test]
    fn test_include_data_false() {
        let blocks = make_blocks(8);
        let config = StreamConfig {
            chunk_size: 8,
            max_buffer_chunks: 4,
            include_data: false,
        };
        let mut iter = BlockStreamIterator::new(blocks, config);
        let chunk = iter.next_chunk().expect("chunk");

        assert_eq!(chunk.cids.len(), 8);
        for d in &chunk.data {
            assert!(d.is_empty());
        }
        assert_eq!(chunk.total_bytes(), 0);
    }

    // ── 7. remaining_blocks decreases as chunks are produced ─────────────────

    #[test]
    fn test_remaining_blocks_decreases() {
        let blocks = make_blocks(20);
        let config = StreamConfig {
            chunk_size: 10,
            max_buffer_chunks: 4,
            include_data: true,
        };
        let mut iter = BlockStreamIterator::new(blocks, config);

        assert_eq!(iter.remaining_blocks(), 20);
        iter.next_chunk();
        assert_eq!(iter.remaining_blocks(), 10);
        iter.next_chunk();
        assert_eq!(iter.remaining_blocks(), 0);
    }

    // ── 8. Empty input → immediate exhaustion ────────────────────────────────

    #[test]
    fn test_empty_input_exhausted() {
        let mut iter = BlockStreamIterator::new(vec![], StreamConfig::default());
        assert!(iter.is_exhausted());
        assert!(iter.next_chunk().is_none());
        assert!(iter.drain_one().is_none());
        assert_eq!(iter.remaining_blocks(), 0);
        assert_eq!(iter.buffered_chunks(), 0);
    }

    // ── 9. Stats accumulate correctly ────────────────────────────────────────

    #[test]
    fn test_stats_accumulate() {
        let blocks = make_blocks(6);
        let config = StreamConfig {
            chunk_size: 3,
            max_buffer_chunks: 4,
            include_data: true,
        };
        let mut iter = BlockStreamIterator::new(blocks, config);

        iter.next_chunk().expect("c0");
        iter.next_chunk().expect("c1");

        assert_eq!(iter.stats.chunks_produced, 2);
        assert_eq!(iter.stats.chunks_drained, 0);

        iter.drain_one().expect("drain c0");
        // Each block is 32 bytes, 3 blocks per chunk → 96 bytes.
        assert_eq!(iter.stats.bytes_streamed, 96);
        assert_eq!(iter.stats.chunks_drained, 1);

        iter.drain_one().expect("drain c1");
        assert_eq!(iter.stats.bytes_streamed, 192);
        assert_eq!(iter.stats.chunks_drained, 2);
    }

    // ── 10. chunk_index increments monotonically ─────────────────────────────

    #[test]
    fn test_chunk_index_monotonic() {
        let blocks = make_blocks(30);
        let config = StreamConfig {
            chunk_size: 10,
            max_buffer_chunks: 4,
            include_data: true,
        };
        let mut iter = BlockStreamIterator::new(blocks, config);

        for expected in 0u64..3 {
            let chunk = iter.next_chunk().expect("chunk");
            assert_eq!(chunk.chunk_index, expected);
        }
    }

    // ── 11. fill_buffer fills up to max_buffer_chunks ────────────────────────

    #[test]
    fn test_fill_buffer() {
        let blocks = make_blocks(100);
        let config = StreamConfig {
            chunk_size: 10,
            max_buffer_chunks: 4,
            include_data: true,
        };
        let mut iter = BlockStreamIterator::new(blocks, config);

        iter.fill_buffer();
        assert_eq!(iter.buffered_chunks(), 4);
        assert_eq!(iter.remaining_blocks(), 60);
    }

    // ── 12. CIDs in chunks match source order ────────────────────────────────

    #[test]
    fn test_cid_order_preserved() {
        let blocks = make_blocks(5);
        let config = StreamConfig {
            chunk_size: 5,
            max_buffer_chunks: 4,
            include_data: true,
        };
        let mut iter = BlockStreamIterator::new(blocks.clone(), config);
        let chunk = iter.next_chunk().expect("chunk");

        for (i, cid) in chunk.cids.iter().enumerate() {
            assert_eq!(*cid, blocks[i].0);
        }
    }

    // ── 13. Single-block input ────────────────────────────────────────────────

    #[test]
    fn test_single_block() {
        let blocks = vec![("cid-single".to_string(), vec![0xAB; 64])];
        let config = StreamConfig::default();
        let mut iter = BlockStreamIterator::new(blocks, config);

        let chunk = iter.next_chunk().expect("chunk");
        assert!(chunk.is_final);
        assert_eq!(chunk.len(), 1);
        assert_eq!(chunk.total_bytes(), 64);

        // Producing another chunk yields nothing.
        assert!(iter.next_chunk().is_none());

        // Not exhausted until buffer drained.
        assert!(!iter.is_exhausted());
        iter.drain_one().expect("drain");
        assert!(iter.is_exhausted());
    }

    // ── 14. Full drain loop exhausts the iterator ────────────────────────────

    #[test]
    fn test_full_drain_loop() {
        let n = 50usize;
        let blocks = make_blocks(n);
        let config = StreamConfig {
            chunk_size: 7,
            max_buffer_chunks: 3,
            include_data: true,
        };
        let mut iter = BlockStreamIterator::new(blocks, config);

        let mut total_blocks_seen = 0usize;
        while !iter.is_exhausted() {
            if let Some(chunk) = iter.next_chunk() {
                total_blocks_seen += chunk.len();
            } else {
                // Either backpressure or source done — drain one.
                if let Some(chunk) = iter.drain_one() {
                    let _ = chunk;
                } else {
                    // Buffer empty and nothing to produce → force exhausted.
                    break;
                }
            }
        }
        // Drain anything left in buffer.
        while iter.drain_one().is_some() {}

        assert!(iter.is_exhausted());
        assert_eq!(iter.stats.chunks_produced, iter.stats.chunks_drained);
        // total blocks seen from next_chunk == n
        assert_eq!(total_blocks_seen, n);
    }

    // ── 15. backpressure_events counter increments correctly ─────────────────

    #[test]
    fn test_backpressure_events_counter() {
        let blocks = make_blocks(20);
        let config = StreamConfig {
            chunk_size: 1,
            max_buffer_chunks: 2,
            include_data: false,
        };
        let mut iter = BlockStreamIterator::new(blocks, config);

        iter.next_chunk();
        iter.next_chunk();
        // Trigger 3 backpressure events without draining.
        iter.next_chunk(); // backpressure #1
        iter.next_chunk(); // backpressure #2
        iter.next_chunk(); // backpressure #3

        assert_eq!(iter.stats.backpressure_events, 3);
    }
}
