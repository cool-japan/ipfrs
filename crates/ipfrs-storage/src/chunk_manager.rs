//! Storage chunk manager for splitting large objects into fixed-size chunks.
//!
//! Manages chunk metadata, state transitions, and reassembly ordering
//! for streaming reads of large data objects.

use std::collections::HashMap;

/// State of an individual chunk within a chunked object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkState {
    /// Chunk has not yet been written.
    Pending,
    /// Chunk was successfully stored.
    Written,
    /// Chunk checksum has been confirmed.
    Verified,
    /// Chunk write failed.
    Failed,
}

/// Metadata for a single chunk within a larger object.
#[derive(Clone, Debug)]
pub struct Chunk {
    /// Unique identifier for this chunk.
    pub chunk_id: u64,
    /// Parent object this chunk belongs to.
    pub object_id: u64,
    /// 0-based position within the object.
    pub chunk_index: u32,
    /// Byte offset within the original data.
    pub offset: u64,
    /// Size of this chunk in bytes.
    pub size_bytes: u64,
    /// Current state of the chunk.
    pub state: ChunkState,
    /// FNV-1a checksum of (object_id XOR chunk_index as u64).
    pub checksum: u64,
}

/// A large object that has been split into fixed-size chunks.
#[derive(Clone, Debug)]
pub struct ChunkedObject {
    /// Unique identifier for this object.
    pub object_id: u64,
    /// Total size of the original data in bytes.
    pub total_size_bytes: u64,
    /// Configured chunk size in bytes.
    pub chunk_size_bytes: u64,
    /// Ordered list of chunks by chunk_index.
    pub chunks: Vec<Chunk>,
}

impl ChunkedObject {
    /// Returns the total number of chunks in this object.
    pub fn total_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Returns the count of chunks in Written or Verified state.
    pub fn written_chunks(&self) -> usize {
        self.chunks
            .iter()
            .filter(|c| matches!(c.state, ChunkState::Written | ChunkState::Verified))
            .count()
    }

    /// Returns true when all chunks are in Written or Verified state.
    pub fn is_complete(&self) -> bool {
        !self.chunks.is_empty()
            && self
                .chunks
                .iter()
                .all(|c| matches!(c.state, ChunkState::Written | ChunkState::Verified))
    }

    /// Returns written_chunks / total_chunks * 100.0; 0.0 if empty.
    pub fn completion_pct(&self) -> f64 {
        let total = self.total_chunks();
        if total == 0 {
            return 0.0;
        }
        (self.written_chunks() as f64 / total as f64) * 100.0
    }
}

/// Aggregate statistics across all managed objects.
#[derive(Clone, Debug, Default)]
pub struct ChunkManagerStats {
    /// Total number of objects being tracked.
    pub total_objects: usize,
    /// Total number of chunks across all objects.
    pub total_chunks: usize,
    /// Number of chunks in Written or Verified state.
    pub written_chunks: usize,
    /// Number of chunks in Failed state.
    pub failed_chunks: usize,
    /// Total bytes across all objects.
    pub total_bytes: u64,
}

/// Computes FNV-1a hash of the 8-byte little-endian representation of `value`.
pub fn fnv1a_u64(value: u64) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;

    let bytes = value.to_le_bytes();
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in &bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Manages splitting large data objects into fixed-size chunks and tracking
/// their metadata for streaming reads and writes.
pub struct StorageChunkManager {
    /// Map from object_id to its chunked representation.
    pub objects: HashMap<u64, ChunkedObject>,
    /// Counter for assigning monotonically increasing object IDs.
    pub next_object_id: u64,
    /// Default chunk size in bytes (1 MB).
    pub chunk_size_bytes: u64,
}

impl StorageChunkManager {
    /// Creates a new `StorageChunkManager` with the given chunk size.
    pub fn new(chunk_size_bytes: u64) -> Self {
        Self {
            objects: HashMap::new(),
            next_object_id: 1,
            chunk_size_bytes,
        }
    }

    /// Splits a data object of `total_size_bytes` into chunks and registers it.
    ///
    /// Returns the assigned object_id.
    pub fn create_object(&mut self, total_size_bytes: u64) -> u64 {
        let object_id = self.next_object_id;
        self.next_object_id += 1;

        let chunk_size = self.chunk_size_bytes;
        let num_chunks = total_size_bytes.div_ceil(chunk_size).max(1);

        let mut chunks = Vec::with_capacity(num_chunks as usize);
        let mut remaining = total_size_bytes;

        for chunk_index in 0..num_chunks {
            let offset = chunk_index * chunk_size;
            let size = remaining.min(chunk_size);
            remaining = remaining.saturating_sub(size);

            let checksum_input = object_id ^ chunk_index;
            let checksum = fnv1a_u64(checksum_input);

            // chunk_id encodes object and index for uniqueness
            let chunk_id = object_id.wrapping_shl(32) | (chunk_index & 0xFFFF_FFFF);

            chunks.push(Chunk {
                chunk_id,
                object_id,
                chunk_index: chunk_index as u32,
                offset,
                size_bytes: size,
                state: ChunkState::Pending,
                checksum,
            });
        }

        self.objects.insert(
            object_id,
            ChunkedObject {
                object_id,
                total_size_bytes,
                chunk_size_bytes: chunk_size,
                chunks,
            },
        );

        object_id
    }

    /// Sets the specified chunk to `Written`. Returns false if not found.
    pub fn mark_written(&mut self, object_id: u64, chunk_index: u32) -> bool {
        self.set_chunk_state(object_id, chunk_index, ChunkState::Written)
    }

    /// Sets the specified chunk to `Verified`. Returns false if not found.
    pub fn mark_verified(&mut self, object_id: u64, chunk_index: u32) -> bool {
        self.set_chunk_state(object_id, chunk_index, ChunkState::Verified)
    }

    /// Sets the specified chunk to `Failed`. Returns false if not found.
    pub fn mark_failed(&mut self, object_id: u64, chunk_index: u32) -> bool {
        self.set_chunk_state(object_id, chunk_index, ChunkState::Failed)
    }

    /// Returns a reference to the `ChunkedObject` if it exists.
    pub fn get_object(&self, object_id: u64) -> Option<&ChunkedObject> {
        self.objects.get(&object_id)
    }

    /// Returns all `Pending` chunks for the given object, sorted by chunk_index.
    pub fn pending_chunks(&self, object_id: u64) -> Vec<&Chunk> {
        match self.objects.get(&object_id) {
            None => Vec::new(),
            Some(obj) => {
                let mut pending: Vec<&Chunk> = obj
                    .chunks
                    .iter()
                    .filter(|c| c.state == ChunkState::Pending)
                    .collect();
                pending.sort_by_key(|c| c.chunk_index);
                pending
            }
        }
    }

    /// Removes the object from the manager. Returns true if it existed.
    pub fn delete_object(&mut self, object_id: u64) -> bool {
        self.objects.remove(&object_id).is_some()
    }

    /// Returns aggregate statistics over all tracked objects.
    pub fn stats(&self) -> ChunkManagerStats {
        let mut stats = ChunkManagerStats {
            total_objects: self.objects.len(),
            ..Default::default()
        };

        for obj in self.objects.values() {
            stats.total_chunks += obj.total_chunks();
            stats.written_chunks += obj
                .chunks
                .iter()
                .filter(|c| matches!(c.state, ChunkState::Written | ChunkState::Verified))
                .count();
            stats.failed_chunks += obj
                .chunks
                .iter()
                .filter(|c| c.state == ChunkState::Failed)
                .count();
            stats.total_bytes += obj.total_size_bytes;
        }

        stats
    }

    // --- Internal helpers ---

    fn set_chunk_state(&mut self, object_id: u64, chunk_index: u32, state: ChunkState) -> bool {
        match self.objects.get_mut(&object_id) {
            None => false,
            Some(obj) => match obj.chunks.iter_mut().find(|c| c.chunk_index == chunk_index) {
                None => false,
                Some(chunk) => {
                    chunk.state = state;
                    true
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Construction ---

    #[test]
    fn test_new_with_chunk_size() {
        let mgr = StorageChunkManager::new(512);
        assert_eq!(mgr.chunk_size_bytes, 512);
        assert!(mgr.objects.is_empty());
    }

    #[test]
    fn test_new_default_chunk_size_constant() {
        let mgr = StorageChunkManager::new(1_048_576);
        assert_eq!(mgr.chunk_size_bytes, 1_048_576);
    }

    // --- create_object: monotonic IDs ---

    #[test]
    fn test_create_object_returns_monotonic_ids() {
        let mut mgr = StorageChunkManager::new(1024);
        let id1 = mgr.create_object(1024);
        let id2 = mgr.create_object(1024);
        let id3 = mgr.create_object(1024);
        assert!(id1 < id2);
        assert!(id2 < id3);
    }

    // --- create_object: chunk count exact fit ---

    #[test]
    fn test_create_object_exact_chunk_count() {
        let chunk_size = 1024_u64;
        let mut mgr = StorageChunkManager::new(chunk_size);
        let id = mgr.create_object(chunk_size * 4);
        let obj = mgr.get_object(id).expect("object should exist");
        assert_eq!(obj.total_chunks(), 4);
    }

    // --- create_object: chunk count partial last chunk ---

    #[test]
    fn test_create_object_partial_last_chunk_count() {
        let chunk_size = 1024_u64;
        let mut mgr = StorageChunkManager::new(chunk_size);
        let id = mgr.create_object(chunk_size * 3 + 1);
        let obj = mgr.get_object(id).expect("object should exist");
        assert_eq!(obj.total_chunks(), 4);
    }

    // --- last chunk smaller size ---

    #[test]
    fn test_last_chunk_has_correct_smaller_size() {
        let chunk_size = 1024_u64;
        let remainder = 300_u64;
        let mut mgr = StorageChunkManager::new(chunk_size);
        let id = mgr.create_object(chunk_size * 2 + remainder);
        let obj = mgr.get_object(id).expect("object should exist");
        assert_eq!(obj.total_chunks(), 3);
        let last = obj.chunks.last().expect("must have last chunk");
        assert_eq!(last.size_bytes, remainder);
    }

    // --- chunk offsets sequential ---

    #[test]
    fn test_chunk_offsets_are_sequential() {
        let chunk_size = 512_u64;
        let mut mgr = StorageChunkManager::new(chunk_size);
        let id = mgr.create_object(chunk_size * 5);
        let obj = mgr.get_object(id).expect("object should exist");
        for (i, chunk) in obj.chunks.iter().enumerate() {
            assert_eq!(chunk.offset, i as u64 * chunk_size);
        }
    }

    // --- checksum computed ---

    #[test]
    fn test_chunk_checksums_computed() {
        let mut mgr = StorageChunkManager::new(256);
        let id = mgr.create_object(256 * 3);
        let obj = mgr.get_object(id).expect("object should exist");
        for chunk in &obj.chunks {
            let expected = fnv1a_u64(chunk.object_id ^ (chunk.chunk_index as u64));
            assert_eq!(chunk.checksum, expected);
        }
    }

    // --- mark_written ---

    #[test]
    fn test_mark_written_sets_written_state() {
        let mut mgr = StorageChunkManager::new(1024);
        let id = mgr.create_object(2048);
        assert!(mgr.mark_written(id, 0));
        let obj = mgr.get_object(id).expect("object should exist");
        assert_eq!(obj.chunks[0].state, ChunkState::Written);
    }

    #[test]
    fn test_mark_written_returns_false_for_unknown_object() {
        let mut mgr = StorageChunkManager::new(1024);
        assert!(!mgr.mark_written(999, 0));
    }

    #[test]
    fn test_mark_written_returns_false_for_unknown_chunk_index() {
        let mut mgr = StorageChunkManager::new(1024);
        let id = mgr.create_object(1024);
        assert!(!mgr.mark_written(id, 99));
    }

    // --- mark_verified ---

    #[test]
    fn test_mark_verified_sets_verified_state() {
        let mut mgr = StorageChunkManager::new(1024);
        let id = mgr.create_object(2048);
        assert!(mgr.mark_verified(id, 1));
        let obj = mgr.get_object(id).expect("object should exist");
        assert_eq!(obj.chunks[1].state, ChunkState::Verified);
    }

    // --- mark_failed ---

    #[test]
    fn test_mark_failed_sets_failed_state() {
        let mut mgr = StorageChunkManager::new(1024);
        let id = mgr.create_object(2048);
        assert!(mgr.mark_failed(id, 0));
        let obj = mgr.get_object(id).expect("object should exist");
        assert_eq!(obj.chunks[0].state, ChunkState::Failed);
    }

    // --- is_complete ---

    #[test]
    fn test_is_complete_true_when_all_written() {
        let mut mgr = StorageChunkManager::new(1024);
        let id = mgr.create_object(2048);
        mgr.mark_written(id, 0);
        mgr.mark_written(id, 1);
        let obj = mgr.get_object(id).expect("object should exist");
        assert!(obj.is_complete());
    }

    #[test]
    fn test_is_complete_true_when_all_verified() {
        let mut mgr = StorageChunkManager::new(1024);
        let id = mgr.create_object(2048);
        mgr.mark_verified(id, 0);
        mgr.mark_verified(id, 1);
        let obj = mgr.get_object(id).expect("object should exist");
        assert!(obj.is_complete());
    }

    #[test]
    fn test_is_complete_true_mixed_written_verified() {
        let mut mgr = StorageChunkManager::new(1024);
        let id = mgr.create_object(2048);
        mgr.mark_written(id, 0);
        mgr.mark_verified(id, 1);
        let obj = mgr.get_object(id).expect("object should exist");
        assert!(obj.is_complete());
    }

    #[test]
    fn test_is_complete_false_when_pending_remain() {
        let mut mgr = StorageChunkManager::new(1024);
        let id = mgr.create_object(2048);
        mgr.mark_written(id, 0);
        // chunk 1 still Pending
        let obj = mgr.get_object(id).expect("object should exist");
        assert!(!obj.is_complete());
    }

    // --- completion_pct ---

    #[test]
    fn test_completion_pct_correct() {
        let mut mgr = StorageChunkManager::new(1024);
        let id = mgr.create_object(4096); // 4 chunks
        mgr.mark_written(id, 0);
        mgr.mark_verified(id, 1);
        let obj = mgr.get_object(id).expect("object should exist");
        let pct = obj.completion_pct();
        assert!((pct - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_completion_pct_zero_when_empty_chunks() {
        let obj = ChunkedObject {
            object_id: 1,
            total_size_bytes: 0,
            chunk_size_bytes: 1024,
            chunks: Vec::new(),
        };
        assert_eq!(obj.completion_pct(), 0.0);
    }

    // --- written_chunks counts Written + Verified ---

    #[test]
    fn test_written_chunks_counts_written_and_verified() {
        let mut mgr = StorageChunkManager::new(512);
        let id = mgr.create_object(512 * 4);
        mgr.mark_written(id, 0);
        mgr.mark_verified(id, 2);
        mgr.mark_failed(id, 3);
        let obj = mgr.get_object(id).expect("object should exist");
        assert_eq!(obj.written_chunks(), 2);
    }

    // --- pending_chunks ---

    #[test]
    fn test_pending_chunks_returns_only_pending_sorted() {
        let mut mgr = StorageChunkManager::new(512);
        let id = mgr.create_object(512 * 4);
        mgr.mark_written(id, 1);
        mgr.mark_verified(id, 3);
        let pending = mgr.pending_chunks(id);
        assert_eq!(pending.len(), 2);
        assert_eq!(pending[0].chunk_index, 0);
        assert_eq!(pending[1].chunk_index, 2);
    }

    #[test]
    fn test_pending_chunks_empty_for_unknown_object() {
        let mgr = StorageChunkManager::new(512);
        let pending = mgr.pending_chunks(999);
        assert!(pending.is_empty());
    }

    // --- get_object ---

    #[test]
    fn test_get_object_returns_some_for_existing() {
        let mut mgr = StorageChunkManager::new(1024);
        let id = mgr.create_object(1024);
        assert!(mgr.get_object(id).is_some());
    }

    #[test]
    fn test_get_object_returns_none_for_missing() {
        let mgr = StorageChunkManager::new(1024);
        assert!(mgr.get_object(42).is_none());
    }

    // --- delete_object ---

    #[test]
    fn test_delete_object_returns_true_for_existing() {
        let mut mgr = StorageChunkManager::new(1024);
        let id = mgr.create_object(1024);
        assert!(mgr.delete_object(id));
        assert!(mgr.get_object(id).is_none());
    }

    #[test]
    fn test_delete_object_returns_false_for_missing() {
        let mut mgr = StorageChunkManager::new(1024);
        assert!(!mgr.delete_object(999));
    }

    // --- stats ---

    #[test]
    fn test_stats_total_objects_and_chunks() {
        let mut mgr = StorageChunkManager::new(1024);
        mgr.create_object(2048); // 2 chunks
        mgr.create_object(4096); // 4 chunks
        let s = mgr.stats();
        assert_eq!(s.total_objects, 2);
        assert_eq!(s.total_chunks, 6);
    }

    #[test]
    fn test_stats_written_and_failed_chunks() {
        let mut mgr = StorageChunkManager::new(1024);
        let id = mgr.create_object(4096); // 4 chunks
        mgr.mark_written(id, 0);
        mgr.mark_verified(id, 1);
        mgr.mark_failed(id, 2);
        let s = mgr.stats();
        assert_eq!(s.written_chunks, 2);
        assert_eq!(s.failed_chunks, 1);
    }

    #[test]
    fn test_stats_total_bytes() {
        let mut mgr = StorageChunkManager::new(1024);
        mgr.create_object(2000);
        mgr.create_object(3000);
        let s = mgr.stats();
        assert_eq!(s.total_bytes, 5000);
    }

    // --- fnv1a helper ---

    #[test]
    fn test_fnv1a_u64_deterministic() {
        let h1 = fnv1a_u64(42);
        let h2 = fnv1a_u64(42);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fnv1a_u64_different_inputs_produce_different_hashes() {
        let h1 = fnv1a_u64(0);
        let h2 = fnv1a_u64(1);
        assert_ne!(h1, h2);
    }
}
