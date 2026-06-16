//! Content-aware block deduplication with variable-length chunking.
//!
//! Implements Content-Defined Chunking (CDC) using a Rabin-fingerprint-style
//! rolling hash based on FNV-1a. Chunk boundaries are detected when the rolling
//! hash of a sliding window satisfies `hash & mask == 0`, producing statistically
//! independent chunk boundaries that average `2^target_bits` bytes apart.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐     chunk_data()      ┌─────────────────────────────┐
//! │  Raw Object  │ ──────────────────>   │  Vec<(ChunkHash, Vec<u8>)>  │
//! └──────────────┘                       └─────────────────────────────┘
//!        │                                          │
//!        │  store_object()                          │ dedup lookup
//!        ▼                                          ▼
//! ┌──────────────────┐              ┌──────────────────────────────────┐
//! │  ObjectManifest  │              │  chunk_store: HashMap<ChunkHash, │
//! │  Vec<ChunkRef>   │              │    (Chunk, Vec<u8>)>             │
//! └──────────────────┘              └──────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust
//! use ipfrs_storage::block_deduplicator::{BlockDeduplicator, ChunkingConfig};
//!
//! let config = ChunkingConfig::default();
//! let mut dedup = BlockDeduplicator::new(config);
//!
//! let data = b"Hello, World! This is some test data for deduplication.".to_vec();
//! let manifest = dedup.store_object("obj1".to_string(), data.clone()).unwrap();
//! let retrieved = dedup.retrieve_object("obj1").unwrap();
//! assert_eq!(data, retrieved);
//! ```

use std::collections::HashMap;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

// ─── FNV-1a helpers ──────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash — fast, non-cryptographic, excellent for CDC boundaries.
///
/// Uses the standard FNV prime and offset basis:
/// - Offset basis: 14695981039346656037
/// - Prime: 1099511628211
#[inline]
pub fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

/// Rolling hash for CDC: compute hash over a sliding window of bytes.
/// Uses FNV-1a as the window function — cheap to compute, good distribution.
#[inline]
fn rolling_hash(window: &[u8]) -> u64 {
    fnv1a_64(window)
}

/// Current wall-clock time in seconds since UNIX epoch.
/// Falls back to 0 if the system clock is unavailable.
fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ─── ChunkHash ───────────────────────────────────────────────────────────────

/// A compact 8-byte content hash derived from FNV-1a over chunk data.
///
/// Stored as little-endian bytes. Displayed as 16 lowercase hex characters.
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub struct ChunkHash([u8; 8]);

impl ChunkHash {
    /// Create a `ChunkHash` directly from raw bytes.
    #[inline]
    pub fn from_bytes(bytes: [u8; 8]) -> Self {
        Self(bytes)
    }

    /// Access the raw bytes.
    #[inline]
    pub fn as_bytes(&self) -> &[u8; 8] {
        &self.0
    }

    /// Convert to `u64` (little-endian interpretation).
    #[inline]
    pub fn to_u64(self) -> u64 {
        u64::from_le_bytes(self.0)
    }
}

impl From<u64> for ChunkHash {
    #[inline]
    fn from(v: u64) -> Self {
        Self(v.to_le_bytes())
    }
}

impl fmt::Display for ChunkHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in &self.0 {
            write!(f, "{:02x}", b)?;
        }
        Ok(())
    }
}

// ─── Chunk ───────────────────────────────────────────────────────────────────

/// Metadata record for a single stored chunk.
///
/// Note: in `lib.rs` this is re-exported as `BddChunk` to avoid collision
/// with `chunk_manager::Chunk`.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Content-defined hash of this chunk's raw bytes.
    pub hash: ChunkHash,
    /// Byte offset of this chunk's *first occurrence* in the original object.
    pub offset: u64,
    /// Length in bytes of the raw (uncompressed) chunk data.
    pub length: usize,
    /// Number of live references to this chunk across all stored objects.
    pub ref_count: u32,
    /// Whether the stored bytes in the chunk store are compressed.
    pub compressed: bool,
}

// ─── DeduplicationStats ──────────────────────────────────────────────────────

/// Aggregate statistics for the [`BlockDeduplicator`].
///
/// Note: re-exported as `BddDeduplicationStats` in `lib.rs`.
#[derive(Debug, Clone)]
pub struct DeduplicationStats {
    /// Total logical chunk references across all live objects (including duplicates).
    pub total_chunks: u64,
    /// Number of physically distinct chunks stored on disk.
    pub unique_chunks: u64,
    /// Logical references that point to already-stored chunks (`total - unique`).
    pub duplicate_chunks: u64,
    /// Total raw bytes ingested via `store_object` calls.
    pub bytes_before: u64,
    /// Physical bytes currently occupying the chunk store.
    pub bytes_after: u64,
    /// Fraction of bytes saved by deduplication: `1 - (bytes_after / bytes_before)`.
    /// Range `[0.0, 1.0]`; higher is better.
    pub dedup_ratio: f64,
    /// Ratio `bytes_before / bytes_after`. Values > 1.0 mean space was saved.
    pub compression_ratio: f64,
}

// ─── ChunkingConfig ──────────────────────────────────────────────────────────

/// Configuration for the Content-Defined Chunking (CDC) algorithm.
///
/// Note: re-exported as `BddChunkingConfig` in `lib.rs` to avoid collision
/// with `dedup::ChunkingConfig`.
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    /// Minimum chunk size in bytes. Boundaries are not checked below this threshold.
    pub min_chunk_size: usize,
    /// Maximum chunk size in bytes. A boundary is forced if the current chunk
    /// reaches this length without a natural hash-based split.
    pub max_chunk_size: usize,
    /// Controls the average target chunk size: avg ≈ `2^target_bits` bytes.
    /// Default 13 → ~8 KiB average.
    pub target_bits: u8,
    /// Sliding window width used for rolling hash computation.
    pub window_size: usize,
    /// Whether to attempt compression on stored chunk data (currently marks metadata only).
    pub enable_compression: bool,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            min_chunk_size: 2048,
            max_chunk_size: 65536,
            target_bits: 13,
            window_size: 48,
            enable_compression: false,
        }
    }
}

impl ChunkingConfig {
    /// Returns the rolling-hash boundary mask: `(1 << target_bits) - 1`.
    #[inline]
    pub fn boundary_mask(&self) -> u64 {
        (1u64 << self.target_bits).wrapping_sub(1)
    }
}

// ─── ChunkRef ────────────────────────────────────────────────────────────────

/// Reference to a chunk within an object manifest.
///
/// Describes one logical slice of the original object, pointing to the
/// physical chunk in the chunk store by hash.
#[derive(Debug, Clone)]
pub struct ChunkRef {
    /// Hash of the referenced chunk.
    pub hash: ChunkHash,
    /// Byte offset of this chunk's data within the reconstructed object.
    pub offset_in_object: u64,
    /// Length of this chunk's data in bytes.
    pub chunk_length: usize,
}

// ─── ObjectManifest ──────────────────────────────────────────────────────────

/// Ordered list of chunk references that fully describe a stored object.
///
/// Reassembling all chunks in order reconstructs the original byte sequence.
#[derive(Debug, Clone)]
pub struct ObjectManifest {
    /// Unique identifier for this object.
    pub object_id: String,
    /// Total uncompressed size of the object in bytes.
    pub total_size: u64,
    /// Ordered sequence of chunk references.
    pub chunks: Vec<ChunkRef>,
    /// UNIX timestamp (seconds) when this manifest was created.
    pub created_at: u64,
}

// ─── DeduplicatorError ───────────────────────────────────────────────────────

/// Errors that may arise from [`BlockDeduplicator`] operations.
#[derive(Debug, thiserror::Error)]
pub enum DeduplicatorError {
    /// The requested object ID does not exist in the manifest store.
    #[error("object not found: {0}")]
    ObjectNotFound(String),

    /// The chunk identified by the given hash is not present in the store.
    #[error("chunk not found: {0}")]
    ChunkNotFound(ChunkHash),

    /// A compression or decompression operation failed.
    #[error("compression failed: {0}")]
    CompressionFailed(String),

    /// The manifest data is structurally invalid or inconsistent.
    #[error("invalid manifest: {0}")]
    InvalidManifest(String),

    /// The chunk store has reached its capacity limit.
    #[error("storage full")]
    StorageFull,
}

// ─── BlockDeduplicator ───────────────────────────────────────────────────────

/// Content-aware block deduplicator using variable-length CDC chunking.
///
/// Stores objects by splitting them into content-defined chunks, using the
/// chunk hash as the storage key. Identical chunks are stored only once and
/// reference-counted. When all objects referencing a chunk are deleted, the
/// chunk data is freed.
///
/// # Algorithm
///
/// Content-Defined Chunking (CDC) uses a rolling hash over a sliding window:
/// - For each byte position after `min_chunk_size` into the current chunk,
///   compute `rolling_hash(data[pos-W..pos])` where `W = window_size`.
/// - If `hash & mask == 0` (where `mask = 2^target_bits - 1`), emit a boundary.
/// - Also emit a boundary when `chunk_length >= max_chunk_size`.
/// - The final remaining bytes are always emitted as the last chunk.
///
/// # Thread safety
///
/// `BlockDeduplicator` is not `Sync`. Use external locking (e.g. `Mutex`) when
/// sharing across threads.
pub struct BlockDeduplicator {
    config: ChunkingConfig,
    /// Primary storage: maps chunk hash → (metadata, raw bytes).
    chunk_store: HashMap<ChunkHash, (Chunk, Vec<u8>)>,
    /// Object manifests: maps object ID → ordered list of chunk refs.
    manifests: HashMap<String, ObjectManifest>,
    /// Cumulative raw bytes passed to `store_object` (denominator for dedup ratio).
    total_bytes_stored: u64,
    /// Bytes that were deduplicated (i.e., stored in already-present chunks).
    total_bytes_deduplicated: u64,
}

impl BlockDeduplicator {
    /// Create a new `BlockDeduplicator` with the given configuration.
    pub fn new(config: ChunkingConfig) -> Self {
        Self {
            config,
            chunk_store: HashMap::new(),
            manifests: HashMap::new(),
            total_bytes_stored: 0,
            total_bytes_deduplicated: 0,
        }
    }

    /// Create a `BlockDeduplicator` with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ChunkingConfig::default())
    }

    /// Split `data` into content-defined chunks using the rolling-hash CDC algorithm.
    ///
    /// Returns an ordered list of `(hash, chunk_bytes)` pairs. The concatenation
    /// of all `chunk_bytes` is guaranteed to equal `data` exactly.
    ///
    /// # Edge cases
    /// - Empty input → returns an empty `Vec`.
    /// - Input shorter than `min_chunk_size` → returns a single chunk.
    /// - Input shorter than `window_size` → rolling hash uses the full available prefix.
    pub fn chunk_data(&self, data: &[u8]) -> Vec<(ChunkHash, Vec<u8>)> {
        if data.is_empty() {
            return Vec::new();
        }

        let min_sz = self.config.min_chunk_size;
        let max_sz = self.config.max_chunk_size;
        let win = self.config.window_size;
        let mask = self.config.boundary_mask();

        let mut result = Vec::new();
        let mut chunk_start = 0usize;

        let mut pos = 0usize;
        while pos < data.len() {
            let chunk_len = pos - chunk_start;

            // We need at least min_chunk_size bytes before we check for a boundary.
            if chunk_len >= min_sz {
                // Determine the window: use min(win, pos) bytes ending at `pos`.
                let win_start = pos.saturating_sub(win);
                let window = &data[win_start..pos];
                let h = rolling_hash(window);

                let is_hash_boundary = (h & mask) == 0;
                let is_max_boundary = chunk_len >= max_sz;

                if is_hash_boundary || is_max_boundary {
                    // Emit chunk [chunk_start..pos]
                    let chunk_bytes = data[chunk_start..pos].to_vec();
                    let hash = ChunkHash::from(fnv1a_64(&chunk_bytes));
                    result.push((hash, chunk_bytes));
                    chunk_start = pos;
                }
            }

            pos += 1;
        }

        // Emit the final (possibly only) chunk — always non-empty since data is non-empty.
        if chunk_start < data.len() {
            let chunk_bytes = data[chunk_start..].to_vec();
            let hash = ChunkHash::from(fnv1a_64(&chunk_bytes));
            result.push((hash, chunk_bytes));
        }

        result
    }

    /// Store an object, deduplicating its content-defined chunks.
    ///
    /// If an identical chunk (same hash) already exists, its `ref_count` is
    /// incremented and the duplicate bytes are not stored again. A manifest is
    /// built and stored internally.
    ///
    /// # Errors
    ///
    /// Currently infallible, but returns `Result` for future extensibility
    /// (e.g., when storage-full limits are enforced).
    pub fn store_object(
        &mut self,
        object_id: String,
        data: Vec<u8>,
    ) -> Result<ObjectManifest, DeduplicatorError> {
        let total_size = data.len() as u64;
        self.total_bytes_stored += total_size;

        let chunks = self.chunk_data(&data);
        let mut chunk_refs = Vec::with_capacity(chunks.len());
        let mut offset_in_object: u64 = 0;
        let mut chunk_offset_in_store: u64 = 0;

        for (hash, chunk_bytes) in chunks {
            let chunk_len = chunk_bytes.len();

            if let Some((existing, _)) = self.chunk_store.get_mut(&hash) {
                // Duplicate: increment reference count.
                existing.ref_count = existing.ref_count.saturating_add(1);
                self.total_bytes_deduplicated += chunk_len as u64;
            } else {
                // New unique chunk: store it.
                let chunk = Chunk {
                    hash,
                    offset: chunk_offset_in_store,
                    length: chunk_len,
                    ref_count: 1,
                    compressed: false,
                };
                self.chunk_store.insert(hash, (chunk, chunk_bytes));
                chunk_offset_in_store += chunk_len as u64;
            }

            chunk_refs.push(ChunkRef {
                hash,
                offset_in_object,
                chunk_length: chunk_len,
            });

            offset_in_object += chunk_len as u64;
        }

        let manifest = ObjectManifest {
            object_id: object_id.clone(),
            total_size,
            chunks: chunk_refs,
            created_at: unix_timestamp(),
        };

        self.manifests.insert(object_id, manifest.clone());
        Ok(manifest)
    }

    /// Retrieve the full byte content of a stored object by reconstructing
    /// it from its chunk manifest.
    ///
    /// # Errors
    ///
    /// - [`DeduplicatorError::ObjectNotFound`] if `object_id` has no manifest.
    /// - [`DeduplicatorError::ChunkNotFound`] if a chunk referenced by the manifest
    ///   is missing from the store (indicates data corruption).
    /// - [`DeduplicatorError::InvalidManifest`] if the reconstructed size does not
    ///   match the manifest's declared `total_size`.
    pub fn retrieve_object(&self, object_id: &str) -> Result<Vec<u8>, DeduplicatorError> {
        let manifest = self
            .manifests
            .get(object_id)
            .ok_or_else(|| DeduplicatorError::ObjectNotFound(object_id.to_string()))?;

        let mut result = Vec::with_capacity(manifest.total_size as usize);

        for chunk_ref in &manifest.chunks {
            let (_, data) = self
                .chunk_store
                .get(&chunk_ref.hash)
                .ok_or(DeduplicatorError::ChunkNotFound(chunk_ref.hash))?;
            result.extend_from_slice(data);
        }

        if result.len() as u64 != manifest.total_size {
            return Err(DeduplicatorError::InvalidManifest(format!(
                "object '{}': expected {} bytes, reconstructed {} bytes",
                object_id,
                manifest.total_size,
                result.len()
            )));
        }

        Ok(result)
    }

    /// Delete a stored object, decrementing ref counts of its chunks.
    ///
    /// Any chunk whose `ref_count` reaches zero is removed from the store.
    /// Returns the hashes of all physically removed chunks.
    ///
    /// # Errors
    ///
    /// - [`DeduplicatorError::ObjectNotFound`] if `object_id` has no manifest.
    pub fn delete_object(&mut self, object_id: &str) -> Result<Vec<ChunkHash>, DeduplicatorError> {
        let manifest = self
            .manifests
            .remove(object_id)
            .ok_or_else(|| DeduplicatorError::ObjectNotFound(object_id.to_string()))?;

        let mut removed = Vec::new();

        for chunk_ref in &manifest.chunks {
            if let Some((meta, _)) = self.chunk_store.get_mut(&chunk_ref.hash) {
                meta.ref_count = meta.ref_count.saturating_sub(1);
                if meta.ref_count == 0 {
                    self.chunk_store.remove(&chunk_ref.hash);
                    removed.push(chunk_ref.hash);
                }
            }
        }

        Ok(removed)
    }

    /// Return a reference to the `Chunk` metadata for the given hash.
    ///
    /// # Errors
    ///
    /// - [`DeduplicatorError::ChunkNotFound`] if no chunk with that hash exists.
    pub fn get_chunk(&self, hash: &ChunkHash) -> Result<&Chunk, DeduplicatorError> {
        self.chunk_store
            .get(hash)
            .map(|(meta, _)| meta)
            .ok_or(DeduplicatorError::ChunkNotFound(*hash))
    }

    /// Return `true` if a chunk with the given hash is present in the store.
    pub fn chunk_exists(&self, hash: &ChunkHash) -> bool {
        self.chunk_store.contains_key(hash)
    }

    /// Remove all chunks whose `ref_count == 0` from the store.
    ///
    /// This is a garbage-collection pass for chunks that were decremented to
    /// zero during earlier `delete_object` calls but not yet physically removed
    /// (e.g., if `delete_object` was called but removal was deferred). In the
    /// current implementation, `delete_object` removes zero-ref chunks eagerly,
    /// so this acts as a safety pass.
    ///
    /// Returns the number of chunks removed.
    pub fn compact(&mut self) -> usize {
        let before = self.chunk_store.len();
        self.chunk_store.retain(|_, (meta, _)| meta.ref_count > 0);
        before - self.chunk_store.len()
    }

    /// Compute and return a snapshot of current deduplication statistics.
    pub fn stats(&self) -> DeduplicationStats {
        let unique_chunks = self.chunk_store.len() as u64;
        let total_chunks: u64 = self
            .chunk_store
            .values()
            .map(|(meta, _)| meta.ref_count as u64)
            .sum();
        let duplicate_chunks = total_chunks.saturating_sub(unique_chunks);

        let bytes_after: u64 = self
            .chunk_store
            .values()
            .map(|(_, data)| data.len() as u64)
            .sum();

        let bytes_before = self.total_bytes_stored;

        let dedup_ratio = if bytes_before > 0 {
            1.0 - (bytes_after as f64 / bytes_before as f64)
        } else {
            0.0
        };

        let compression_ratio = bytes_before as f64 / bytes_after.max(1) as f64;

        DeduplicationStats {
            total_chunks,
            unique_chunks,
            duplicate_chunks,
            bytes_before,
            bytes_after,
            dedup_ratio,
            compression_ratio,
        }
    }

    /// Return a sorted list of all stored object IDs.
    pub fn list_objects(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.manifests.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Return the number of unique chunks currently in the store.
    pub fn chunk_count(&self) -> usize {
        self.chunk_store.len()
    }

    /// Return the number of objects currently stored.
    pub fn object_count(&self) -> usize {
        self.manifests.len()
    }

    /// Retrieve the manifest for a stored object without reconstructing data.
    pub fn get_manifest(&self, object_id: &str) -> Option<&ObjectManifest> {
        self.manifests.get(object_id)
    }

    /// Return a reference to the current chunking configuration.
    pub fn config(&self) -> &ChunkingConfig {
        &self.config
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── xorshift64 PRNG for deterministic test data (no rand crate) ─────────

    fn xorshift64(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    fn gen_bytes(seed: u64, len: usize) -> Vec<u8> {
        let mut state = seed;
        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            let v = xorshift64(&mut state);
            let bytes = v.to_le_bytes();
            let remaining = len - out.len();
            let take = remaining.min(8);
            out.extend_from_slice(&bytes[..take]);
        }
        out
    }

    fn default_dedup() -> BlockDeduplicator {
        BlockDeduplicator::with_defaults()
    }

    // ── 1. CDC Chunking tests ────────────────────────────────────────────────

    #[test]
    fn test_single_chunk_small_data() {
        let dedup = default_dedup();
        // 100 bytes < min_chunk_size (2048) → must produce exactly 1 chunk
        let data = gen_bytes(1, 100);
        let chunks = dedup.chunk_data(&data);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].1, data);
    }

    #[test]
    fn test_chunk_data_empty() {
        let dedup = default_dedup();
        let chunks = dedup.chunk_data(&[]);
        assert!(chunks.is_empty(), "empty input should produce no chunks");
    }

    #[test]
    fn test_chunk_data_exact_min() {
        let config = ChunkingConfig {
            min_chunk_size: 128,
            max_chunk_size: 65536,
            target_bits: 8, // low target bits → boundaries rare but possible
            window_size: 48,
            enable_compression: false,
        };
        let dedup = BlockDeduplicator::new(config);
        // Exactly min_chunk_size bytes — may or may not split depending on hash,
        // but the full data must be covered.
        let data = gen_bytes(42, 128);
        let chunks = dedup.chunk_data(&data);
        assert!(!chunks.is_empty());
        let total: usize = chunks.iter().map(|(_, b)| b.len()).sum();
        assert_eq!(total, 128);
    }

    #[test]
    fn test_chunk_data_max_size_boundary() {
        let config = ChunkingConfig {
            min_chunk_size: 512,
            max_chunk_size: 1024,
            target_bits: 20, // very high bits → hash boundary almost never fires
            window_size: 32,
            enable_compression: false,
        };
        let dedup = BlockDeduplicator::new(config);
        // 3 * max_chunk_size bytes → at least 3 chunks forced by max boundary
        let data = gen_bytes(7, 3 * 1024);
        let chunks = dedup.chunk_data(&data);
        assert!(
            chunks.len() >= 3,
            "expected at least 3 chunks, got {}",
            chunks.len()
        );
    }

    #[test]
    fn test_chunk_data_deterministic() {
        let dedup = default_dedup();
        let data = gen_bytes(99, 200_000);
        let c1 = dedup.chunk_data(&data);
        let c2 = dedup.chunk_data(&data);
        assert_eq!(c1.len(), c2.len());
        for (a, b) in c1.iter().zip(c2.iter()) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1, b.1);
        }
    }

    #[test]
    fn test_chunk_data_window_smaller_than_data() {
        let config = ChunkingConfig {
            min_chunk_size: 4,
            max_chunk_size: 65536,
            target_bits: 4, // lots of boundaries
            window_size: 48,
            enable_compression: false,
        };
        let dedup = BlockDeduplicator::new(config);
        // data shorter than window_size (48) but larger than min_chunk_size (4)
        let data = gen_bytes(3, 20);
        let chunks = dedup.chunk_data(&data);
        // must cover all bytes
        let total: usize = chunks.iter().map(|(_, b)| b.len()).sum();
        assert_eq!(total, 20);
    }

    #[test]
    fn test_chunk_data_covers_all_bytes() {
        let dedup = default_dedup();
        let data = gen_bytes(55, 500_000);
        let chunks = dedup.chunk_data(&data);
        let total: usize = chunks.iter().map(|(_, b)| b.len()).sum();
        assert_eq!(total, data.len(), "chunks must cover all bytes exactly");
    }

    #[test]
    fn test_chunk_hash_hex_display() {
        let hash = ChunkHash::from(0xDEADBEEF_CAFEBABE_u64);
        let s = format!("{}", hash);
        assert_eq!(s.len(), 16, "ChunkHash hex must be exactly 16 chars");
        assert!(s.chars().all(|c| c.is_ascii_hexdigit()), "must be hex");
    }

    #[test]
    fn test_chunk_hash_from_u64_roundtrip() {
        let v: u64 = 0x0102030405060708;
        let hash = ChunkHash::from(v);
        assert_eq!(hash.to_u64(), v);
    }

    #[test]
    fn test_rolling_hash_produces_boundaries() {
        // Craft data where we know a boundary fires by brute-forcing a window that
        // produces a hash value satisfying `h & mask == 0` for a small mask.
        let config = ChunkingConfig {
            min_chunk_size: 4,
            max_chunk_size: 65536,
            target_bits: 4, // mask = 0xF → boundary every ~16 bytes on average
            window_size: 4,
            enable_compression: false,
        };
        let dedup = BlockDeduplicator::new(config);
        // With 10,000 random bytes and target_bits=4, we expect many boundaries.
        let data = gen_bytes(1234, 10_000);
        let chunks = dedup.chunk_data(&data);
        // There should be multiple chunks
        assert!(
            chunks.len() > 2,
            "expected multiple hash-triggered boundaries, got {}",
            chunks.len()
        );
        // All bytes must still be accounted for
        let total: usize = chunks.iter().map(|(_, b)| b.len()).sum();
        assert_eq!(total, 10_000);
    }

    #[test]
    fn test_chunk_data_large_uniform_zeros() {
        let config = ChunkingConfig {
            min_chunk_size: 512,
            max_chunk_size: 1024,
            target_bits: 20, // hash boundary essentially never fires for uniform data
            window_size: 32,
            enable_compression: false,
        };
        let dedup = BlockDeduplicator::new(config);
        // Uniform zeros — boundaries only triggered by max_chunk_size
        let data = vec![0u8; 5 * 1024];
        let chunks = dedup.chunk_data(&data);
        // At least 5 chunks (max_chunk_size = 1024, data = 5120 bytes)
        assert!(chunks.len() >= 5);
        let total: usize = chunks.iter().map(|(_, b)| b.len()).sum();
        assert_eq!(total, 5 * 1024);
    }

    #[test]
    fn test_chunk_data_multiple_chunks() {
        let config = ChunkingConfig {
            min_chunk_size: 256,
            max_chunk_size: 1024,
            target_bits: 20, // force max-size splits
            window_size: 32,
            enable_compression: false,
        };
        let dedup = BlockDeduplicator::new(config);
        let data = gen_bytes(11, 4096);
        let chunks = dedup.chunk_data(&data);
        assert!(
            chunks.len() >= 4,
            "expected ≥4 chunks from 4096 bytes with max_size=1024, got {}",
            chunks.len()
        );
    }

    #[test]
    fn test_chunk_hash_from_bytes() {
        let bytes = [0u8, 1, 2, 3, 4, 5, 6, 7];
        let hash = ChunkHash::from_bytes(bytes);
        assert_eq!(hash.as_bytes(), &bytes);
    }

    #[test]
    fn test_chunk_data_each_chunk_nonempty() {
        let dedup = default_dedup();
        let data = gen_bytes(77, 300_000);
        let chunks = dedup.chunk_data(&data);
        for (i, (_, b)) in chunks.iter().enumerate() {
            assert!(!b.is_empty(), "chunk {} must not be empty", i);
        }
    }

    // ── 2. Store / Retrieve / Delete tests ──────────────────────────────────

    #[test]
    fn test_store_and_retrieve_small() {
        let mut dedup = default_dedup();
        let data = b"hello world".to_vec();
        dedup
            .store_object("obj1".to_string(), data.clone())
            .unwrap();
        let retrieved = dedup.retrieve_object("obj1").unwrap();
        assert_eq!(data, retrieved);
    }

    #[test]
    fn test_store_and_retrieve_large() {
        let mut dedup = default_dedup();
        let data = gen_bytes(1001, 500_000);
        dedup
            .store_object("large".to_string(), data.clone())
            .unwrap();
        let retrieved = dedup.retrieve_object("large").unwrap();
        assert_eq!(data, retrieved, "large object round-trip failed");
    }

    #[test]
    fn test_store_duplicate_objects() {
        let mut dedup = default_dedup();
        let data = gen_bytes(22, 100_000);
        dedup.store_object("a".to_string(), data.clone()).unwrap();
        dedup.store_object("b".to_string(), data.clone()).unwrap();
        let ra = dedup.retrieve_object("a").unwrap();
        let rb = dedup.retrieve_object("b").unwrap();
        assert_eq!(data, ra);
        assert_eq!(data, rb);
    }

    #[test]
    fn test_store_two_different_objects() {
        let mut dedup = default_dedup();
        let d1 = gen_bytes(101, 50_000);
        let d2 = gen_bytes(202, 60_000);
        dedup.store_object("x".to_string(), d1.clone()).unwrap();
        dedup.store_object("y".to_string(), d2.clone()).unwrap();
        assert_eq!(dedup.retrieve_object("x").unwrap(), d1);
        assert_eq!(dedup.retrieve_object("y").unwrap(), d2);
    }

    #[test]
    fn test_delete_object_removes_manifest() {
        let mut dedup = default_dedup();
        let data = gen_bytes(5, 10_000);
        dedup.store_object("del_me".to_string(), data).unwrap();
        dedup.delete_object("del_me").unwrap();
        let err = dedup.retrieve_object("del_me").unwrap_err();
        assert!(matches!(err, DeduplicatorError::ObjectNotFound(_)));
    }

    #[test]
    fn test_delete_decrements_ref_count() {
        let mut dedup = BlockDeduplicator::new(ChunkingConfig {
            min_chunk_size: 256,
            max_chunk_size: 65536,
            target_bits: 20, // very few natural splits → objects share same chunks
            window_size: 32,
            enable_compression: false,
        });
        let data = gen_bytes(9, 1024); // fits in single chunk
        dedup
            .store_object("obj_a".to_string(), data.clone())
            .unwrap();
        dedup
            .store_object("obj_b".to_string(), data.clone())
            .unwrap();

        // After storing twice, chunk should have ref_count == 2.
        // Get the hash of the single chunk.
        let manifest = dedup.get_manifest("obj_a").unwrap().clone();
        let hash = manifest.chunks[0].hash;
        assert_eq!(dedup.get_chunk(&hash).unwrap().ref_count, 2);

        // Delete one object — ref_count should drop to 1, chunk still present.
        let removed = dedup.delete_object("obj_a").unwrap();
        assert!(
            removed.is_empty(),
            "chunk still referenced by obj_b, should not be removed"
        );
        assert_eq!(dedup.get_chunk(&hash).unwrap().ref_count, 1);
    }

    #[test]
    fn test_delete_removes_unshared_chunks() {
        let mut dedup = default_dedup();
        let data = gen_bytes(13, 50_000);
        let manifest = dedup.store_object("only_obj".to_string(), data).unwrap();
        let n_chunks = manifest.chunks.len();
        let removed = dedup.delete_object("only_obj").unwrap();
        assert_eq!(
            removed.len(),
            n_chunks,
            "all unshared chunks should be removed"
        );
    }

    #[test]
    fn test_delete_nonexistent_object() {
        let mut dedup = default_dedup();
        let err = dedup.delete_object("ghost").unwrap_err();
        assert!(matches!(err, DeduplicatorError::ObjectNotFound(_)));
    }

    #[test]
    fn test_retrieve_nonexistent_object() {
        let dedup = default_dedup();
        let err = dedup.retrieve_object("ghost").unwrap_err();
        assert!(matches!(err, DeduplicatorError::ObjectNotFound(_)));
    }

    #[test]
    fn test_store_empty_data() {
        let mut dedup = default_dedup();
        let manifest = dedup.store_object("empty".to_string(), vec![]).unwrap();
        assert_eq!(manifest.total_size, 0);
        assert!(manifest.chunks.is_empty());
        let retrieved = dedup.retrieve_object("empty").unwrap();
        assert!(retrieved.is_empty());
    }

    #[test]
    fn test_store_overwrite_same_id() {
        let mut dedup = default_dedup();
        let d1 = gen_bytes(1, 1000);
        let d2 = gen_bytes(2, 2000);
        dedup.store_object("obj".to_string(), d1).unwrap();
        dedup.store_object("obj".to_string(), d2.clone()).unwrap();
        // The manifest is overwritten; retrieve should return d2.
        let retrieved = dedup.retrieve_object("obj").unwrap();
        assert_eq!(retrieved, d2);
    }

    #[test]
    fn test_manifest_chunk_order() {
        let mut dedup = default_dedup();
        let data = gen_bytes(33, 200_000);
        let manifest = dedup.store_object("ordered".to_string(), data).unwrap();
        let mut prev_offset = 0u64;
        for (i, cr) in manifest.chunks.iter().enumerate() {
            assert!(
                cr.offset_in_object >= prev_offset || i == 0,
                "chunk {} has non-monotonic offset",
                i
            );
            prev_offset = cr.offset_in_object;
        }
    }

    #[test]
    fn test_manifest_total_size() {
        let mut dedup = default_dedup();
        let data = gen_bytes(44, 150_000);
        let expected_size = data.len() as u64;
        let manifest = dedup.store_object("sized".to_string(), data).unwrap();
        assert_eq!(manifest.total_size, expected_size);
    }

    #[test]
    fn test_retrieve_exact_bytes() {
        let mut dedup = default_dedup();
        let data: Vec<u8> = (0..=255u8).cycle().take(10_000).collect();
        dedup
            .store_object("pattern".to_string(), data.clone())
            .unwrap();
        let retrieved = dedup.retrieve_object("pattern").unwrap();
        assert_eq!(data, retrieved, "byte-for-byte equality required");
    }

    #[test]
    fn test_delete_returns_removed_hashes() {
        let mut dedup = default_dedup();
        let data = gen_bytes(66, 30_000);
        let manifest = dedup.store_object("ret_hashes".to_string(), data).unwrap();
        let expected_hashes: Vec<ChunkHash> = manifest.chunks.iter().map(|cr| cr.hash).collect();
        let removed = dedup.delete_object("ret_hashes").unwrap();
        for h in &expected_hashes {
            assert!(removed.contains(h), "hash {} should be in removed list", h);
        }
    }

    #[test]
    fn test_manifest_chunk_offsets_are_contiguous() {
        let mut dedup = default_dedup();
        let data = gen_bytes(88, 300_000);
        let manifest = dedup.store_object("contiguous".to_string(), data).unwrap();
        let mut expected_offset = 0u64;
        for cr in &manifest.chunks {
            assert_eq!(
                cr.offset_in_object, expected_offset,
                "chunk offsets must be contiguous"
            );
            expected_offset += cr.chunk_length as u64;
        }
    }

    // ── 3. Dedup ratio / ref counting tests ─────────────────────────────────

    #[test]
    fn test_ref_count_increments_on_duplicate() {
        let mut dedup = BlockDeduplicator::new(ChunkingConfig {
            min_chunk_size: 256,
            max_chunk_size: 65536,
            target_bits: 20,
            window_size: 32,
            enable_compression: false,
        });
        let data = gen_bytes(111, 1024); // single chunk
        dedup.store_object("d1".to_string(), data.clone()).unwrap();
        dedup.store_object("d2".to_string(), data.clone()).unwrap();

        let manifest = dedup.get_manifest("d1").unwrap().clone();
        let hash = manifest.chunks[0].hash;
        assert_eq!(
            dedup.get_chunk(&hash).unwrap().ref_count,
            2,
            "ref_count should be 2 after two stores"
        );
    }

    #[test]
    fn test_ref_count_one_unique() {
        let mut dedup = default_dedup();
        let data = gen_bytes(222, 50_000);
        let manifest = dedup.store_object("unique_obj".to_string(), data).unwrap();
        for cr in &manifest.chunks {
            let chunk = dedup.get_chunk(&cr.hash).unwrap();
            assert_eq!(
                chunk.ref_count, 1,
                "unique chunk should have ref_count == 1"
            );
        }
    }

    #[test]
    fn test_dedup_ratio_identical_objects() {
        let mut dedup = default_dedup();
        let data = gen_bytes(333, 200_000);
        dedup
            .store_object("copy1".to_string(), data.clone())
            .unwrap();
        dedup
            .store_object("copy2".to_string(), data.clone())
            .unwrap();

        let stats = dedup.stats();
        // bytes_before = 2 * 200_000; bytes_after = ~200_000
        // dedup_ratio should be close to 0.5
        assert!(
            stats.dedup_ratio > 0.3,
            "expected dedup ratio > 0.3, got {}",
            stats.dedup_ratio
        );
    }

    #[test]
    fn test_dedup_ratio_unique_objects() {
        let mut dedup = default_dedup();
        let d1 = gen_bytes(401, 100_000);
        let d2 = gen_bytes(402, 100_000);
        dedup.store_object("u1".to_string(), d1).unwrap();
        dedup.store_object("u2".to_string(), d2).unwrap();

        let stats = dedup.stats();
        // No sharing expected → bytes_before ≈ bytes_after → ratio ≈ 0
        assert!(
            stats.dedup_ratio < 0.2,
            "expected low dedup ratio for unique data, got {}",
            stats.dedup_ratio
        );
    }

    #[test]
    fn test_stats_total_chunks() {
        let mut dedup = BlockDeduplicator::new(ChunkingConfig {
            min_chunk_size: 256,
            max_chunk_size: 65536,
            target_bits: 20,
            window_size: 32,
            enable_compression: false,
        });
        let data = gen_bytes(500, 1024); // single chunk, ref counted twice
        dedup.store_object("s1".to_string(), data.clone()).unwrap();
        dedup.store_object("s2".to_string(), data.clone()).unwrap();

        let stats = dedup.stats();
        // 1 unique chunk with ref_count == 2 → total_chunks == 2
        assert_eq!(stats.total_chunks, 2);
        assert_eq!(stats.unique_chunks, 1);
        assert_eq!(stats.duplicate_chunks, 1);
    }

    #[test]
    fn test_stats_unique_chunks_equals_store_size() {
        let mut dedup = default_dedup();
        let d1 = gen_bytes(601, 50_000);
        let d2 = gen_bytes(602, 50_000);
        dedup.store_object("a".to_string(), d1).unwrap();
        dedup.store_object("b".to_string(), d2).unwrap();
        let stats = dedup.stats();
        assert_eq!(stats.unique_chunks, dedup.chunk_count() as u64);
    }

    #[test]
    fn test_stats_bytes_before_after() {
        let mut dedup = default_dedup();
        let data = gen_bytes(700, 100_000);
        dedup
            .store_object("ba_test".to_string(), data.clone())
            .unwrap();
        let stats = dedup.stats();
        assert_eq!(stats.bytes_before, 100_000);
        // bytes_after should equal the sum of stored chunk sizes
        let expected_after: u64 = dedup
            .chunk_store
            .values()
            .map(|(_, d)| d.len() as u64)
            .sum();
        assert_eq!(stats.bytes_after, expected_after);
    }

    #[test]
    fn test_chunk_exists_true() {
        let mut dedup = default_dedup();
        let data = b"some data for hashing".to_vec();
        let manifest = dedup.store_object("ce_true".to_string(), data).unwrap();
        let hash = manifest.chunks[0].hash;
        assert!(dedup.chunk_exists(&hash));
    }

    #[test]
    fn test_chunk_exists_false() {
        let dedup = default_dedup();
        let fake_hash = ChunkHash::from(0xFFFF_FFFF_FFFF_FFFFu64);
        assert!(!dedup.chunk_exists(&fake_hash));
    }

    #[test]
    fn test_get_chunk_success() {
        let mut dedup = default_dedup();
        let data = b"chunk metadata test".to_vec();
        let expected_len = data.len();
        let manifest = dedup.store_object("gc_ok".to_string(), data).unwrap();
        let hash = manifest.chunks[0].hash;
        let chunk = dedup.get_chunk(&hash).unwrap();
        assert_eq!(chunk.hash, hash);
        assert_eq!(chunk.length, expected_len);
        assert_eq!(chunk.ref_count, 1);
    }

    #[test]
    fn test_get_chunk_not_found() {
        let dedup = default_dedup();
        let missing = ChunkHash::from(0xDEAD_C0DE_0000_0001u64);
        let err = dedup.get_chunk(&missing).unwrap_err();
        assert!(matches!(err, DeduplicatorError::ChunkNotFound(_)));
    }

    #[test]
    fn test_dedup_shared_chunks_across_objects() {
        // Two objects that are identical share all their chunks.
        // We store the same data under two different object IDs and confirm
        // every chunk has ref_count == 2.
        let config = ChunkingConfig {
            min_chunk_size: 512,
            max_chunk_size: 4096,
            target_bits: 20,
            window_size: 32,
            enable_compression: false,
        };
        let mut dedup = BlockDeduplicator::new(config);
        // Use 4096 bytes → exactly one max-size chunk, guaranteed identical for both objects.
        let data = gen_bytes(801, 4096);

        dedup
            .store_object("obj1".to_string(), data.clone())
            .unwrap();
        dedup
            .store_object("obj2".to_string(), data.clone())
            .unwrap();

        // All chunks should have ref_count == 2 (shared between both objects)
        let manifest1 = dedup.get_manifest("obj1").unwrap().clone();
        for cr in &manifest1.chunks {
            let chunk = dedup.get_chunk(&cr.hash).unwrap();
            assert_eq!(
                chunk.ref_count, 2,
                "shared chunk {} should have ref_count == 2",
                cr.hash
            );
        }
    }

    // ── 4. Compact tests ─────────────────────────────────────────────────────

    #[test]
    fn test_compact_removes_orphans() {
        let mut dedup = default_dedup();
        let data = gen_bytes(901, 50_000);
        dedup.store_object("to_compact".to_string(), data).unwrap();
        // Force ref_count to 0 by directly manipulating (simulate partial delete bug).
        // Instead, do a normal delete and then compact.
        dedup.delete_object("to_compact").unwrap();
        // After delete, chunks with ref_count=0 are already removed eagerly.
        // compact() should find nothing extra to remove.
        let removed = dedup.compact();
        // Whether 0 or more, the store must be empty afterwards.
        assert_eq!(dedup.chunk_count(), 0);
        let _ = removed; // count is valid either way
    }

    #[test]
    fn test_compact_keeps_referenced() {
        let mut dedup = BlockDeduplicator::new(ChunkingConfig {
            min_chunk_size: 256,
            max_chunk_size: 65536,
            target_bits: 20,
            window_size: 32,
            enable_compression: false,
        });
        let data = gen_bytes(1001, 1024);
        dedup
            .store_object("keep_a".to_string(), data.clone())
            .unwrap();
        dedup
            .store_object("keep_b".to_string(), data.clone())
            .unwrap();
        dedup.delete_object("keep_a").unwrap();

        // chunk still referenced by keep_b → compact must keep it
        let removed = dedup.compact();
        assert_eq!(removed, 0, "no orphans should exist");
        assert_eq!(dedup.chunk_count(), 1, "chunk still referenced by keep_b");
    }

    #[test]
    fn test_compact_returns_count() {
        let mut dedup = default_dedup();
        let data = gen_bytes(1100, 50_000);
        let manifest = dedup.store_object("count_test".to_string(), data).unwrap();
        let n_chunks = manifest.chunks.len();

        // Manually set all ref_counts to 0 to simulate orphaned chunks
        // (bypassing normal delete to test compact independently).
        for (meta, _) in dedup.chunk_store.values_mut() {
            meta.ref_count = 0;
        }
        // Also remove the manifest so list_objects is clean.
        dedup.manifests.remove("count_test");

        let removed = dedup.compact();
        assert_eq!(
            removed, n_chunks,
            "compact should remove all orphaned chunks"
        );
        assert_eq!(dedup.chunk_count(), 0);
    }

    #[test]
    fn test_compact_empty_store() {
        let mut dedup = default_dedup();
        let removed = dedup.compact();
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_compact_idempotent() {
        let mut dedup = default_dedup();
        let data = gen_bytes(1200, 50_000);
        dedup.store_object("idem".to_string(), data).unwrap();
        dedup.delete_object("idem").unwrap();
        let _r1 = dedup.compact();
        let r2 = dedup.compact();
        assert_eq!(r2, 0, "second compact should remove nothing");
    }

    #[test]
    fn test_compact_after_delete_all() {
        let mut dedup = default_dedup();
        let d1 = gen_bytes(1301, 60_000);
        let d2 = gen_bytes(1302, 60_000);
        dedup.store_object("c1".to_string(), d1).unwrap();
        dedup.store_object("c2".to_string(), d2).unwrap();
        dedup.delete_object("c1").unwrap();
        dedup.delete_object("c2").unwrap();
        dedup.compact();
        assert_eq!(dedup.chunk_count(), 0, "all chunks should be gone");
        assert_eq!(dedup.object_count(), 0);
    }

    // ── 5. Error cases ───────────────────────────────────────────────────────

    #[test]
    fn test_error_object_not_found_display() {
        let err = DeduplicatorError::ObjectNotFound("my-special-object".to_string());
        let msg = format!("{}", err);
        assert!(
            msg.contains("my-special-object"),
            "error message must contain object id: {}",
            msg
        );
    }

    #[test]
    fn test_error_chunk_not_found_display() {
        let hash = ChunkHash::from(0xABCD_EF01_2345_6789u64);
        let err = DeduplicatorError::ChunkNotFound(hash);
        let msg = format!("{}", err);
        // The display includes the hash hex
        assert!(!msg.is_empty(), "error message must not be empty");
    }

    #[test]
    fn test_chunk_hash_equality() {
        let b = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let h1 = ChunkHash::from_bytes(b);
        let h2 = ChunkHash::from_bytes(b);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_deduplicator_list_objects_empty() {
        let dedup = default_dedup();
        let objs = dedup.list_objects();
        assert!(objs.is_empty());
    }

    #[test]
    fn test_deduplicator_list_objects_sorted() {
        let mut dedup = default_dedup();
        let ids = ["zebra", "alpha", "mango", "beta"];
        for id in &ids {
            dedup
                .store_object((*id).to_string(), gen_bytes(42, 100))
                .unwrap();
        }
        let listed = dedup.list_objects();
        let mut expected: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
        expected.sort();
        assert_eq!(listed, expected);
    }

    // ── 6. Integration / misc tests ──────────────────────────────────────────

    #[test]
    fn test_large_object_chunked_correctly() {
        let mut dedup = default_dedup();
        let data = gen_bytes(9999, 1_000_000); // 1 MB
        dedup
            .store_object("mega".to_string(), data.clone())
            .unwrap();
        let retrieved = dedup.retrieve_object("mega").unwrap();
        assert_eq!(data, retrieved, "1 MB round-trip failed");
    }

    #[test]
    fn test_many_small_objects() {
        let mut dedup = default_dedup();
        let mut originals: Vec<Vec<u8>> = Vec::new();
        let mut seed = 12345u64;
        for i in 0..100 {
            let data = gen_bytes(seed, 500 + i * 7);
            seed = seed.wrapping_add(1);
            dedup
                .store_object(format!("small_{}", i), data.clone())
                .unwrap();
            originals.push(data);
        }
        for (i, original) in originals.iter().enumerate() {
            let retrieved = dedup.retrieve_object(&format!("small_{}", i)).unwrap();
            assert_eq!(*original, retrieved, "small object {} failed round-trip", i);
        }
    }

    #[test]
    fn test_config_default_values() {
        let cfg = ChunkingConfig::default();
        assert_eq!(cfg.min_chunk_size, 2048);
        assert_eq!(cfg.max_chunk_size, 65536);
        assert_eq!(cfg.target_bits, 13);
        assert_eq!(cfg.window_size, 48);
        assert!(!cfg.enable_compression);
    }

    #[test]
    fn test_deduplicator_new_starts_empty() {
        let dedup = default_dedup();
        assert_eq!(dedup.chunk_count(), 0);
        assert_eq!(dedup.object_count(), 0);
        assert!(dedup.list_objects().is_empty());
        let stats = dedup.stats();
        assert_eq!(stats.total_chunks, 0);
        assert_eq!(stats.bytes_before, 0);
    }

    #[test]
    fn test_full_dedup_workflow() {
        let mut dedup = default_dedup();

        // 1. Store two objects with overlapping content.
        let shared = gen_bytes(5555, 200_000);
        let extra1 = gen_bytes(6001, 50_000);
        let extra2 = gen_bytes(6002, 50_000);

        let mut obj_a = shared.clone();
        obj_a.extend_from_slice(&extra1);
        let mut obj_b = shared.clone();
        obj_b.extend_from_slice(&extra2);

        let m_a = dedup
            .store_object("workflow_a".to_string(), obj_a.clone())
            .unwrap();
        let m_b = dedup
            .store_object("workflow_b".to_string(), obj_b.clone())
            .unwrap();

        // 2. Both objects retrievable and correct.
        assert_eq!(dedup.retrieve_object("workflow_a").unwrap(), obj_a);
        assert_eq!(dedup.retrieve_object("workflow_b").unwrap(), obj_b);

        // 3. Objects are listed.
        let listed = dedup.list_objects();
        assert!(listed.contains(&"workflow_a".to_string()));
        assert!(listed.contains(&"workflow_b".to_string()));

        // 4. Stats reflect two stores.
        let stats = dedup.stats();
        assert_eq!(stats.bytes_before, (obj_a.len() + obj_b.len()) as u64);

        // 5. Delete one object.
        dedup.delete_object("workflow_a").unwrap();
        assert!(dedup
            .retrieve_object("workflow_a")
            .unwrap_err()
            .to_string()
            .contains("workflow_a"));

        // 6. Other object still intact.
        assert_eq!(dedup.retrieve_object("workflow_b").unwrap(), obj_b);

        // 7. Compact.
        dedup.compact();

        // 8. Remaining manifest is consistent.
        let n_chunks_b = m_b.chunks.len();
        assert!(dedup.chunk_count() > 0, "workflow_b chunks still present");
        // But fewer than the combined chunk set.
        assert!(dedup.chunk_count() <= m_a.chunks.len() + n_chunks_b);

        // 9. Delete last object and compact → empty store.
        dedup.delete_object("workflow_b").unwrap();
        dedup.compact();
        assert_eq!(dedup.chunk_count(), 0);
        assert_eq!(dedup.object_count(), 0);
    }

    #[test]
    fn test_chunk_data_chunk_hashes_are_content_based() {
        let dedup = default_dedup();
        let d1 = vec![0u8; 100];
        let d2 = vec![1u8; 100];
        let c1 = dedup.chunk_data(&d1);
        let c2 = dedup.chunk_data(&d2);
        assert_ne!(
            c1[0].0, c2[0].0,
            "different content must produce different hashes"
        );
    }

    #[test]
    fn test_boundary_mask_calculation() {
        let config = ChunkingConfig {
            target_bits: 13,
            ..ChunkingConfig::default()
        };
        let mask = config.boundary_mask();
        assert_eq!(mask, (1u64 << 13) - 1);
        assert_eq!(mask, 8191);
    }

    #[test]
    fn test_store_single_byte() {
        let mut dedup = default_dedup();
        let data = vec![42u8];
        let manifest = dedup
            .store_object("single_byte".to_string(), data.clone())
            .unwrap();
        assert_eq!(manifest.total_size, 1);
        assert_eq!(manifest.chunks.len(), 1);
        let retrieved = dedup.retrieve_object("single_byte").unwrap();
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_stats_compression_ratio_at_least_one() {
        let mut dedup = default_dedup();
        let data = gen_bytes(7777, 100_000);
        dedup.store_object("cr_test".to_string(), data).unwrap();
        let stats = dedup.stats();
        // With no actual compression, bytes_before == bytes_after → ratio == 1.0
        assert!(
            (stats.compression_ratio - 1.0).abs() < 1e-6,
            "expected ratio ~1.0, got {}",
            stats.compression_ratio
        );
    }

    #[test]
    fn test_chunk_length_matches_metadata() {
        let mut dedup = default_dedup();
        let data = gen_bytes(8888, 200_000);
        let manifest = dedup.store_object("len_check".to_string(), data).unwrap();
        for cr in &manifest.chunks {
            let chunk_meta = dedup.get_chunk(&cr.hash).unwrap();
            assert_eq!(
                chunk_meta.length, cr.chunk_length,
                "chunk metadata length must match ChunkRef length"
            );
        }
    }

    #[test]
    fn test_object_count_updates_correctly() {
        let mut dedup = default_dedup();
        assert_eq!(dedup.object_count(), 0);
        dedup
            .store_object("oc1".to_string(), gen_bytes(1, 1000))
            .unwrap();
        assert_eq!(dedup.object_count(), 1);
        dedup
            .store_object("oc2".to_string(), gen_bytes(2, 1000))
            .unwrap();
        assert_eq!(dedup.object_count(), 2);
        dedup.delete_object("oc1").unwrap();
        assert_eq!(dedup.object_count(), 1);
        dedup.delete_object("oc2").unwrap();
        assert_eq!(dedup.object_count(), 0);
    }
}
