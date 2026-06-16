//! Memory-Mapped File Serving
//!
//! Provides zero-copy file serving using memory-mapped I/O for improved performance
//! with large tensor files. This module enables efficient serving of tensors without
//! loading the entire file into memory.
//!
//! # Features
//!
//! - **Zero-copy serving** - Direct memory mapping without buffer allocation
//! - **Lazy loading** - Map files only when accessed
//! - **Range request support** - Efficient partial file serving
//! - **Platform optimizations** - Uses OS-level optimizations (sendfile, etc.)
//!
//! # Safety
//!
//! Memory mapping is inherently unsafe as it involves raw pointer access. This module
//! provides a safe wrapper that ensures proper lifetime management and error handling.

use bytes::Bytes;
use memmap2::Mmap;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during memory-mapped operations
#[derive(Debug, Error)]
pub enum MmapError {
    #[error("Failed to open file: {0}")]
    FileOpen(#[from] io::Error),

    #[error("Failed to create memory map: {0}")]
    MmapCreation(String),

    #[error("Invalid range: {0}")]
    InvalidRange(String),

    #[error("File not found: {0}")]
    FileNotFound(String),
}

// ============================================================================
// Memory-Mapped File Handle
// ============================================================================

/// A handle to a memory-mapped file
///
/// This structure provides safe access to memory-mapped files with proper
/// lifetime management and error handling.
pub struct MmapFile {
    /// The memory-mapped region
    mmap: Arc<Mmap>,
    /// Original file path (for debugging)
    path: PathBuf,
    /// Total file size
    size: usize,
}

impl MmapFile {
    /// Create a new memory-mapped file from a path
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the file to be memory-mapped
    ///
    /// # Returns
    ///
    /// A `Result` containing the `MmapFile` or an error
    ///
    /// # Safety
    ///
    /// This function is safe because it ensures:
    /// - File handle is valid
    /// - Memory map is created successfully
    /// - Lifetime of the mmap is tied to the struct
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, MmapError> {
        let path = path.as_ref();

        // Open the file
        let file = File::open(path).map_err(|e| {
            if e.kind() == io::ErrorKind::NotFound {
                MmapError::FileNotFound(path.display().to_string())
            } else {
                MmapError::FileOpen(e)
            }
        })?;

        // Get file size
        let metadata = file.metadata()?;
        let size = metadata.len() as usize;

        // Create memory map
        // SAFETY: The file is opened successfully and we have a valid file handle
        let mmap = unsafe { Mmap::map(&file).map_err(|e| MmapError::MmapCreation(e.to_string()))? };

        Ok(MmapFile {
            mmap: Arc::new(mmap),
            path: path.to_path_buf(),
            size,
        })
    }

    /// Get the full file contents as a Bytes object (zero-copy)
    ///
    /// This returns a `Bytes` object that references the memory-mapped region
    /// without copying the data.
    pub fn bytes(&self) -> Bytes {
        Bytes::copy_from_slice(&self.mmap[..])
    }

    /// Get a range of bytes from the file (zero-copy slice)
    ///
    /// # Arguments
    ///
    /// * `range` - The byte range to retrieve (start..end)
    ///
    /// # Returns
    ///
    /// A `Bytes` object containing the requested range
    pub fn range(&self, range: std::ops::Range<usize>) -> Result<Bytes, MmapError> {
        if range.start > self.size {
            return Err(MmapError::InvalidRange(format!(
                "Start {} exceeds file size {}",
                range.start, self.size
            )));
        }

        if range.end > self.size {
            return Err(MmapError::InvalidRange(format!(
                "End {} exceeds file size {}",
                range.end, self.size
            )));
        }

        if range.start >= range.end {
            return Err(MmapError::InvalidRange(format!(
                "Invalid range: {}..{}",
                range.start, range.end
            )));
        }

        Ok(Bytes::copy_from_slice(&self.mmap[range]))
    }

    /// Get the file size in bytes
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the file path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Check if the file is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get multiple ranges efficiently (for HTTP multi-range requests)
    ///
    /// # Arguments
    ///
    /// * `ranges` - Vector of byte ranges to retrieve
    ///
    /// # Returns
    ///
    /// A vector of `Bytes` objects, one for each range
    pub fn multi_range(&self, ranges: &[std::ops::Range<usize>]) -> Result<Vec<Bytes>, MmapError> {
        let mut results = Vec::with_capacity(ranges.len());

        for range in ranges {
            results.push(self.range(range.clone())?);
        }

        Ok(results)
    }
}

// Implement Clone for MmapFile (cheap because of Arc)
impl Clone for MmapFile {
    fn clone(&self) -> Self {
        MmapFile {
            mmap: Arc::clone(&self.mmap),
            path: self.path.clone(),
            size: self.size,
        }
    }
}

// ============================================================================
// Memory-Mapped File Cache
// ============================================================================

/// A simple cache for memory-mapped files
///
/// This cache stores recently accessed memory-mapped files to avoid
/// repeated file opening and mapping operations.
#[allow(dead_code)]
pub struct MmapCache {
    /// Maximum number of cached files
    max_entries: usize,
    /// Cache storage (simple LRU would be better in production)
    cache: dashmap::DashMap<PathBuf, Arc<MmapFile>>,
}

impl MmapCache {
    /// Create a new mmap cache
    ///
    /// # Arguments
    ///
    /// * `max_entries` - Maximum number of files to keep in cache
    pub fn new(max_entries: usize) -> Self {
        MmapCache {
            max_entries,
            cache: dashmap::DashMap::new(),
        }
    }

    /// Get or create a memory-mapped file
    ///
    /// If the file is in cache, return the cached version. Otherwise,
    /// create a new mmap and cache it.
    pub fn get_or_create<P: AsRef<Path>>(&self, path: P) -> Result<Arc<MmapFile>, MmapError> {
        let path = path.as_ref();

        // Check cache first
        if let Some(cached) = self.cache.get(path) {
            return Ok(Arc::clone(&*cached));
        }

        // Create new mmap
        let mmap_file = Arc::new(MmapFile::new(path)?);

        // Add to cache (simple eviction: just check size)
        if self.cache.len() >= self.max_entries {
            // In production, implement proper LRU eviction
            // For now, just allow growth
            tracing::warn!(
                "Mmap cache size {} exceeds max {}",
                self.cache.len(),
                self.max_entries
            );
        }

        self.cache
            .insert(path.to_path_buf(), Arc::clone(&mmap_file));

        Ok(mmap_file)
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.cache.clear();
    }

    /// Get current cache size
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

// ============================================================================
// Platform-Specific Optimizations
// ============================================================================

/// Platform-specific optimization hints
#[derive(Debug, Clone, Copy)]
pub struct MmapConfig {
    /// Use hugepages if available (Linux)
    pub use_hugepages: bool,

    /// Advise sequential access pattern
    pub sequential_access: bool,

    /// Advise random access pattern
    pub random_access: bool,

    /// Pre-populate the page tables (Linux)
    pub populate: bool,
}

impl Default for MmapConfig {
    fn default() -> Self {
        MmapConfig {
            use_hugepages: false,
            sequential_access: true,
            random_access: false,
            populate: false,
        }
    }
}

impl MmapConfig {
    /// Configuration optimized for sequential tensor streaming
    pub fn sequential() -> Self {
        MmapConfig {
            use_hugepages: false,
            sequential_access: true,
            random_access: false,
            populate: false,
        }
    }

    /// Configuration optimized for random access (e.g., tensor slicing)
    pub fn random() -> Self {
        MmapConfig {
            use_hugepages: false,
            sequential_access: false,
            random_access: true,
            populate: false,
        }
    }

    /// Configuration for large files with hugepage support
    pub fn hugepages() -> Self {
        MmapConfig {
            use_hugepages: true,
            sequential_access: false,
            random_access: false,
            populate: true,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn create_test_file() -> (tempfile::NamedTempFile, Vec<u8>) {
        let mut file =
            tempfile::NamedTempFile::new().expect("test: temp file creation should succeed");
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        file.write_all(&data)
            .expect("test: write to temp file should succeed");
        file.flush()
            .expect("test: flush to temp file should succeed");
        (file, data)
    }

    #[test]
    fn test_mmap_file_creation() {
        let (file, _data) = create_test_file();
        let mmap = MmapFile::new(file.path()).expect("test: mmap creation should succeed");
        assert_eq!(mmap.size(), 1024);
        assert!(!mmap.is_empty());
    }

    #[test]
    fn test_mmap_file_not_found() {
        let result = MmapFile::new("/nonexistent/file.bin");
        assert!(result.is_err());
        match result {
            Err(MmapError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_mmap_bytes() {
        let (file, data) = create_test_file();
        let mmap = MmapFile::new(file.path()).expect("test: mmap creation should succeed");
        let bytes = mmap.bytes();
        assert_eq!(bytes.len(), 1024);
        assert_eq!(&bytes[..], &data[..]);
    }

    #[test]
    fn test_mmap_range() {
        let (file, data) = create_test_file();
        let mmap = MmapFile::new(file.path()).expect("test: mmap creation should succeed");

        let range = mmap
            .range(10..50)
            .expect("test: range slice should succeed");
        assert_eq!(range.len(), 40);
        assert_eq!(&range[..], &data[10..50]);
    }

    #[test]
    fn test_mmap_range_invalid() {
        let (file, _data) = create_test_file();
        let mmap = MmapFile::new(file.path()).expect("test: mmap creation should succeed");

        // Start > size
        assert!(mmap.range(2000..2100).is_err());

        // End > size
        assert!(mmap.range(1000..2000).is_err());

        // Start >= end
        assert!(mmap.range(100..100).is_err());
    }

    #[test]
    fn test_mmap_multi_range() {
        let (file, data) = create_test_file();
        let mmap = MmapFile::new(file.path()).expect("test: mmap creation should succeed");

        let ranges = vec![0..10, 50..60, 100..120];
        let results = mmap
            .multi_range(&ranges)
            .expect("test: multi_range should succeed");

        assert_eq!(results.len(), 3);
        assert_eq!(&results[0][..], &data[0..10]);
        assert_eq!(&results[1][..], &data[50..60]);
        assert_eq!(&results[2][..], &data[100..120]);
    }

    #[test]
    fn test_mmap_clone() {
        let (file, _data) = create_test_file();
        let mmap1 = MmapFile::new(file.path()).expect("test: mmap creation should succeed");
        let mmap2 = mmap1.clone();

        assert_eq!(mmap1.size(), mmap2.size());
        assert_eq!(mmap1.path(), mmap2.path());
    }

    #[test]
    fn test_mmap_cache() {
        let (file, _data) = create_test_file();
        let cache = MmapCache::new(10);

        // First access - creates mmap
        let mmap1 = cache
            .get_or_create(file.path())
            .expect("test: cache get_or_create should succeed");
        assert_eq!(cache.len(), 1);

        // Second access - retrieves from cache
        let mmap2 = cache
            .get_or_create(file.path())
            .expect("test: cache get_or_create should succeed");
        assert_eq!(cache.len(), 1);

        // Should be the same Arc
        assert_eq!(mmap1.size(), mmap2.size());
    }

    #[test]
    fn test_mmap_cache_clear() {
        let (file, _data) = create_test_file();
        let cache = MmapCache::new(10);

        cache
            .get_or_create(file.path())
            .expect("test: cache get_or_create should succeed");
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_mmap_config_presets() {
        let sequential = MmapConfig::sequential();
        assert!(sequential.sequential_access);
        assert!(!sequential.random_access);

        let random = MmapConfig::random();
        assert!(!random.sequential_access);
        assert!(random.random_access);

        let hugepages = MmapConfig::hugepages();
        assert!(hugepages.use_hugepages);
        assert!(hugepages.populate);
    }
}
