//! Shared memory support for zero-copy IPC
//!
//! Provides mmap-based buffer sharing for:
//! - Cross-process tensor sharing
//! - Zero-copy IPC between processes
//! - Memory-efficient model serving

use crate::arrow::TensorDtype;
use memmap2::{Mmap, MmapMut};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Shared memory region for tensor data
pub struct SharedTensorBuffer {
    /// Memory-mapped region
    mmap: MmapMut,
    /// Header with metadata
    header: SharedBufferHeader,
    /// Path to the backing file
    path: PathBuf,
}

/// Header stored at the beginning of shared memory
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SharedBufferHeader {
    /// Magic number for validation
    pub magic: u64,
    /// Version number
    pub version: u32,
    /// Flags
    pub flags: u32,
    /// Total size of the buffer
    pub total_size: u64,
    /// Data offset (after header and metadata)
    pub data_offset: u64,
    /// Number of tensors
    pub num_tensors: u32,
    /// Checksum of data
    pub checksum: u64,
    /// Reference count (for multi-process access)
    pub ref_count: u64,
}

impl SharedBufferHeader {
    const MAGIC: u64 = 0x4950_4652_5354_454E; // "IPFRSTN"

    /// Create a new header
    pub fn new(total_size: u64, data_offset: u64, num_tensors: u32) -> Self {
        Self {
            magic: Self::MAGIC,
            version: 1,
            flags: 0,
            total_size,
            data_offset,
            num_tensors,
            checksum: 0,
            ref_count: 1,
        }
    }

    /// Validate the header
    pub fn validate(&self) -> bool {
        self.magic == Self::MAGIC && self.version == 1
    }
}

/// Metadata for tensors in shared memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedTensorInfo {
    /// Tensor name
    pub name: String,
    /// Data type
    pub dtype: TensorDtype,
    /// Shape
    pub shape: Vec<usize>,
    /// Offset from data start
    pub offset: usize,
    /// Size in bytes
    pub size: usize,
}

impl SharedTensorBuffer {
    /// Create a new shared tensor buffer
    pub fn create<P: AsRef<Path>>(
        path: P,
        size: usize,
        tensors: &[SharedTensorInfo],
    ) -> Result<Self, SharedMemoryError> {
        let path = path.as_ref().to_path_buf();

        // Serialize tensor metadata
        let metadata_json = serde_json::to_vec(tensors)?;
        let metadata_size = metadata_json.len();

        // Calculate total size needed
        let header_size = std::mem::size_of::<SharedBufferHeader>();
        let metadata_offset = header_size;
        let data_offset = metadata_offset + metadata_size + 8; // 8 bytes for metadata length
        let total_size = data_offset + size;

        // Create and size the file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        file.set_len(total_size as u64)?;

        // Memory map the file
        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        // Write header
        let header =
            SharedBufferHeader::new(total_size as u64, data_offset as u64, tensors.len() as u32);
        let header_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &header as *const SharedBufferHeader as *const u8,
                std::mem::size_of::<SharedBufferHeader>(),
            )
        };
        mmap[..header_size].copy_from_slice(header_bytes);

        // Write metadata length and metadata
        let metadata_len_bytes = (metadata_size as u64).to_le_bytes();
        mmap[metadata_offset..metadata_offset + 8].copy_from_slice(&metadata_len_bytes);
        mmap[metadata_offset + 8..metadata_offset + 8 + metadata_size]
            .copy_from_slice(&metadata_json);

        mmap.flush()?;

        Ok(Self { mmap, header, path })
    }

    /// Open an existing shared tensor buffer
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, SharedMemoryError> {
        let path = path.as_ref().to_path_buf();

        let file = OpenOptions::new().read(true).write(true).open(&path)?;

        let mmap = unsafe { MmapMut::map_mut(&file)? };

        // Read and validate header
        let header_size = std::mem::size_of::<SharedBufferHeader>();
        if mmap.len() < header_size {
            return Err(SharedMemoryError::InvalidFormat("File too small".into()));
        }

        let header: SharedBufferHeader =
            unsafe { std::ptr::read(mmap.as_ptr() as *const SharedBufferHeader) };

        if !header.validate() {
            return Err(SharedMemoryError::InvalidFormat(
                "Invalid header magic or version".into(),
            ));
        }

        Ok(Self { mmap, header, path })
    }

    /// Open read-only
    pub fn open_readonly<P: AsRef<Path>>(
        path: P,
    ) -> Result<SharedTensorBufferReadOnly, SharedMemoryError> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let header_size = std::mem::size_of::<SharedBufferHeader>();
        if mmap.len() < header_size {
            return Err(SharedMemoryError::InvalidFormat("File too small".into()));
        }

        let header: SharedBufferHeader =
            unsafe { std::ptr::read(mmap.as_ptr() as *const SharedBufferHeader) };

        if !header.validate() {
            return Err(SharedMemoryError::InvalidFormat(
                "Invalid header magic or version".into(),
            ));
        }

        Ok(SharedTensorBufferReadOnly { mmap, header, path })
    }

    /// Get tensor metadata
    pub fn tensor_metadata(&self) -> Result<Vec<SharedTensorInfo>, SharedMemoryError> {
        let header_size = std::mem::size_of::<SharedBufferHeader>();

        // Read metadata length
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&self.mmap[header_size..header_size + 8]);
        let metadata_len = u64::from_le_bytes(len_bytes) as usize;

        // Read metadata
        let metadata_bytes = &self.mmap[header_size + 8..header_size + 8 + metadata_len];
        let tensors: Vec<SharedTensorInfo> = serde_json::from_slice(metadata_bytes)?;

        Ok(tensors)
    }

    /// Get mutable data slice for a tensor
    pub fn tensor_data_mut(&mut self, info: &SharedTensorInfo) -> &mut [u8] {
        let start = self.header.data_offset as usize + info.offset;
        let end = start + info.size;
        &mut self.mmap[start..end]
    }

    /// Get data slice for a tensor
    pub fn tensor_data(&self, info: &SharedTensorInfo) -> &[u8] {
        let start = self.header.data_offset as usize + info.offset;
        let end = start + info.size;
        &self.mmap[start..end]
    }

    /// Write tensor data
    pub fn write_tensor<T: Copy>(&mut self, info: &SharedTensorInfo, data: &[T]) {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        self.tensor_data_mut(info).copy_from_slice(bytes);
    }

    /// Read tensor data as typed Vec (safe copy)
    pub fn read_tensor<T: Copy + Default>(&self, info: &SharedTensorInfo) -> Vec<T> {
        let bytes = self.tensor_data(info);
        let elem_size = std::mem::size_of::<T>();
        let count = bytes.len() / elem_size;
        let mut result = vec![T::default(); count];

        // Safe copy using byte manipulation
        let result_bytes = unsafe {
            std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, count * elem_size)
        };
        result_bytes.copy_from_slice(&bytes[..count * elem_size]);
        result
    }

    /// Update checksum
    pub fn update_checksum(&mut self) {
        let data_start = self.header.data_offset as usize;
        let data = &self.mmap[data_start..];

        // Simple checksum (could use CRC32 or Blake3)
        let checksum: u64 = data.iter().fold(0u64, |acc, &b| acc.wrapping_add(b as u64));

        // Update header
        let header_bytes = &mut self.mmap[..std::mem::size_of::<SharedBufferHeader>()];
        let offset = std::mem::offset_of!(SharedBufferHeader, checksum);
        header_bytes[offset..offset + 8].copy_from_slice(&checksum.to_le_bytes());
    }

    /// Flush changes to disk
    pub fn flush(&self) -> Result<(), SharedMemoryError> {
        self.mmap.flush()?;
        Ok(())
    }

    /// Get the path to the backing file
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get total size
    pub fn size(&self) -> usize {
        self.header.total_size as usize
    }
}

/// Read-only shared tensor buffer
pub struct SharedTensorBufferReadOnly {
    /// Memory-mapped region
    mmap: Mmap,
    /// Header
    header: SharedBufferHeader,
    /// Path
    path: PathBuf,
}

impl SharedTensorBufferReadOnly {
    /// Get tensor metadata
    pub fn tensor_metadata(&self) -> Result<Vec<SharedTensorInfo>, SharedMemoryError> {
        let header_size = std::mem::size_of::<SharedBufferHeader>();

        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&self.mmap[header_size..header_size + 8]);
        let metadata_len = u64::from_le_bytes(len_bytes) as usize;

        let metadata_bytes = &self.mmap[header_size + 8..header_size + 8 + metadata_len];
        let tensors: Vec<SharedTensorInfo> = serde_json::from_slice(metadata_bytes)?;

        Ok(tensors)
    }

    /// Get data slice for a tensor
    pub fn tensor_data(&self, info: &SharedTensorInfo) -> &[u8] {
        let start = self.header.data_offset as usize + info.offset;
        let end = start + info.size;
        &self.mmap[start..end]
    }

    /// Read tensor data as typed Vec (safe copy)
    pub fn read_tensor<T: Copy + Default>(&self, info: &SharedTensorInfo) -> Vec<T> {
        let bytes = self.tensor_data(info);
        let elem_size = std::mem::size_of::<T>();
        let count = bytes.len() / elem_size;
        let mut result = vec![T::default(); count];

        // Safe copy using byte manipulation
        let result_bytes = unsafe {
            std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, count * elem_size)
        };
        result_bytes.copy_from_slice(&bytes[..count * elem_size]);
        result
    }

    /// Verify checksum
    pub fn verify_checksum(&self) -> bool {
        let data_start = self.header.data_offset as usize;
        let data = &self.mmap[data_start..];

        let computed: u64 = data.iter().fold(0u64, |acc, &b| acc.wrapping_add(b as u64));
        computed == self.header.checksum
    }

    /// Get path
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Shared memory pool for managing multiple buffers
#[allow(dead_code)]
pub struct SharedMemoryPool {
    /// Base directory for shared memory files
    base_dir: PathBuf,
    /// Active buffers
    buffers: HashMap<String, Arc<SharedTensorBufferReadOnly>>,
    /// Maximum total size
    max_size: usize,
    /// Current total size
    current_size: AtomicU64,
}

impl SharedMemoryPool {
    /// Create a new pool
    pub fn new<P: AsRef<Path>>(base_dir: P, max_size: usize) -> Self {
        std::fs::create_dir_all(base_dir.as_ref()).ok();

        Self {
            base_dir: base_dir.as_ref().to_path_buf(),
            buffers: HashMap::new(),
            max_size,
            current_size: AtomicU64::new(0),
        }
    }

    /// Register a buffer in the pool
    pub fn register(
        &mut self,
        name: &str,
        buffer: SharedTensorBufferReadOnly,
    ) -> Result<(), SharedMemoryError> {
        let size = buffer.mmap.len();

        // Check size limit
        let current = self.current_size.load(Ordering::Relaxed);
        if current + size as u64 > self.max_size as u64 {
            return Err(SharedMemoryError::PoolFull);
        }

        self.current_size.fetch_add(size as u64, Ordering::Relaxed);
        self.buffers.insert(name.to_string(), Arc::new(buffer));

        Ok(())
    }

    /// Get a buffer by name
    pub fn get(&self, name: &str) -> Option<Arc<SharedTensorBufferReadOnly>> {
        self.buffers.get(name).cloned()
    }

    /// Remove a buffer
    pub fn remove(&mut self, name: &str) -> Option<Arc<SharedTensorBufferReadOnly>> {
        if let Some(buffer) = self.buffers.remove(name) {
            let size = buffer.mmap.len() as u64;
            self.current_size.fetch_sub(size, Ordering::Relaxed);
            Some(buffer)
        } else {
            None
        }
    }

    /// List all buffer names
    pub fn list(&self) -> Vec<&str> {
        self.buffers.keys().map(|s| s.as_str()).collect()
    }

    /// Get current memory usage
    pub fn memory_usage(&self) -> usize {
        self.current_size.load(Ordering::Relaxed) as usize
    }

    /// Get available memory
    pub fn available(&self) -> usize {
        self.max_size.saturating_sub(self.memory_usage())
    }
}

/// Error types for shared memory operations
#[derive(Debug)]
pub enum SharedMemoryError {
    /// IO error
    Io(std::io::Error),
    /// Invalid format
    InvalidFormat(String),
    /// JSON serialization error
    Json(serde_json::Error),
    /// Pool is full
    PoolFull,
    /// Buffer not found
    NotFound(String),
}

impl From<std::io::Error> for SharedMemoryError {
    fn from(err: std::io::Error) -> Self {
        SharedMemoryError::Io(err)
    }
}

impl From<serde_json::Error> for SharedMemoryError {
    fn from(err: serde_json::Error) -> Self {
        SharedMemoryError::Json(err)
    }
}

impl std::fmt::Display for SharedMemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SharedMemoryError::Io(e) => write!(f, "IO error: {}", e),
            SharedMemoryError::InvalidFormat(s) => write!(f, "Invalid format: {}", s),
            SharedMemoryError::Json(e) => write!(f, "JSON error: {}", e),
            SharedMemoryError::PoolFull => write!(f, "Shared memory pool is full"),
            SharedMemoryError::NotFound(s) => write!(f, "Buffer not found: {}", s),
        }
    }
}

impl std::error::Error for SharedMemoryError {}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_shared_buffer_create_and_read() {
        let dir = tempdir().expect("test: should succeed");
        let path = dir.path().join("test.shm");

        // Define tensors
        let tensors = vec![
            SharedTensorInfo {
                name: "weights".to_string(),
                dtype: TensorDtype::Float32,
                shape: vec![2, 3],
                offset: 0,
                size: 24, // 6 * 4 bytes
            },
            SharedTensorInfo {
                name: "bias".to_string(),
                dtype: TensorDtype::Float32,
                shape: vec![3],
                offset: 24,
                size: 12, // 3 * 4 bytes
            },
        ];

        // Create buffer
        let mut buffer =
            SharedTensorBuffer::create(&path, 36, &tensors).expect("test: should succeed");

        // Write data
        let weights: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bias: Vec<f32> = vec![0.1, 0.2, 0.3];

        buffer.write_tensor(&tensors[0], &weights);
        buffer.write_tensor(&tensors[1], &bias);
        buffer.update_checksum();
        buffer.flush().expect("test: should succeed");

        // Read back
        let read_buffer = SharedTensorBuffer::open_readonly(&path).expect("test: should succeed");
        let metadata = read_buffer.tensor_metadata().expect("test: should succeed");

        assert_eq!(metadata.len(), 2);
        assert_eq!(metadata[0].name, "weights");
        assert_eq!(metadata[1].name, "bias");

        let read_weights: Vec<f32> = read_buffer.read_tensor(&metadata[0]);
        let read_bias: Vec<f32> = read_buffer.read_tensor(&metadata[1]);

        assert_eq!(read_weights, weights);
        assert_eq!(read_bias, bias);
    }

    #[test]
    fn test_memory_pool() {
        let dir = tempdir().expect("test: should succeed");
        let pool_dir = dir.path().join("pool");

        let mut pool = SharedMemoryPool::new(&pool_dir, 1024 * 1024);

        // Create a buffer
        let path = pool_dir.join("test1.shm");
        let tensors = vec![SharedTensorInfo {
            name: "test".to_string(),
            dtype: TensorDtype::Float32,
            shape: vec![4],
            offset: 0,
            size: 16,
        }];

        SharedTensorBuffer::create(&path, 16, &tensors).expect("test: should succeed");

        // Register in pool
        let buffer = SharedTensorBuffer::open_readonly(&path).expect("test: should succeed");
        pool.register("test1", buffer)
            .expect("test: should succeed");

        assert_eq!(pool.list().len(), 1);
        assert!(pool.get("test1").is_some());

        // Remove
        pool.remove("test1");
        assert!(pool.get("test1").is_none());
    }
}
