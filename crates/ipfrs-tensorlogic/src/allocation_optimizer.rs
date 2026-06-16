//! Allocation Optimization Utilities
//!
//! This module provides utilities for reducing heap allocations in conversion code,
//! including buffer pooling and stack-based allocation helpers.

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AllocationError {
    #[error("Buffer pool exhausted")]
    PoolExhausted,

    #[error("Invalid buffer size: {0}")]
    InvalidSize(usize),

    #[error("Buffer too small: required {required}, available {available}")]
    BufferTooSmall { required: usize, available: usize },
}

/// Buffer pool for reusable byte buffers
pub struct BufferPool {
    buffers: Arc<Mutex<VecDeque<Vec<u8>>>>,
    buffer_size: usize,
    max_buffers: usize,
}

impl BufferPool {
    /// Create a new buffer pool
    pub fn new(buffer_size: usize, max_buffers: usize) -> Self {
        Self {
            buffers: Arc::new(Mutex::new(VecDeque::new())),
            buffer_size,
            max_buffers,
        }
    }

    /// Acquire a buffer from the pool
    pub fn acquire(&self) -> PooledBuffer {
        let mut buffers = self.buffers.lock();

        let buffer = if let Some(mut buf) = buffers.pop_front() {
            buf.clear();
            buf.reserve(self.buffer_size);
            buf
        } else {
            Vec::with_capacity(self.buffer_size)
        };

        PooledBuffer {
            buffer,
            pool: Arc::clone(&self.buffers),
            max_buffers: self.max_buffers,
        }
    }

    /// Get current pool size
    pub fn size(&self) -> usize {
        self.buffers.lock().len()
    }

    /// Get buffer capacity
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }
}

/// RAII guard for pooled buffer
pub struct PooledBuffer {
    buffer: Vec<u8>,
    pool: Arc<Mutex<VecDeque<Vec<u8>>>>,
    max_buffers: usize,
}

impl PooledBuffer {
    /// Get mutable reference to the buffer
    #[allow(clippy::should_implement_trait)]
    pub fn as_mut(&mut self) -> &mut Vec<u8> {
        &mut self.buffer
    }

    /// Get reference to the buffer
    #[allow(clippy::should_implement_trait)]
    pub fn as_ref(&self) -> &[u8] {
        &self.buffer
    }

    /// Get the buffer length
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        let mut pool = self.pool.lock();
        if pool.len() < self.max_buffers {
            // Return buffer to pool
            let buffer = std::mem::take(&mut self.buffer);
            pool.push_back(buffer);
        }
    }
}

impl std::ops::Deref for PooledBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl std::ops::DerefMut for PooledBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
    }
}

/// Typed buffer pool for specific types
pub struct TypedBufferPool<T> {
    buffers: Arc<Mutex<VecDeque<Vec<T>>>>,
    buffer_capacity: usize,
    max_buffers: usize,
}

impl<T> TypedBufferPool<T> {
    /// Create a new typed buffer pool
    pub fn new(buffer_capacity: usize, max_buffers: usize) -> Self {
        Self {
            buffers: Arc::new(Mutex::new(VecDeque::new())),
            buffer_capacity,
            max_buffers,
        }
    }

    /// Acquire a buffer from the pool
    pub fn acquire(&self) -> TypedPooledBuffer<T> {
        let mut buffers = self.buffers.lock();

        let buffer = if let Some(mut buf) = buffers.pop_front() {
            buf.clear();
            buf.reserve(self.buffer_capacity);
            buf
        } else {
            Vec::with_capacity(self.buffer_capacity)
        };

        TypedPooledBuffer {
            buffer,
            pool: Arc::clone(&self.buffers),
            max_buffers: self.max_buffers,
        }
    }

    /// Get current pool size
    pub fn size(&self) -> usize {
        self.buffers.lock().len()
    }
}

/// RAII guard for typed pooled buffer
pub struct TypedPooledBuffer<T> {
    buffer: Vec<T>,
    pool: Arc<Mutex<VecDeque<Vec<T>>>>,
    max_buffers: usize,
}

impl<T> TypedPooledBuffer<T> {
    /// Get mutable reference to the buffer
    #[allow(clippy::should_implement_trait)]
    pub fn as_mut(&mut self) -> &mut Vec<T> {
        &mut self.buffer
    }

    /// Get reference to the buffer
    #[allow(clippy::should_implement_trait)]
    pub fn as_ref(&self) -> &[T] {
        &self.buffer
    }

    /// Get the buffer length
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Push an element to the buffer
    pub fn push(&mut self, value: T) {
        self.buffer.push(value);
    }

    /// Extend buffer with iterator
    pub fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.buffer.extend(iter);
    }
}

impl<T> Drop for TypedPooledBuffer<T> {
    fn drop(&mut self) {
        let mut pool = self.pool.lock();
        if pool.len() < self.max_buffers {
            // Return buffer to pool
            let buffer = std::mem::take(&mut self.buffer);
            pool.push_back(buffer);
        }
    }
}

impl<T> std::ops::Deref for TypedPooledBuffer<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl<T> std::ops::DerefMut for TypedPooledBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
    }
}

/// Zero-copy conversion utilities
pub struct ZeroCopyConverter;

impl ZeroCopyConverter {
    /// Convert slice to different representation without copying
    #[inline]
    pub fn cast_slice<T, U>(slice: &[T]) -> &[U]
    where
        T: bytemuck::Pod,
        U: bytemuck::Pod,
    {
        bytemuck::cast_slice(slice)
    }

    /// Convert mutable slice to different representation without copying
    #[inline]
    pub fn cast_slice_mut<T, U>(slice: &mut [T]) -> &mut [U]
    where
        T: bytemuck::Pod,
        U: bytemuck::Pod,
    {
        bytemuck::cast_slice_mut(slice)
    }

    /// Convert bytes to typed slice
    #[inline]
    pub fn bytes_to_slice<T: bytemuck::Pod>(bytes: &[u8]) -> &[T] {
        bytemuck::cast_slice(bytes)
    }

    /// Convert typed slice to bytes
    #[inline]
    pub fn slice_to_bytes<T: bytemuck::Pod>(slice: &[T]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
}

/// Stack-based buffer for small allocations
pub struct StackBuffer<const N: usize> {
    data: [u8; N],
    len: usize,
}

impl<const N: usize> StackBuffer<N> {
    /// Create a new stack buffer
    #[inline]
    pub const fn new() -> Self {
        Self {
            data: [0u8; N],
            len: 0,
        }
    }

    /// Get capacity
    #[inline]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Get length
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get remaining capacity
    #[inline]
    pub const fn remaining(&self) -> usize {
        N - self.len
    }

    /// Write bytes to buffer
    #[inline]
    pub fn write(&mut self, bytes: &[u8]) -> Result<(), AllocationError> {
        if bytes.len() > self.remaining() {
            return Err(AllocationError::BufferTooSmall {
                required: bytes.len(),
                available: self.remaining(),
            });
        }

        self.data[self.len..self.len + bytes.len()].copy_from_slice(bytes);
        self.len += bytes.len();
        Ok(())
    }

    /// Get slice of written data
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.data[..self.len]
    }

    /// Clear buffer
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }
}

impl<const N: usize> Default for StackBuffer<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Conversion buffer that uses stack for small sizes, heap for large
pub enum AdaptiveBuffer {
    Stack(Box<StackBuffer<256>>),
    Heap(Vec<u8>),
}

impl AdaptiveBuffer {
    /// Create a new adaptive buffer
    #[inline]
    pub fn new(size_hint: usize) -> Self {
        if size_hint <= 256 {
            Self::Stack(Box::default())
        } else {
            Self::Heap(Vec::with_capacity(size_hint))
        }
    }

    /// Write bytes to buffer
    pub fn write(&mut self, bytes: &[u8]) -> Result<(), AllocationError> {
        match self {
            Self::Stack(buf) => {
                if buf.remaining() >= bytes.len() {
                    buf.write(bytes)
                } else {
                    // Upgrade to heap
                    let mut heap = Vec::with_capacity(buf.len() + bytes.len());
                    heap.extend_from_slice(buf.as_slice());
                    heap.extend_from_slice(bytes);
                    *self = Self::Heap(heap);
                    Ok(())
                }
            }
            Self::Heap(vec) => {
                vec.extend_from_slice(bytes);
                Ok(())
            }
        }
    }

    /// Get slice of written data
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::Stack(buf) => buf.as_slice(),
            Self::Heap(vec) => vec.as_slice(),
        }
    }

    /// Get length
    pub fn len(&self) -> usize {
        match self {
            Self::Stack(buf) => buf.len(),
            Self::Heap(vec) => vec.len(),
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_pool() {
        let pool = BufferPool::new(1024, 4);

        let mut buffer1 = pool.acquire();
        buffer1.as_mut().extend_from_slice(&[1, 2, 3]);
        assert_eq!(buffer1.len(), 3);

        drop(buffer1);

        // Buffer should be returned to pool
        assert_eq!(pool.size(), 1);

        let buffer2 = pool.acquire();
        assert_eq!(buffer2.len(), 0); // Should be cleared
    }

    #[test]
    fn test_typed_buffer_pool() {
        let pool = TypedBufferPool::<f32>::new(100, 4);

        let mut buffer1 = pool.acquire();
        buffer1.push(1.0);
        buffer1.push(2.0);
        assert_eq!(buffer1.len(), 2);

        drop(buffer1);

        // Buffer should be returned to pool
        assert_eq!(pool.size(), 1);

        let buffer2 = pool.acquire();
        assert_eq!(buffer2.len(), 0); // Should be cleared
    }

    #[test]
    fn test_zero_copy_converter() {
        let floats: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes = ZeroCopyConverter::slice_to_bytes(&floats);
        assert_eq!(bytes.len(), 16); // 4 floats * 4 bytes

        let floats_back: &[f32] = ZeroCopyConverter::bytes_to_slice(bytes);
        assert_eq!(floats_back, &floats);
    }

    #[test]
    fn test_stack_buffer() {
        let mut buf = StackBuffer::<64>::new();
        assert_eq!(buf.capacity(), 64);
        assert_eq!(buf.len(), 0);

        buf.write(&[1, 2, 3]).expect("test: should succeed");
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.as_slice(), &[1, 2, 3]);

        buf.clear();
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_stack_buffer_overflow() {
        let mut buf = StackBuffer::<4>::new();
        assert!(buf.write(&[1, 2, 3, 4]).is_ok());
        assert!(buf.write(&[5]).is_err()); // Should fail
    }

    #[test]
    fn test_adaptive_buffer_small() {
        let mut buf = AdaptiveBuffer::new(10);
        buf.write(&[1, 2, 3]).expect("test: should succeed");

        assert!(matches!(buf, AdaptiveBuffer::Stack(_)));
        assert_eq!(buf.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_adaptive_buffer_large() {
        let mut buf = AdaptiveBuffer::new(512);
        buf.write(&[1, 2, 3]).expect("test: should succeed");

        assert!(matches!(buf, AdaptiveBuffer::Heap(_)));
        assert_eq!(buf.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_adaptive_buffer_upgrade() {
        let mut buf = AdaptiveBuffer::new(10);
        buf.write(&[1; 100]).expect("test: should succeed"); // Small hint but large write
        buf.write(&[2; 200]).expect("test: should succeed"); // Continue writing

        // Should have upgraded to heap
        assert!(matches!(buf, AdaptiveBuffer::Heap(_)));
        assert_eq!(buf.len(), 300);
    }

    #[test]
    fn test_pooled_buffer_deref() {
        let pool = BufferPool::new(1024, 4);
        let mut buffer = pool.acquire();

        buffer.as_mut().extend_from_slice(&[1, 2, 3, 4]);

        // Test deref
        assert_eq!(buffer[0], 1);
        assert_eq!(buffer[3], 4);
    }
}
