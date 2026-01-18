//! Zero-copy buffer management
//!
//! Provides efficient buffer management with zero-copy semantics using Bytes.

use bytes::Bytes;

/// Zero-copy buffer for efficient data transfer
///
/// Uses `Bytes` internally which provides reference-counted, immutable byte buffers
/// that can be sliced without copying.
#[derive(Debug, Clone)]
pub struct ZeroCopyBuffer {
    data: Bytes,
}

impl ZeroCopyBuffer {
    /// Create a new zero-copy buffer from Bytes
    pub fn new(data: Bytes) -> Self {
        Self { data }
    }

    /// Create from a vector (takes ownership)
    pub fn from_vec(vec: Vec<u8>) -> Self {
        Self {
            data: Bytes::from(vec),
        }
    }

    /// Create from a static byte slice
    pub fn from_static(bytes: &'static [u8]) -> Self {
        Self {
            data: Bytes::from_static(bytes),
        }
    }

    /// Get a reference to the underlying data
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get the length of the buffer
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Create a zero-copy slice of this buffer
    ///
    /// This operation is very efficient as it only increments a reference count
    /// and doesn't copy any data.
    pub fn slice(&self, range: std::ops::Range<usize>) -> Self {
        Self {
            data: self.data.slice(range),
        }
    }

    /// Get the underlying Bytes
    pub fn into_bytes(self) -> Bytes {
        self.data
    }

    /// Get a reference to the underlying Bytes
    pub fn bytes(&self) -> &Bytes {
        &self.data
    }

    /// Split the buffer at an index
    ///
    /// Returns two zero-copy buffers: [0, at) and [at, len).
    /// This is a zero-copy operation.
    pub fn split_at(&self, at: usize) -> (Self, Self) {
        let left = self.data.slice(0..at);
        let right = self.data.slice(at..);
        (Self { data: left }, Self { data: right })
    }

    /// Split off the first n bytes
    pub fn split_to(&self, at: usize) -> Self {
        Self {
            data: self.data.slice(0..at),
        }
    }

    /// Split off the last n bytes
    pub fn split_from(&self, at: usize) -> Self {
        Self {
            data: self.data.slice(at..),
        }
    }
}

impl From<Vec<u8>> for ZeroCopyBuffer {
    fn from(vec: Vec<u8>) -> Self {
        Self::from_vec(vec)
    }
}

impl From<Bytes> for ZeroCopyBuffer {
    fn from(bytes: Bytes) -> Self {
        Self::new(bytes)
    }
}

impl From<&'static [u8]> for ZeroCopyBuffer {
    fn from(bytes: &'static [u8]) -> Self {
        Self::from_static(bytes)
    }
}

impl AsRef<[u8]> for ZeroCopyBuffer {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_buffer_basic() {
        let data = vec![1, 2, 3, 4, 5];
        let buf = ZeroCopyBuffer::from_vec(data);

        assert_eq!(buf.len(), 5);
        assert!(!buf.is_empty());
        assert_eq!(buf.as_slice(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_zero_copy_slice() {
        let buf = ZeroCopyBuffer::from_vec(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let slice = buf.slice(2..7);

        assert_eq!(slice.len(), 5);
        assert_eq!(slice.as_slice(), &[2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_zero_copy_split_at() {
        let buf = ZeroCopyBuffer::from_vec(vec![1, 2, 3, 4, 5]);
        let (left, right) = buf.split_at(3);

        assert_eq!(left.as_slice(), &[1, 2, 3]);
        assert_eq!(right.as_slice(), &[4, 5]);
    }

    #[test]
    fn test_zero_copy_split_to() {
        let buf = ZeroCopyBuffer::from_vec(vec![1, 2, 3, 4, 5]);
        let first = buf.split_to(3);

        assert_eq!(first.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_zero_copy_split_from() {
        let buf = ZeroCopyBuffer::from_vec(vec![1, 2, 3, 4, 5]);
        let last = buf.split_from(3);

        assert_eq!(last.as_slice(), &[4, 5]);
    }

    #[test]
    fn test_zero_copy_from_static() {
        let buf = ZeroCopyBuffer::from_static(b"hello world");

        assert_eq!(buf.len(), 11);
        assert_eq!(buf.as_slice(), b"hello world");
    }

    #[test]
    fn test_zero_copy_empty() {
        let buf = ZeroCopyBuffer::from_vec(vec![]);

        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }
}
