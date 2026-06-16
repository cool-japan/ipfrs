//! Partial block requests (range queries)
//!
//! Support for requesting byte ranges from blocks, useful for:
//! - Partial tensor loading
//! - Sparse block access
//! - Efficient streaming of large blocks

use ipfrs_core::Cid;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::ops::Range;
use thiserror::Error;

/// Serialize CID as string
fn serialize_cid<S>(cid: &Cid, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(&cid.to_string())
}

/// Deserialize CID from string
fn deserialize_cid<'de, D>(deserializer: D) -> Result<Cid, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    s.parse().map_err(serde::de::Error::custom)
}

/// Error types for range requests
#[derive(Error, Debug)]
pub enum RangeError {
    #[error("Invalid range: {0}")]
    InvalidRange(String),
    #[error("Range out of bounds: requested {requested}, available {available}")]
    OutOfBounds { requested: u64, available: u64 },
    #[error("Block not found: {0}")]
    BlockNotFound(Cid),
    #[error("Unsatisfiable range")]
    Unsatisfiable,
}

/// Byte range specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ByteRange {
    /// Request bytes from offset to end
    FromTo { start: u64, end: u64 },
    /// Request bytes from offset to end of block
    From(u64),
    /// Request last N bytes
    Suffix(u64),
    /// Request entire block
    All,
}

impl ByteRange {
    /// Create a range from start to end (inclusive)
    pub fn from_to(start: u64, end: u64) -> Result<Self, RangeError> {
        if start > end {
            return Err(RangeError::InvalidRange(format!(
                "start ({}) > end ({})",
                start, end
            )));
        }
        Ok(ByteRange::FromTo { start, end })
    }

    /// Create a range from offset to end
    pub fn from(start: u64) -> Self {
        ByteRange::From(start)
    }

    /// Create a range for the last N bytes
    pub fn suffix(count: u64) -> Self {
        ByteRange::Suffix(count)
    }

    /// Convert to concrete byte range given total size
    pub fn to_range(&self, total_size: u64) -> Result<Range<u64>, RangeError> {
        match self {
            ByteRange::FromTo { start, end } => {
                if *end >= total_size {
                    return Err(RangeError::OutOfBounds {
                        requested: *end,
                        available: total_size,
                    });
                }
                Ok(*start..*end + 1)
            }
            ByteRange::From(start) => {
                if *start >= total_size {
                    return Err(RangeError::OutOfBounds {
                        requested: *start,
                        available: total_size,
                    });
                }
                Ok(*start..total_size)
            }
            ByteRange::Suffix(count) => {
                if *count > total_size {
                    Ok(0..total_size)
                } else {
                    Ok(total_size - count..total_size)
                }
            }
            ByteRange::All => Ok(0..total_size),
        }
    }

    /// Check if this range overlaps with another
    pub fn overlaps(&self, other: &ByteRange, total_size: u64) -> bool {
        if let (Ok(r1), Ok(r2)) = (self.to_range(total_size), other.to_range(total_size)) {
            r1.start < r2.end && r2.start < r1.end
        } else {
            false
        }
    }

    /// Merge two ranges if they overlap or are adjacent
    pub fn merge(&self, other: &ByteRange, total_size: u64) -> Option<ByteRange> {
        if let (Ok(r1), Ok(r2)) = (self.to_range(total_size), other.to_range(total_size)) {
            if r1.start <= r2.end && r2.start <= r1.end {
                let start = r1.start.min(r2.start);
                let end = (r1.end - 1).max(r2.end - 1);
                Some(ByteRange::FromTo { start, end })
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Get the size of this range
    pub fn size(&self, total_size: u64) -> u64 {
        self.to_range(total_size)
            .map(|r| r.end - r.start)
            .unwrap_or(0)
    }
}

/// Range request for a specific block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeRequest {
    /// CID of the block
    #[serde(serialize_with = "serialize_cid", deserialize_with = "deserialize_cid")]
    pub cid: Cid,
    /// Byte range to request
    pub range: ByteRange,
    /// Priority (higher = more important)
    pub priority: i32,
}

impl RangeRequest {
    /// Create a new range request
    pub fn new(cid: Cid, range: ByteRange) -> Self {
        Self {
            cid,
            range,
            priority: 0,
        }
    }

    /// Create a range request with priority
    pub fn with_priority(cid: Cid, range: ByteRange, priority: i32) -> Self {
        Self {
            cid,
            range,
            priority,
        }
    }
}

/// Range response containing partial block data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeResponse {
    /// CID of the block
    #[serde(serialize_with = "serialize_cid", deserialize_with = "deserialize_cid")]
    pub cid: Cid,
    /// Byte range of this response
    pub range: Range<u64>,
    /// Partial block data
    pub data: Vec<u8>,
    /// Total size of the complete block
    pub total_size: u64,
}

impl RangeResponse {
    /// Create a new range response
    pub fn new(cid: Cid, range: Range<u64>, data: Vec<u8>, total_size: u64) -> Self {
        Self {
            cid,
            range,
            data,
            total_size,
        }
    }

    /// Check if this response satisfies a request
    pub fn satisfies(&self, request: &RangeRequest) -> bool {
        if self.cid != request.cid {
            return false;
        }
        if let Ok(req_range) = request.range.to_range(self.total_size) {
            self.range.start <= req_range.start && self.range.end >= req_range.end
        } else {
            false
        }
    }

    /// Extract data for a specific sub-range
    pub fn extract_range(&self, range: &Range<u64>) -> Result<Vec<u8>, RangeError> {
        if range.start < self.range.start || range.end > self.range.end {
            return Err(RangeError::OutOfBounds {
                requested: range.end,
                available: self.range.end,
            });
        }

        let offset = (range.start - self.range.start) as usize;
        let len = (range.end - range.start) as usize;

        if offset + len > self.data.len() {
            return Err(RangeError::OutOfBounds {
                requested: (offset + len) as u64,
                available: self.data.len() as u64,
            });
        }

        Ok(self.data[offset..offset + len].to_vec())
    }
}

/// Manager for assembling partial blocks from range responses
pub struct RangeAssembler {
    /// CID of the block being assembled
    cid: Cid,
    /// Total size of the block
    total_size: u64,
    /// Received ranges and their data
    received: Vec<(Range<u64>, Vec<u8>)>,
}

impl RangeAssembler {
    /// Create a new range assembler
    pub fn new(cid: Cid, total_size: u64) -> Self {
        Self {
            cid,
            total_size,
            received: Vec::new(),
        }
    }

    /// Add a range response
    pub fn add_range(&mut self, response: RangeResponse) -> Result<(), RangeError> {
        if response.cid != self.cid {
            return Err(RangeError::InvalidRange("CID mismatch".to_string()));
        }

        if response.total_size != self.total_size {
            return Err(RangeError::InvalidRange("Total size mismatch".to_string()));
        }

        self.received.push((response.range, response.data));
        Ok(())
    }

    /// Check if the block is complete
    pub fn is_complete(&self) -> bool {
        let mut covered = vec![false; self.total_size as usize];

        for (range, _) in &self.received {
            for i in range.start..range.end {
                if (i as usize) < covered.len() {
                    covered[i as usize] = true;
                }
            }
        }

        covered.iter().all(|&x| x)
    }

    /// Get missing ranges
    pub fn missing_ranges(&self) -> Vec<Range<u64>> {
        let mut covered = vec![false; self.total_size as usize];

        for (range, _) in &self.received {
            for i in range.start..range.end {
                if (i as usize) < covered.len() {
                    covered[i as usize] = true;
                }
            }
        }

        let mut missing = Vec::new();
        let mut start = None;

        for (i, &is_covered) in covered.iter().enumerate() {
            if !is_covered && start.is_none() {
                start = Some(i as u64);
            } else if is_covered && start.is_some() {
                missing.push(start.expect("just checked start.is_some()")..i as u64);
                start = None;
            }
        }

        if let Some(s) = start {
            missing.push(s..self.total_size);
        }

        missing
    }

    /// Assemble the complete block
    pub fn assemble(&self) -> Result<Vec<u8>, RangeError> {
        if !self.is_complete() {
            return Err(RangeError::InvalidRange("Block incomplete".to_string()));
        }

        let mut data = vec![0u8; self.total_size as usize];

        for (range, chunk) in &self.received {
            let start = range.start as usize;
            let end = range.end as usize;
            let len = end - start;

            if chunk.len() != len {
                return Err(RangeError::InvalidRange("Chunk size mismatch".to_string()));
            }

            data[start..end].copy_from_slice(chunk);
        }

        Ok(data)
    }

    /// Get completion percentage
    pub fn completion_percentage(&self) -> f64 {
        let mut covered = vec![false; self.total_size as usize];

        for (range, _) in &self.received {
            for i in range.start..range.end {
                if (i as usize) < covered.len() {
                    covered[i as usize] = true;
                }
            }
        }

        let covered_count = covered.iter().filter(|&&x| x).count();
        (covered_count as f64 / self.total_size as f64) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cid() -> Cid {
        "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse()
            .expect("test: valid CID string")
    }

    #[test]
    fn test_byte_range_from_to() {
        let range = ByteRange::from_to(0, 99).expect("test: create byte range");
        assert_eq!(
            range.to_range(1000).expect("test: convert to range"),
            0..100
        );
    }

    #[test]
    fn test_byte_range_from() {
        let range = ByteRange::from(500);
        assert_eq!(
            range.to_range(1000).expect("test: convert to range"),
            500..1000
        );
    }

    #[test]
    fn test_byte_range_suffix() {
        let range = ByteRange::suffix(100);
        assert_eq!(
            range.to_range(1000).expect("test: convert to range"),
            900..1000
        );
    }

    #[test]
    fn test_byte_range_all() {
        let range = ByteRange::All;
        assert_eq!(
            range.to_range(1000).expect("test: convert to range"),
            0..1000
        );
    }

    #[test]
    fn test_byte_range_out_of_bounds() {
        let range = ByteRange::from_to(0, 1500).expect("test: create byte range");
        assert!(range.to_range(1000).is_err());
    }

    #[test]
    fn test_byte_range_invalid() {
        assert!(ByteRange::from_to(100, 50).is_err());
    }

    #[test]
    fn test_byte_range_overlaps() {
        let range1 = ByteRange::from_to(0, 99).expect("test: create byte range");
        let range2 = ByteRange::from_to(50, 149).expect("test: create byte range");
        assert!(range1.overlaps(&range2, 1000));

        let range3 = ByteRange::from_to(200, 299).expect("test: create byte range");
        assert!(!range1.overlaps(&range3, 1000));
    }

    #[test]
    fn test_byte_range_merge() {
        let range1 = ByteRange::from_to(0, 99).expect("test: create byte range");
        let range2 = ByteRange::from_to(50, 149).expect("test: create byte range");
        let merged = range1.merge(&range2, 1000).expect("test: merge ranges");
        assert_eq!(
            merged.to_range(1000).expect("test: convert to range"),
            0..150
        );
    }

    #[test]
    fn test_byte_range_size() {
        let range = ByteRange::from_to(100, 199).expect("test: create byte range");
        assert_eq!(range.size(1000), 100);
    }

    #[test]
    fn test_range_request() {
        let cid = test_cid();
        let range = ByteRange::from_to(0, 99).expect("test: create byte range");
        let req = RangeRequest::new(cid, range);
        assert_eq!(req.priority, 0);

        let req2 = RangeRequest::with_priority(cid, range, 10);
        assert_eq!(req2.priority, 10);
    }

    #[test]
    fn test_range_response_satisfies() {
        let cid = test_cid();
        let range = ByteRange::from_to(0, 99).expect("test: create byte range");
        let req = RangeRequest::new(cid, range);

        let response = RangeResponse::new(cid, 0..100, vec![0u8; 100], 1000);
        assert!(response.satisfies(&req));

        let response2 = RangeResponse::new(cid, 50..150, vec![0u8; 100], 1000);
        assert!(!response2.satisfies(&req));
    }

    #[test]
    fn test_range_response_extract() {
        let cid = test_cid();
        let data = (0..100).collect::<Vec<u8>>();
        let response = RangeResponse::new(cid, 0..100, data.clone(), 1000);

        let extracted = response
            .extract_range(&(10..20))
            .expect("test: extract range");
        assert_eq!(extracted, &data[10..20]);
    }

    #[test]
    fn test_range_assembler() {
        let cid = test_cid();
        let mut assembler = RangeAssembler::new(cid, 100);

        assert!(!assembler.is_complete());
        assert_eq!(assembler.completion_percentage(), 0.0);

        let resp1 = RangeResponse::new(cid, 0..50, vec![1u8; 50], 100);
        assembler
            .add_range(resp1)
            .expect("test: add range to assembler");
        assert_eq!(assembler.completion_percentage(), 50.0);

        let resp2 = RangeResponse::new(cid, 50..100, vec![2u8; 50], 100);
        assembler
            .add_range(resp2)
            .expect("test: add range to assembler");
        assert!(assembler.is_complete());
        assert_eq!(assembler.completion_percentage(), 100.0);

        let data = assembler.assemble().expect("test: assemble ranges");
        assert_eq!(data.len(), 100);
        assert_eq!(&data[0..50], &vec![1u8; 50][..]);
        assert_eq!(&data[50..100], &vec![2u8; 50][..]);
    }

    #[test]
    fn test_range_assembler_missing_ranges() {
        let cid = test_cid();
        let mut assembler = RangeAssembler::new(cid, 100);

        let resp1 = RangeResponse::new(cid, 0..25, vec![0u8; 25], 100);
        assembler
            .add_range(resp1)
            .expect("test: add range to assembler");

        let resp2 = RangeResponse::new(cid, 75..100, vec![0u8; 25], 100);
        assembler
            .add_range(resp2)
            .expect("test: add range to assembler");

        let missing = assembler.missing_ranges();
        assert_eq!(missing, vec![25..75]);
    }

    #[test]
    fn test_range_assembler_overlapping() {
        let cid = test_cid();
        let mut assembler = RangeAssembler::new(cid, 100);

        let resp1 = RangeResponse::new(cid, 0..60, vec![1u8; 60], 100);
        assembler
            .add_range(resp1)
            .expect("test: add range to assembler");

        let resp2 = RangeResponse::new(cid, 40..100, vec![2u8; 60], 100);
        assembler
            .add_range(resp2)
            .expect("test: add range to assembler");

        assert!(assembler.is_complete());
    }

    #[test]
    fn test_range_assembler_incomplete() {
        let cid = test_cid();
        let mut assembler = RangeAssembler::new(cid, 100);

        let resp = RangeResponse::new(cid, 0..50, vec![0u8; 50], 100);
        assembler
            .add_range(resp)
            .expect("test: add range to assembler");

        assert!(assembler.assemble().is_err());
    }
}
