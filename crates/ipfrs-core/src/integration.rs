//! Integration utilities combining multiple ipfrs-core features.
//!
//! This module provides high-level utilities that combine tensor operations,
//! Arrow integration, and content-addressed storage for common workflows.

use crate::arrow::{arrow_to_tensor_block, TensorBlockArrowExt};
use crate::error::Result;
use crate::hash::global_hash_registry;
use crate::tensor::{TensorBlock, TensorShape};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Schema;
use multihash_codetable::Code;
use std::sync::Arc;

/// Batch processor for tensor blocks
pub struct TensorBatchProcessor {
    hash_algo: Code,
}

impl TensorBatchProcessor {
    /// Create a new batch processor with the specified hash algorithm
    pub fn new(hash_algo: Code) -> Self {
        Self { hash_algo }
    }

    /// Process multiple tensors and generate CIDs with hardware acceleration
    pub fn process_batch(&self, tensors: &[TensorBlock]) -> Result<Vec<String>> {
        let registry = global_hash_registry();
        let mut cids = Vec::with_capacity(tensors.len());

        for tensor in tensors {
            let data = tensor.data();
            let _hash = registry.digest(self.hash_algo, data)?;
            cids.push(tensor.cid().to_string());
        }

        Ok(cids)
    }

    /// Convert multiple tensors to an Arrow RecordBatch
    pub fn to_arrow_batch(&self, tensors: Vec<(&str, &TensorBlock)>) -> Result<RecordBatch> {
        let mut fields = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        for (name, tensor) in tensors {
            fields.push(tensor.to_arrow_field(name));
            arrays.push(tensor.to_arrow_array()?);
        }

        let schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, arrays).map_err(|e| {
            crate::error::Error::InvalidInput(format!("Failed to create RecordBatch: {}", e))
        })
    }

    /// Process Arrow RecordBatch and convert to tensor blocks
    pub fn from_arrow_batch(
        &self,
        batch: &RecordBatch,
        shapes: Vec<TensorShape>,
    ) -> Result<Vec<TensorBlock>> {
        if batch.num_columns() != shapes.len() {
            return Err(crate::error::Error::InvalidInput(format!(
                "Column count {} doesn't match shape count {}",
                batch.num_columns(),
                shapes.len()
            )));
        }

        let mut tensors = Vec::with_capacity(batch.num_columns());

        for (col_idx, shape) in shapes.into_iter().enumerate() {
            let array = batch.column(col_idx);
            let tensor = arrow_to_tensor_block(array.as_ref(), shape)?;
            tensors.push(tensor);
        }

        Ok(tensors)
    }
}

impl Default for TensorBatchProcessor {
    fn default() -> Self {
        Self {
            hash_algo: Code::Sha2_256,
        }
    }
}

/// Utility for tensor deduplication using content-addressed storage
pub struct TensorDeduplicator {
    seen_cids: std::collections::HashMap<String, usize>,
}

impl TensorDeduplicator {
    /// Create a new deduplicator
    pub fn new() -> Self {
        Self {
            seen_cids: std::collections::HashMap::new(),
        }
    }

    /// Check if a tensor has been seen before (by CID)
    /// Returns the index of the first occurrence if found
    pub fn check(&mut self, tensor: &TensorBlock) -> Option<usize> {
        let cid = tensor.cid().to_string();
        self.seen_cids.get(&cid).copied()
    }

    /// Register a tensor and return its index
    pub fn register(&mut self, tensor: &TensorBlock) -> usize {
        let cid = tensor.cid().to_string();
        let idx = self.seen_cids.len();
        self.seen_cids.entry(cid).or_insert(idx);
        idx
    }

    /// Get the number of unique tensors
    pub fn unique_count(&self) -> usize {
        self.seen_cids.len()
    }

    /// Get deduplication statistics
    pub fn stats(&self) -> DeduplicationStats {
        DeduplicationStats {
            unique_tensors: self.seen_cids.len(),
            total_checked: self.seen_cids.len(),
        }
    }
}

impl Default for TensorDeduplicator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for tensor deduplication
#[derive(Debug, Clone)]
pub struct DeduplicationStats {
    /// Number of unique tensors seen (distinct CIDs)
    pub unique_tensors: usize,
    /// Total number of tensors checked for deduplication
    pub total_checked: usize,
}

impl DeduplicationStats {
    /// Calculate the deduplication ratio
    pub fn dedup_ratio(&self) -> f64 {
        if self.total_checked == 0 {
            return 0.0;
        }
        self.unique_tensors as f64 / self.total_checked as f64
    }
}

/// High-level API for tensor storage and retrieval
pub struct TensorStore {
    tensors: std::collections::HashMap<String, TensorBlock>,
}

impl TensorStore {
    /// Create a new tensor store
    pub fn new() -> Self {
        Self {
            tensors: std::collections::HashMap::new(),
        }
    }

    /// Store a tensor and return its CID
    pub fn store(&mut self, tensor: TensorBlock) -> String {
        let cid = tensor.cid().to_string();
        self.tensors.insert(cid.clone(), tensor);
        cid
    }

    /// Retrieve a tensor by CID
    pub fn get(&self, cid: &str) -> Option<&TensorBlock> {
        self.tensors.get(cid)
    }

    /// Check if a tensor exists
    pub fn contains(&self, cid: &str) -> bool {
        self.tensors.contains_key(cid)
    }

    /// Get the number of stored tensors
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// List all CIDs in the store
    pub fn list_cids(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    /// Clear all tensors from the store
    pub fn clear(&mut self) {
        self.tensors.clear();
    }
}

impl Default for TensorStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_processor() {
        let processor = TensorBatchProcessor::default();

        // Create test tensors
        let data1 = vec![1.0f32, 2.0, 3.0, 4.0];
        let data2 = vec![5.0f32, 6.0, 7.0, 8.0];

        let tensor1 = TensorBlock::from_f32_slice(&data1, TensorShape::new(vec![2, 2])).unwrap();
        let tensor2 = TensorBlock::from_f32_slice(&data2, TensorShape::new(vec![2, 2])).unwrap();

        let cids = processor.process_batch(&[tensor1, tensor2]).unwrap();
        assert_eq!(cids.len(), 2);
        assert_ne!(cids[0], cids[1]); // Different data should have different CIDs
    }

    #[test]
    fn test_arrow_batch_roundtrip() {
        let processor = TensorBatchProcessor::default();

        // Create test tensors
        let data1 = vec![1.0f32, 2.0, 3.0, 4.0];
        let data2 = vec![5.0f32, 6.0, 7.0, 8.0];

        let tensor1 = TensorBlock::from_f32_slice(&data1, TensorShape::new(vec![4])).unwrap();
        let tensor2 = TensorBlock::from_f32_slice(&data2, TensorShape::new(vec![4])).unwrap();

        // Convert to Arrow batch
        let batch = processor
            .to_arrow_batch(vec![("t1", &tensor1), ("t2", &tensor2)])
            .unwrap();

        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 4);

        // Convert back to tensors
        let shapes = vec![TensorShape::new(vec![4]), TensorShape::new(vec![4])];
        let recovered = processor.from_arrow_batch(&batch, shapes).unwrap();

        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].to_f32_vec().unwrap(), data1);
        assert_eq!(recovered[1].to_f32_vec().unwrap(), data2);
    }

    #[test]
    fn test_tensor_deduplicator() {
        let mut dedup = TensorDeduplicator::new();

        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor1 = TensorBlock::from_f32_slice(&data, TensorShape::new(vec![4])).unwrap();
        let tensor2 = TensorBlock::from_f32_slice(&data, TensorShape::new(vec![4])).unwrap(); // Same data

        // First tensor should be new
        assert_eq!(dedup.check(&tensor1), None);
        let idx1 = dedup.register(&tensor1);

        // Second tensor should be duplicate
        assert_eq!(dedup.check(&tensor2), Some(idx1));

        assert_eq!(dedup.unique_count(), 1);
    }

    #[test]
    fn test_tensor_store() {
        let mut store = TensorStore::new();
        assert!(store.is_empty());

        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = TensorBlock::from_f32_slice(&data, TensorShape::new(vec![4])).unwrap();

        // Store tensor
        let cid = store.store(tensor.clone());
        assert_eq!(store.len(), 1);
        assert!(store.contains(&cid));

        // Retrieve tensor
        let retrieved = store.get(&cid).unwrap();
        assert_eq!(retrieved.to_f32_vec().unwrap(), data);

        // List CIDs
        let cids = store.list_cids();
        assert_eq!(cids.len(), 1);
        assert_eq!(cids[0], cid);

        // Clear store
        store.clear();
        assert!(store.is_empty());
    }

    #[test]
    fn test_deduplication_stats() {
        let mut dedup = TensorDeduplicator::new();

        let data1 = vec![1.0f32, 2.0];
        let data2 = vec![3.0f32, 4.0];

        let t1 = TensorBlock::from_f32_slice(&data1, TensorShape::new(vec![2])).unwrap();
        let t2 = TensorBlock::from_f32_slice(&data2, TensorShape::new(vec![2])).unwrap();
        let t3 = TensorBlock::from_f32_slice(&data1, TensorShape::new(vec![2])).unwrap(); // Duplicate of t1

        dedup.register(&t1);
        dedup.register(&t2);
        let _ = dedup.check(&t3);

        let stats = dedup.stats();
        assert_eq!(stats.unique_tensors, 2);
    }
}
