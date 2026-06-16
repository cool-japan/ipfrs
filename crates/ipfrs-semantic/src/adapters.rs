//! Vector database adapters for external integration.
//!
//! This module provides a unified interface for working with different vector database
//! backends, including IPFRS-native indices and external systems like Qdrant, Milvus,
//! Pinecone, and Weaviate.
//!
//! # Architecture
//!
//! The adapter layer provides:
//! - A common `VectorBackend` trait for all implementations
//! - Type-safe operations for indexing and search
//! - Migration utilities between different backends
//! - Batch operation support for efficiency
//!
//! # Basic Usage
//!
//! ```
//! use ipfrs_semantic::adapters::{VectorBackend, IpfrsBackend, BackendConfig};
//! use ipfrs_core::Cid;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create an IPFRS-native backend with custom dimension
//! let config = BackendConfig {
//!     dimension: 4,
//!     ..Default::default()
//! };
//! let mut backend = IpfrsBackend::new(config)?;
//!
//! // Insert vectors
//! let cid: Cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi".parse()?;
//! let embedding = vec![0.1, 0.2, 0.3, 0.4];
//! backend.insert(cid, &embedding, None)?;
//!
//! // Search for similar vectors
//! let query = vec![0.15, 0.25, 0.35, 0.45];
//! let results = backend.search(&query, 10, None)?;
//!
//! println!("Found {} results", results.len());
//! # Ok(())
//! # }
//! ```
//!
//! # Implementing Custom Backends
//!
//! To integrate with external vector databases, implement the `VectorBackend` trait:
//!
//! ```ignore
//! use ipfrs_semantic::adapters::*;
//! use ipfrs_core::{Cid, Result};
//!
//! struct MyCustomBackend {
//!     // Your backend state (e.g., connection pool, client)
//!     dimension: usize,
//! }
//!
//! impl VectorBackend for MyCustomBackend {
//!     fn insert(&mut self, cid: Cid, vector: &[f32], metadata: Option<Metadata>) -> Result<()> {
//!         // Validate dimension
//!         if vector.len() != self.dimension {
//!             return Err(ipfrs_core::Error::InvalidInput(
//!                 format!("Expected {} dimensions, got {}", self.dimension, vector.len())
//!             ));
//!         }
//!
//!         // Insert into your backend
//!         // Example: self.client.insert(cid.to_string(), vector, metadata)?;
//!         Ok(())
//!     }
//!
//!     fn search(
//!         &mut self,
//!         query: &[f32],
//!         k: usize,
//!         filter: Option<&MetadataFilter>,
//!     ) -> Result<Vec<BackendSearchResult>> {
//!         // Perform search in your backend
//!         // Example: let results = self.client.search(query, k, filter)?;
//!         Ok(vec![])
//!     }
//!
//!     fn delete(&mut self, cid: &Cid) -> Result<()> {
//!         // Delete from your backend
//!         // Example: self.client.delete(cid.to_string())?;
//!         Ok(())
//!     }
//!
//!     fn get(&self, cid: &Cid) -> Result<Option<(Vec<f32>, Option<Metadata>)>> {
//!         // Retrieve from your backend
//!         // Example: self.client.get(cid.to_string())
//!         Ok(None)
//!     }
//!
//!     fn count(&self) -> Result<usize> {
//!         // Return total count from your backend
//!         // Example: self.client.count()
//!         Ok(0)
//!     }
//!
//!     fn clear(&mut self) -> Result<()> {
//!         // Clear all data from your backend
//!         // Example: self.client.clear()
//!         Ok(())
//!     }
//!
//!     fn stats(&self) -> BackendStats {
//!         BackendStats::default()
//!     }
//! }
//! ```
//!
//! # Migration Between Backends
//!
//! The module provides utilities to migrate data between different backends:
//!
//! ```ignore
//! use ipfrs_semantic::adapters::*;
//!
//! // Migrate specific CIDs from one backend to another
//! let cids = vec![/* ... */];
//! let stats = migrate_vectors(&mut source_backend, &mut dest_backend, &cids)?;
//! println!("Migrated {} vectors, {} not found", stats.migrated, stats.not_found);
//! ```

use async_trait::async_trait;
use ipfrs_core::{Cid, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::hnsw::{DistanceMetric, VectorIndex};
use crate::metadata::{Metadata, MetadataFilter};

/// Configuration for vector database backends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    /// Vector dimension
    pub dimension: usize,
    /// Distance metric to use
    pub metric: DistanceMetric,
    /// Backend-specific parameters
    pub params: HashMap<String, String>,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            dimension: 768,
            metric: DistanceMetric::Cosine,
            params: HashMap::new(),
        }
    }
}

/// Search result from a vector backend
#[derive(Debug, Clone)]
pub struct BackendSearchResult {
    /// Content identifier
    pub cid: Cid,
    /// Distance/similarity score
    pub score: f32,
    /// Optional metadata
    pub metadata: Option<Metadata>,
}

/// Common interface for vector database backends
#[async_trait]
pub trait VectorBackend: Send + Sync {
    /// Insert a single vector with optional metadata
    fn insert(&mut self, cid: Cid, vector: &[f32], metadata: Option<Metadata>) -> Result<()>;

    /// Insert multiple vectors in batch
    fn insert_batch(&mut self, items: &[(Cid, Vec<f32>, Option<Metadata>)]) -> Result<()> {
        for (cid, vector, metadata) in items {
            self.insert(*cid, vector, metadata.clone())?;
        }
        Ok(())
    }

    /// Search for k nearest neighbors
    fn search(
        &mut self,
        query: &[f32],
        k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<BackendSearchResult>>;

    /// Search with multiple queries in batch
    fn search_batch(
        &mut self,
        queries: &[Vec<f32>],
        k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<Vec<BackendSearchResult>>> {
        let mut results = Vec::new();
        for query in queries {
            results.push(self.search(query, k, filter)?);
        }
        Ok(results)
    }

    /// Delete a vector by CID
    fn delete(&mut self, cid: &Cid) -> Result<()>;

    /// Update vector for existing CID
    fn update(&mut self, cid: &Cid, vector: &[f32], metadata: Option<Metadata>) -> Result<()> {
        self.delete(cid)?;
        self.insert(*cid, vector, metadata)
    }

    /// Get vector by CID
    fn get(&self, cid: &Cid) -> Result<Option<(Vec<f32>, Option<Metadata>)>>;

    /// Count total vectors in the backend
    fn count(&self) -> Result<usize>;

    /// Clear all vectors
    fn clear(&mut self) -> Result<()>;

    /// Get backend name/type
    fn backend_name(&self) -> &str;

    /// Get backend statistics
    fn stats(&self) -> BackendStats;
}

/// Statistics for a vector backend
#[derive(Debug, Clone, Default)]
pub struct BackendStats {
    /// Total number of vectors
    pub vector_count: usize,
    /// Total searches performed
    pub searches: usize,
    /// Total insertions performed
    pub insertions: usize,
    /// Backend-specific metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// IPFRS-native backend using HNSW index
pub struct IpfrsBackend {
    /// HNSW vector index
    index: VectorIndex,
    /// Vector storage for retrieval
    vector_store: HashMap<Cid, Vec<f32>>,
    /// Metadata storage
    metadata_store: HashMap<Cid, Metadata>,
    /// Configuration
    config: BackendConfig,
    /// Statistics
    stats: BackendStats,
}

impl IpfrsBackend {
    /// Create a new IPFRS backend
    pub fn new(config: BackendConfig) -> Result<Self> {
        let index = VectorIndex::new(
            config.dimension,
            config.metric,
            16,  // max_connections
            200, // ef_construction
        )?;

        Ok(Self {
            index,
            vector_store: HashMap::new(),
            metadata_store: HashMap::new(),
            config,
            stats: BackendStats::default(),
        })
    }

    /// Get the underlying HNSW index (for advanced usage)
    pub fn index(&self) -> &VectorIndex {
        &self.index
    }

    /// Get mutable reference to the underlying HNSW index
    pub fn index_mut(&mut self) -> &mut VectorIndex {
        &mut self.index
    }
}

#[async_trait]
impl VectorBackend for IpfrsBackend {
    fn insert(&mut self, cid: Cid, vector: &[f32], metadata: Option<Metadata>) -> Result<()> {
        self.index.insert(&cid, vector)?;
        self.vector_store.insert(cid, vector.to_vec());
        if let Some(meta) = metadata {
            self.metadata_store.insert(cid, meta);
        }
        self.stats.insertions += 1;
        self.stats.vector_count = self.index.len();
        Ok(())
    }

    fn insert_batch(&mut self, items: &[(Cid, Vec<f32>, Option<Metadata>)]) -> Result<()> {
        for (cid, vector, metadata) in items {
            self.index.insert(cid, vector)?;
            self.vector_store.insert(*cid, vector.clone());
            if let Some(meta) = metadata {
                self.metadata_store.insert(*cid, meta.clone());
            }
            self.stats.insertions += 1;
        }
        self.stats.vector_count = self.index.len();
        Ok(())
    }

    fn search(
        &mut self,
        query: &[f32],
        k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<BackendSearchResult>> {
        let ef_search = 50; // Default ef_search parameter
        let raw_results = self.index.search(query, k * 2, ef_search)?; // Get more results for filtering
        self.stats.searches += 1;

        let mut results = Vec::new();
        for result in raw_results {
            // Apply metadata filter if provided
            if let Some(filter) = filter {
                if let Some(metadata) = self.metadata_store.get(&result.cid) {
                    if !filter.matches(metadata) {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            results.push(BackendSearchResult {
                cid: result.cid,
                score: result.score,
                metadata: self.metadata_store.get(&result.cid).cloned(),
            });

            if results.len() >= k {
                break;
            }
        }

        Ok(results)
    }

    fn delete(&mut self, cid: &Cid) -> Result<()> {
        self.index.delete(cid)?;
        self.vector_store.remove(cid);
        self.metadata_store.remove(cid);
        self.stats.vector_count = self.index.len();
        Ok(())
    }

    fn get(&self, cid: &Cid) -> Result<Option<(Vec<f32>, Option<Metadata>)>> {
        if let Some(vector) = self.vector_store.get(cid) {
            let metadata = self.metadata_store.get(cid).cloned();
            Ok(Some((vector.clone(), metadata)))
        } else {
            Ok(None)
        }
    }

    fn count(&self) -> Result<usize> {
        Ok(self.index.len())
    }

    fn clear(&mut self) -> Result<()> {
        // VectorIndex doesn't have a clear method, so we need to recreate it
        self.index = VectorIndex::new(self.config.dimension, self.config.metric, 16, 200)?;
        self.vector_store.clear();
        self.metadata_store.clear();
        self.stats = BackendStats::default();
        Ok(())
    }

    fn backend_name(&self) -> &str {
        "ipfrs-hnsw"
    }

    fn stats(&self) -> BackendStats {
        self.stats.clone()
    }
}

/// Migration utilities for moving data between backends
pub struct BackendMigration;

impl BackendMigration {
    /// Migrate all data from source to destination backend
    #[allow(dead_code)]
    pub fn migrate(
        _source: &dyn VectorBackend,
        _dest: &mut dyn VectorBackend,
    ) -> Result<MigrationStats> {
        let stats = MigrationStats::default();

        // This is a simplified migration - real implementation would need
        // a way to iterate over all vectors in the source backend
        // For now, this serves as the interface structure

        Ok(stats)
    }

    /// Migrate specific CIDs from source to destination
    pub fn migrate_cids(
        source: &dyn VectorBackend,
        dest: &mut dyn VectorBackend,
        cids: &[Cid],
    ) -> Result<MigrationStats> {
        Self::migrate_cids_with_progress(source, dest, cids, |_, _| {})
    }

    /// Migrate specific CIDs with progress tracking
    ///
    /// The progress callback receives (current_index, total_count) for each processed CID
    ///
    /// # Example
    ///
    /// ```ignore
    /// use ipfrs_semantic::adapters::BackendMigration;
    ///
    /// let stats = BackendMigration::migrate_cids_with_progress(
    ///     &source,
    ///     &mut dest,
    ///     &cids,
    ///     |current, total| {
    ///         println!("Progress: {}/{} ({:.1}%)", current, total, (current as f64 / total as f64) * 100.0);
    ///     }
    /// )?;
    /// ```
    pub fn migrate_cids_with_progress<F>(
        source: &dyn VectorBackend,
        dest: &mut dyn VectorBackend,
        cids: &[Cid],
        mut progress_callback: F,
    ) -> Result<MigrationStats>
    where
        F: FnMut(usize, usize),
    {
        let mut stats = MigrationStats::default();
        let total = cids.len();

        for (index, cid) in cids.iter().enumerate() {
            if let Some((vector, metadata)) = source.get(cid)? {
                dest.insert(*cid, &vector, metadata)?;
                stats.migrated += 1;
            } else {
                stats.not_found += 1;
            }

            // Report progress
            progress_callback(index + 1, total);
        }

        Ok(stats)
    }

    /// Export vectors to a portable format
    pub fn export_to_json(backend: &dyn VectorBackend, cids: &[Cid]) -> Result<String> {
        let mut exports = Vec::new();

        for cid in cids {
            if let Some((vector, metadata)) = backend.get(cid)? {
                let export = ExportedVector {
                    cid: cid.to_string(),
                    vector,
                    metadata,
                };
                exports.push(export);
            }
        }

        serde_json::to_string_pretty(&exports)
            .map_err(|e| Error::Serialization(format!("JSON export failed: {}", e)))
    }

    /// Import vectors from JSON
    pub fn import_from_json(backend: &mut dyn VectorBackend, json: &str) -> Result<usize> {
        let exports: Vec<ExportedVector> = serde_json::from_str(json)
            .map_err(|e| Error::Serialization(format!("JSON import failed: {}", e)))?;

        let mut count = 0;
        for export in exports {
            let cid: Cid = export
                .cid
                .parse()
                .map_err(|e| Error::InvalidInput(format!("Invalid CID: {}", e)))?;
            backend.insert(cid, &export.vector, export.metadata)?;
            count += 1;
        }

        Ok(count)
    }
}

/// Statistics from a migration operation
#[derive(Debug, Clone, Default)]
pub struct MigrationStats {
    /// Number of vectors successfully migrated
    pub migrated: usize,
    /// Number of vectors not found
    pub not_found: usize,
    /// Number of errors encountered
    pub errors: usize,
}

/// Exported vector format for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExportedVector {
    cid: String,
    vector: Vec<f32>,
    metadata: Option<Metadata>,
}

/// Backend registry for managing multiple backends
pub struct BackendRegistry {
    backends: HashMap<String, Box<dyn VectorBackend>>,
    default_backend: Option<String>,
}

impl BackendRegistry {
    /// Create a new backend registry
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
            default_backend: None,
        }
    }

    /// Register a backend with a name
    pub fn register(&mut self, name: String, backend: Box<dyn VectorBackend>) {
        if self.default_backend.is_none() {
            self.default_backend = Some(name.clone());
        }
        self.backends.insert(name, backend);
    }

    /// Get a backend by name
    pub fn get(&self, name: &str) -> Option<&dyn VectorBackend> {
        self.backends.get(name).map(|b| b.as_ref())
    }

    /// Get a mutable backend by name
    pub fn get_mut(&mut self, name: &str) -> Option<&mut (dyn VectorBackend + '_)> {
        match self.backends.get_mut(name) {
            Some(backend) => Some(backend.as_mut()),
            None => None,
        }
    }

    /// Get the default backend
    pub fn get_default(&self) -> Option<&dyn VectorBackend> {
        self.default_backend
            .as_ref()
            .and_then(|name| self.get(name))
    }

    /// Get the default backend mutably
    pub fn get_default_mut(&mut self) -> Option<&mut (dyn VectorBackend + '_)> {
        if let Some(name) = self.default_backend.clone() {
            self.get_mut(&name)
        } else {
            None
        }
    }

    /// Set the default backend
    pub fn set_default(&mut self, name: String) -> Result<()> {
        if self.backends.contains_key(&name) {
            self.default_backend = Some(name);
            Ok(())
        } else {
            Err(Error::NotFound(format!("Backend '{}' not found", name)))
        }
    }

    /// List all registered backend names
    pub fn list_backends(&self) -> Vec<String> {
        self.backends.keys().cloned().collect()
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipfrs_backend_creation() {
        let config = BackendConfig::default();
        let backend = IpfrsBackend::new(config);
        assert!(backend.is_ok());
    }

    #[test]
    fn test_insert_and_search() {
        let config = BackendConfig {
            dimension: 4,
            ..Default::default()
        };
        let mut backend = IpfrsBackend::new(config)
            .expect("test: backend creation with valid config should succeed");

        let cid = Cid::default();
        let vector = vec![1.0, 2.0, 3.0, 4.0];
        backend
            .insert(cid, &vector, None)
            .expect("test: insert with valid vector should succeed");

        let query = vec![1.1, 2.1, 3.1, 4.1];
        let results = backend
            .search(&query, 1, None)
            .expect("test: search should succeed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].cid, cid);
    }

    #[test]
    fn test_insert_with_metadata() {
        use crate::metadata::MetadataValue;

        let config = BackendConfig {
            dimension: 3,
            ..Default::default()
        };
        let mut backend = IpfrsBackend::new(config)
            .expect("test: backend creation with valid config should succeed");

        let cid = Cid::default();
        let vector = vec![1.0, 2.0, 3.0];
        let mut metadata = Metadata::new();
        metadata.set("key", MetadataValue::String("value".to_string()));

        backend
            .insert(cid, &vector, Some(metadata))
            .expect("test: insert with metadata should succeed");

        let retrieved = backend
            .get(&cid)
            .expect("test: get after insert should return Some");
        assert!(retrieved.is_some());
        let (_, meta) = retrieved.expect("test: retrieved value should be Some after insert");
        assert!(meta.is_some());
    }

    #[test]
    fn test_batch_insert() {
        use multihash_codetable::{Code, MultihashDigest};

        let config = BackendConfig {
            dimension: 2,
            ..Default::default()
        };
        let mut backend = IpfrsBackend::new(config)
            .expect("test: backend creation with valid config should succeed");

        // Create unique CIDs for each item
        let cid1 = Cid::new_v1(0x55, Code::Sha2_256.digest(b"test_batch_1"));
        let cid2 = Cid::new_v1(0x55, Code::Sha2_256.digest(b"test_batch_2"));
        let cid3 = Cid::new_v1(0x55, Code::Sha2_256.digest(b"test_batch_3"));

        let items = vec![
            (cid1, vec![1.0, 2.0], None),
            (cid2, vec![3.0, 4.0], None),
            (cid3, vec![5.0, 6.0], None),
        ];

        backend
            .insert_batch(&items)
            .expect("test: batch insert should succeed");
        assert_eq!(backend.count().expect("test: count should succeed"), 3);
    }

    #[test]
    fn test_delete() {
        let config = BackendConfig {
            dimension: 2,
            ..Default::default()
        };
        let mut backend = IpfrsBackend::new(config)
            .expect("test: backend creation with valid config should succeed");

        let cid = Cid::default();
        let vector = vec![1.0, 2.0];
        backend
            .insert(cid, &vector, None)
            .expect("test: insert with valid vector should succeed");

        assert_eq!(backend.count().expect("test: count should succeed"), 1);

        backend.delete(&cid).expect("test: delete should succeed");
        assert_eq!(backend.count().expect("test: count should succeed"), 0);
    }

    #[test]
    fn test_update() {
        let config = BackendConfig {
            dimension: 2,
            ..Default::default()
        };
        let mut backend = IpfrsBackend::new(config)
            .expect("test: backend creation with valid config should succeed");

        let cid = Cid::default();
        let vector1 = vec![1.0, 2.0];
        backend
            .insert(cid, &vector1, None)
            .expect("test: insert with valid vector should succeed");

        let vector2 = vec![3.0, 4.0];
        backend
            .update(&cid, &vector2, None)
            .expect("test: update should succeed");

        let retrieved = backend
            .get(&cid)
            .expect("test: get after update should return Some")
            .expect("test: retrieved option should be Some");
        assert_eq!(retrieved.0, vector2);
    }

    #[test]
    fn test_clear() {
        use multihash_codetable::{Code, MultihashDigest};

        let config = BackendConfig {
            dimension: 2,
            ..Default::default()
        };
        let mut backend = IpfrsBackend::new(config)
            .expect("test: backend creation with valid config should succeed");

        // Create unique CIDs for each item
        let cid1 = Cid::new_v1(0x55, Code::Sha2_256.digest(b"test_clear_1"));
        let cid2 = Cid::new_v1(0x55, Code::Sha2_256.digest(b"test_clear_2"));

        backend
            .insert(cid1, &[1.0, 2.0], None)
            .expect("test: first insert should succeed");
        backend
            .insert(cid2, &[3.0, 4.0], None)
            .expect("test: second insert should succeed");

        assert_eq!(backend.count().expect("test: count should succeed"), 2);

        backend.clear().expect("test: clear should succeed");
        assert_eq!(backend.count().expect("test: count should succeed"), 0);
    }

    #[test]
    fn test_stats() {
        let config = BackendConfig {
            dimension: 2,
            ..Default::default()
        };
        let mut backend = IpfrsBackend::new(config)
            .expect("test: backend creation with valid config should succeed");

        backend
            .insert(Cid::default(), &[1.0, 2.0], None)
            .expect("test: insert should succeed");
        backend
            .search(&[1.0, 2.0], 1, None)
            .expect("test: search should succeed");

        let stats = backend.stats();
        assert_eq!(stats.insertions, 1);
        assert_eq!(stats.searches, 1);
    }

    #[test]
    fn test_backend_registry() {
        let mut registry = BackendRegistry::new();

        let config = BackendConfig {
            dimension: 2,
            ..Default::default()
        };
        let backend = IpfrsBackend::new(config).expect("test: backend creation should succeed");

        registry.register("test".to_string(), Box::new(backend));

        assert!(registry.get("test").is_some());
        assert_eq!(registry.list_backends().len(), 1);
    }

    #[test]
    fn test_migration_stats() {
        let stats = MigrationStats::default();
        assert_eq!(stats.migrated, 0);
        assert_eq!(stats.not_found, 0);
        assert_eq!(stats.errors, 0);
    }

    #[test]
    fn test_export_import() {
        let config = BackendConfig {
            dimension: 3,
            ..Default::default()
        };
        let mut backend =
            IpfrsBackend::new(config.clone()).expect("test: backend creation should succeed");

        let cid = Cid::default();
        let vector = vec![1.0, 2.0, 3.0];
        backend
            .insert(cid, &vector, None)
            .expect("test: insert should succeed");

        // Export
        let json = BackendMigration::export_to_json(&backend, &[cid])
            .expect("test: export to JSON should succeed");
        assert!(!json.is_empty());

        // Import to new backend
        let mut backend2 =
            IpfrsBackend::new(config).expect("test: second backend creation should succeed");
        let count = BackendMigration::import_from_json(&mut backend2, &json)
            .expect("test: import from JSON should succeed");
        assert_eq!(count, 1);
        assert_eq!(
            backend2
                .count()
                .expect("test: count after import should succeed"),
            1
        );
    }
}
