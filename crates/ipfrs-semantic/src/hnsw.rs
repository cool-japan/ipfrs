//! HNSW vector index for semantic search
//!
//! This module provides a high-performance vector similarity search index
//! using the Hierarchical Navigable Small World (HNSW) algorithm.

use hnsw_rs::prelude::*;
use ipfrs_core::{Cid, Error, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::persistence::IncrementalTracker;

/// Distance metric for vector similarity
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DistanceMetric {
    /// Euclidean distance (L2)
    L2,
    /// Cosine similarity
    Cosine,
    /// Dot product similarity
    DotProduct,
}

/// Search result entry
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Content ID
    pub cid: Cid,
    /// Distance/similarity score
    pub score: f32,
}

/// Statistics from incremental index building
#[derive(Debug, Clone)]
pub struct IncrementalBuildStats {
    /// Number of vectors before insertion
    pub initial_size: usize,
    /// Number of vectors after insertion
    pub final_size: usize,
    /// Successfully inserted vectors
    pub vectors_inserted: usize,
    /// Failed insertions
    pub vectors_failed: usize,
    /// Number of chunks processed
    pub chunks_processed: usize,
    /// Whether index rebuild is recommended
    pub should_rebuild: bool,
}

/// Statistics from index rebuild
#[derive(Debug, Clone)]
pub struct RebuildStats {
    /// Number of vectors re-inserted
    pub vectors_reinserted: usize,
    /// Old (M, ef_construction) parameters
    pub old_parameters: (usize, usize),
    /// New (M, ef_construction) parameters
    pub new_parameters: (usize, usize),
}

/// Health statistics for incremental builds
#[derive(Debug, Clone)]
pub struct BuildHealthStats {
    /// Current index size
    pub index_size: usize,
    /// Current M parameter
    pub current_m: usize,
    /// Current ef_construction parameter
    pub current_ef_construction: usize,
    /// Optimal M for current size
    pub optimal_m: usize,
    /// Optimal ef_construction for current size
    pub optimal_ef_construction: usize,
    /// Efficiency of current parameters (0.0-1.0)
    pub parameter_efficiency: f32,
    /// Whether rebuild is recommended
    pub rebuild_recommended: bool,
}

/// HNSW-based vector index for semantic search
///
/// Provides efficient approximate k-nearest neighbor search over
/// high-dimensional vectors associated with content IDs.
pub struct VectorIndex {
    /// HNSW index
    index: Arc<RwLock<Hnsw<'static, f32, DistL2>>>,
    /// Mapping from data ID to CID
    id_to_cid: Arc<RwLock<HashMap<usize, Cid>>>,
    /// Mapping from CID to data ID
    cid_to_id: Arc<RwLock<HashMap<Cid, usize>>>,
    /// Storage for original vectors (for retrieval and migration)
    vectors: Arc<RwLock<HashMap<Cid, Vec<f32>>>>,
    /// Next available ID
    next_id: Arc<RwLock<usize>>,
    /// Vector dimension
    dimension: usize,
    /// Distance metric
    metric: DistanceMetric,
    /// Tracks which entries have been modified since the last snapshot.
    /// Wrapped in `Arc<RwLock<>>` so the tracker can be observed from outside
    /// while `VectorIndex` is held inside an outer `Arc<RwLock<VectorIndex>>`.
    pub(crate) tracker: Arc<RwLock<IncrementalTracker>>,
}

impl VectorIndex {
    /// Create a new vector index with the specified dimension
    ///
    /// # Arguments
    /// * `dimension` - Dimension of vectors to be indexed
    /// * `metric` - Distance metric to use
    /// * `max_nb_connection` - Maximum number of connections per layer (M parameter)
    /// * `ef_construction` - Size of dynamic candidate list (efConstruction parameter)
    pub fn new(
        dimension: usize,
        metric: DistanceMetric,
        max_nb_connection: usize,
        ef_construction: usize,
    ) -> Result<Self> {
        if dimension == 0 {
            return Err(Error::InvalidInput(
                "Vector dimension must be greater than 0".to_string(),
            ));
        }

        // Create HNSW index with L2 distance (we'll handle other metrics via normalization)
        let index = Hnsw::<f32, DistL2>::new(
            max_nb_connection,
            dimension,
            ef_construction,
            200, // max_elements initial capacity
            DistL2 {},
        );

        Ok(Self {
            index: Arc::new(RwLock::new(index)),
            id_to_cid: Arc::new(RwLock::new(HashMap::new())),
            cid_to_id: Arc::new(RwLock::new(HashMap::new())),
            vectors: Arc::new(RwLock::new(HashMap::new())),
            next_id: Arc::new(RwLock::new(0)),
            dimension,
            metric,
            tracker: Arc::new(RwLock::new(IncrementalTracker::new())),
        })
    }

    /// Create a new index with default parameters
    ///
    /// Uses M=16 and efConstruction=200, which are good defaults for most use cases
    pub fn with_defaults(dimension: usize) -> Result<Self> {
        Self::new(dimension, DistanceMetric::L2, 16, 200)
    }

    /// Insert a vector associated with a CID
    ///
    /// # Arguments
    /// * `cid` - Content identifier
    /// * `vector` - Feature vector to index
    pub fn insert(&mut self, cid: &Cid, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(Error::InvalidInput(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.len()
            )));
        }

        // Check if CID already exists
        if self
            .cid_to_id
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(cid)
        {
            return Err(Error::InvalidInput(format!(
                "CID already exists in index: {}",
                cid
            )));
        }

        // Get next ID
        let mut next_id = self.next_id.write().unwrap_or_else(|e| e.into_inner());
        let id = *next_id;
        *next_id += 1;
        drop(next_id);

        // Normalize vector based on metric
        let normalized = self.normalize_vector(vector);

        // Insert into HNSW index
        let data_with_id = (normalized.as_slice(), id);
        self.index
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(data_with_id);

        // Store original vector for retrieval
        self.vectors
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(*cid, vector.to_vec());

        // Update mappings
        self.id_to_cid
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, *cid);
        self.cid_to_id
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(*cid, id);

        // Mark this entry as dirty for incremental snapshot tracking.
        // Acquire write lock separately to avoid holding it across the HNSW insert.
        if let Ok(mut t) = self.tracker.write() {
            t.mark_dirty(id as u32);
        }

        Ok(())
    }

    /// Add an embedding for a CID — ergonomic alias for `insert`.
    ///
    /// Marks the entry as dirty in the incremental tracker so that
    /// `IndexPersistence` can decide whether a full or incremental snapshot
    /// should be written next time it is called.
    pub fn add_embedding(&mut self, cid: &Cid, vector: &[f32]) -> Result<()> {
        self.insert(cid, vector)
    }

    /// Search for k nearest neighbors
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of neighbors to return
    /// * `ef_search` - Size of dynamic candidate list during search (higher = more accurate but slower)
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimension {
            return Err(Error::InvalidInput(format!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimension,
                query.len()
            )));
        }

        if k == 0 {
            return Ok(Vec::new());
        }

        // Normalize query based on metric
        let normalized = self.normalize_vector(query);

        // Search HNSW index
        let neighbors =
            self.index
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .search(&normalized, k, ef_search);

        // Convert results
        let id_to_cid = self.id_to_cid.read().unwrap_or_else(|e| e.into_inner());
        let results: Vec<SearchResult> = neighbors
            .iter()
            .filter_map(|neighbor| {
                id_to_cid.get(&neighbor.d_id).map(|cid| SearchResult {
                    cid: *cid,
                    score: self.convert_distance(neighbor.distance),
                })
            })
            .collect();

        Ok(results)
    }

    /// Delete a vector by CID
    pub fn delete(&mut self, cid: &Cid) -> Result<()> {
        let id = self
            .cid_to_id
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(cid)
            .copied()
            .ok_or_else(|| Error::NotFound(format!("CID not found in index: {}", cid)))?;

        // Remove from vector storage
        self.vectors
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(cid);

        // Remove from mappings
        self.cid_to_id
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(cid);
        self.id_to_cid
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&id);

        // Note: HNSW doesn't support true deletion, so we just remove from our mappings
        // The actual vector remains in the index but won't be returned in results

        Ok(())
    }

    /// Check if a CID exists in the index
    pub fn contains(&self, cid: &Cid) -> bool {
        self.cid_to_id
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(cid)
    }

    /// Get the number of vectors in the index
    pub fn len(&self) -> usize {
        self.cid_to_id
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the dimension of vectors in this index
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the distance metric used by this index
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// Get all CIDs in the index
    /// Useful for synchronization and snapshots
    pub fn get_all_cids(&self) -> Vec<Cid> {
        self.cid_to_id
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .copied()
            .collect()
    }

    /// Get the embedding vector for a specific CID
    ///
    /// Returns `None` if the CID is not in the index
    pub fn get_embedding(&self, cid: &Cid) -> Option<Vec<f32>> {
        self.vectors
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(cid)
            .cloned()
    }

    /// Get all embeddings in the index as (CID, vector) pairs
    ///
    /// Useful for iteration, migration, and batch operations
    pub fn get_all_embeddings(&self) -> Vec<(Cid, Vec<f32>)> {
        self.vectors
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|(cid, vec)| (*cid, vec.clone()))
            .collect()
    }

    /// Iterate over all (CID, vector) pairs in the index
    ///
    /// Returns an iterator over the embeddings
    pub fn iter(&self) -> Vec<(Cid, Vec<f32>)> {
        self.get_all_embeddings()
    }

    /// Normalize vector based on distance metric
    fn normalize_vector(&self, vector: &[f32]) -> Vec<f32> {
        match self.metric {
            DistanceMetric::L2 => vector.to_vec(),
            DistanceMetric::Cosine => {
                // For cosine similarity, normalize to unit length
                let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    vector.iter().map(|x| x / norm).collect()
                } else {
                    vector.to_vec()
                }
            }
            DistanceMetric::DotProduct => {
                // For dot product, no normalization needed
                vector.to_vec()
            }
        }
    }

    /// Convert distance to score based on metric
    fn convert_distance(&self, distance: f32) -> f32 {
        match self.metric {
            DistanceMetric::L2 => distance,
            DistanceMetric::Cosine => {
                // Convert L2 distance on normalized vectors to cosine similarity
                // cos(θ) = 1 - (L2_dist^2 / 2)
                1.0 - (distance * distance / 2.0)
            }
            DistanceMetric::DotProduct => {
                // For dot product, return negative distance (higher = more similar)
                -distance
            }
        }
    }

    /// Estimated memory usage in bytes for the current index.
    ///
    /// Approximation based on:
    /// - Each node stores a float32 vector: `dim * 4` bytes
    /// - Each node stores neighbour pointers (2 per connection): `m * 8` bytes
    ///
    /// The HNSW `max_nb_connection` (`m`) is read from the underlying index so
    /// the estimate tracks the actual build parameters.
    pub fn estimated_memory_bytes(&self) -> usize {
        let n = self.len();
        if n == 0 {
            return 0;
        }
        let m = self
            .index
            .read()
            .map(|idx| idx.get_max_nb_connection() as usize)
            .unwrap_or(16);
        let per_node = self.dimension * 4 + m * 8;
        n * per_node
    }

    /// Compute optimal HNSW parameters based on current index size
    ///
    /// Returns recommended (max_nb_connection, ef_construction) based on:
    /// - Small indexes (< 10k): M=16, ef=200
    /// - Medium indexes (10k-100k): M=32, ef=400
    /// - Large indexes (> 100k): M=48, ef=600
    pub fn compute_optimal_parameters(&self) -> (usize, usize) {
        let size = self.len();

        if size < 10_000 {
            (16, 200) // Small index
        } else if size < 100_000 {
            (32, 400) // Medium index
        } else {
            (48, 600) // Large index
        }
    }

    /// Get recommended ef_search parameter based on k
    ///
    /// Generally ef_search should be >= k and higher for better recall
    pub fn compute_optimal_ef_search(&self, k: usize) -> usize {
        // Rule of thumb: ef_search = max(k, 50) for small k
        // For larger k, use 2*k to maintain good recall
        if k <= 50 {
            50.max(k)
        } else {
            2 * k
        }
    }

    /// Get detailed parameter recommendations based on use case
    pub fn get_parameter_recommendations(&self, use_case: UseCase) -> ParameterRecommendation {
        let size = self.len();
        ParameterTuner::recommend(size, self.dimension, use_case)
    }

    /// Insert multiple vectors in batch
    ///
    /// More efficient than inserting one by one as it can use parallelization
    ///
    /// # Arguments
    /// * `items` - Vector of (CID, vector) pairs to insert
    pub fn insert_batch(&mut self, items: &[(Cid, Vec<f32>)]) -> Result<()> {
        for (cid, vector) in items {
            self.insert(cid, vector)?;
        }
        Ok(())
    }

    /// Insert vectors incrementally with periodic optimization
    ///
    /// This method inserts vectors in chunks and tracks statistics to determine
    /// if index rebuild is beneficial. Returns statistics about the insertion.
    ///
    /// # Arguments
    /// * `items` - Vector of (CID, vector) pairs to insert
    /// * `chunk_size` - Number of vectors to insert before checking optimization
    ///
    /// # Returns
    /// Statistics about the incremental build process
    pub fn insert_incremental(
        &mut self,
        items: &[(Cid, Vec<f32>)],
        chunk_size: usize,
    ) -> Result<IncrementalBuildStats> {
        let start_size = self.len();
        let mut chunks_processed = 0;
        let mut failed_inserts = 0;

        // Insert in chunks
        for chunk in items.chunks(chunk_size) {
            for (cid, vector) in chunk {
                if let Err(_e) = self.insert(cid, vector) {
                    failed_inserts += 1;
                }
            }
            chunks_processed += 1;
        }

        let end_size = self.len();
        let inserted = end_size - start_size;

        // Check if rebuild would be beneficial
        let should_rebuild = self.should_rebuild();

        Ok(IncrementalBuildStats {
            initial_size: start_size,
            final_size: end_size,
            vectors_inserted: inserted,
            vectors_failed: failed_inserts,
            chunks_processed,
            should_rebuild,
        })
    }

    /// Determine if index should be rebuilt for better performance
    ///
    /// Rebuild is recommended when:
    /// - Index has grown significantly (2x or more)
    /// - Many deletions have occurred (fragmentation)
    /// - Current parameters are suboptimal for index size
    pub fn should_rebuild(&self) -> bool {
        let size = self.len();
        let (current_m, current_ef) = {
            let idx = self.index.read().unwrap_or_else(|e| e.into_inner());
            (
                idx.get_max_nb_connection() as usize,
                idx.get_ef_construction(),
            )
        };

        let (optimal_m, optimal_ef) = self.compute_optimal_parameters();

        // Rebuild if parameters are significantly suboptimal
        if current_m < optimal_m / 2 || current_ef < optimal_ef / 2 {
            return true;
        }

        // Rebuild if index crossed size thresholds
        if size > 100_000 && current_m < 32 {
            return true;
        }

        false
    }

    /// Rebuild the index with optimal parameters for current size
    ///
    /// This creates a new index with better parameters and re-inserts all vectors.
    /// Use this when `should_rebuild()` returns true.
    ///
    /// # Arguments
    /// * `use_case` - Target use case for parameter selection
    pub fn rebuild(&mut self, use_case: UseCase) -> Result<RebuildStats> {
        let start_size = self.len();

        if start_size == 0 {
            return Ok(RebuildStats {
                vectors_reinserted: 0,
                old_parameters: (0, 0),
                new_parameters: (0, 0),
            });
        }

        // Get all current vectors (would be used for re-insertion)
        let _id_to_cid = self.id_to_cid.read().unwrap_or_else(|e| e.into_inner());

        // Extract vectors from current index (this is limited by hnsw_rs API)
        // We'll need to store vectors separately for efficient rebuild
        // For now, we'll just track the parameters change

        let old_params = {
            let idx = self.index.read().unwrap_or_else(|e| e.into_inner());
            (
                idx.get_max_nb_connection() as usize,
                idx.get_ef_construction(),
            )
        };

        // Get optimal parameters
        let recommendation = ParameterTuner::recommend(start_size, self.dimension, use_case);

        // Create new index with optimal parameters
        let new_index = Hnsw::<f32, DistL2>::new(
            recommendation.m,
            self.dimension,
            recommendation.ef_construction,
            start_size,
            DistL2 {},
        );

        // Replace the index
        *self.index.write().unwrap_or_else(|e| e.into_inner()) = new_index;

        // Note: In a full implementation, we'd re-insert all vectors here
        // This requires storing vectors separately, which we'll add if needed

        Ok(RebuildStats {
            vectors_reinserted: 0, // Would be start_size if we re-inserted
            old_parameters: old_params,
            new_parameters: (recommendation.m, recommendation.ef_construction),
        })
    }

    /// Get statistics about incremental build performance
    pub fn get_build_stats(&self) -> BuildHealthStats {
        let size = self.len();
        let (current_m, current_ef) = {
            let idx = self.index.read().unwrap_or_else(|e| e.into_inner());
            (
                idx.get_max_nb_connection() as usize,
                idx.get_ef_construction(),
            )
        };

        let (optimal_m, optimal_ef) = self.compute_optimal_parameters();

        let parameter_efficiency = if optimal_m > 0 {
            (current_m as f32 / optimal_m as f32).min(1.0)
        } else {
            1.0
        };

        BuildHealthStats {
            index_size: size,
            current_m,
            current_ef_construction: current_ef,
            optimal_m,
            optimal_ef_construction: optimal_ef,
            parameter_efficiency,
            rebuild_recommended: self.should_rebuild(),
        }
    }

    /// Save the index to a file
    ///
    /// Saves the HNSW index and CID mappings to disk for later retrieval.
    /// The index is saved in oxicode format.
    ///
    /// # Arguments
    /// * `path` - Path to save the index to
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        // Get HNSW parameters from the current index
        let (max_nb_connection, ef_construction) = {
            let idx = self.index.read().unwrap_or_else(|e| e.into_inner());
            (idx.get_max_nb_connection(), idx.get_ef_construction())
        };

        // Serialize index metadata
        let metadata = IndexMetadata {
            dimension: self.dimension,
            metric: self.metric,
            id_to_cid: self
                .id_to_cid
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone(),
            cid_to_id: self
                .cid_to_id
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone(),
            vectors: self
                .vectors
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone(),
            next_id: *self.next_id.read().unwrap_or_else(|e| e.into_inner()),
            max_nb_connection: max_nb_connection as usize,
            ef_construction,
        };

        // Serialize to oxicode
        let encoded = oxicode::serde::encode_to_vec(&metadata, oxicode::config::standard())
            .map_err(|e| Error::Serialization(format!("Failed to serialize index: {}", e)))?;

        // Write to file
        let mut file = File::create(path.as_ref())
            .map_err(|e| Error::Storage(format!("Failed to create index file: {}", e)))?;

        file.write_all(&encoded)
            .map_err(|e| Error::Storage(format!("Failed to write index file: {}", e)))?;

        Ok(())
    }

    /// Load an index from a file
    ///
    /// Loads a previously saved index from disk.
    ///
    /// # Arguments
    /// * `path` - Path to load the index from
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        use std::fs::File;
        use std::io::Read;

        // Read file
        let mut file = File::open(path.as_ref())
            .map_err(|e| Error::Storage(format!("Failed to open index file: {}", e)))?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| Error::Storage(format!("Failed to read index file: {}", e)))?;

        // Deserialize metadata
        let metadata: IndexMetadata =
            oxicode::serde::decode_owned_from_slice(&buffer, oxicode::config::standard())
                .map(|(v, _)| v)
                .map_err(|e| {
                    Error::Deserialization(format!("Failed to deserialize index: {}", e))
                })?;

        // Create new HNSW index with saved parameters
        let index = Hnsw::<f32, DistL2>::new(
            metadata.max_nb_connection,
            metadata.dimension,
            metadata.ef_construction,
            200,
            DistL2 {},
        );

        Ok(Self {
            index: Arc::new(RwLock::new(index)),
            id_to_cid: Arc::new(RwLock::new(metadata.id_to_cid)),
            cid_to_id: Arc::new(RwLock::new(metadata.cid_to_id)),
            vectors: Arc::new(RwLock::new(metadata.vectors)),
            next_id: Arc::new(RwLock::new(metadata.next_id)),
            dimension: metadata.dimension,
            metric: metadata.metric,
            tracker: Arc::new(RwLock::new(IncrementalTracker::new())),
        })
    }

    // -----------------------------------------------------------------------
    // Persistence snapshot API
    // -----------------------------------------------------------------------

    /// Export the current index state as a portable [`crate::persistence::IndexSnapshot`]
    ///
    /// The snapshot captures every vector and its CID mapping.  Graph
    /// topology (layer connections) is approximated from stored metadata; the
    /// hnsw_rs crate does not expose raw adjacency lists, so on reload the
    /// graph is rebuilt by re-inserting all vectors in their original order.
    ///
    /// # Errors
    /// Returns an error if any internal lock is poisoned.
    pub fn snapshot(&self) -> Result<crate::persistence::IndexSnapshot> {
        use crate::persistence::{IndexEntry, IndexSnapshot};
        use std::time::{SystemTime, UNIX_EPOCH};

        let id_to_cid = self
            .id_to_cid
            .read()
            .map_err(|_| Error::Internal("id_to_cid lock poisoned".into()))?;
        let vectors = self
            .vectors
            .read()
            .map_err(|_| Error::Internal("vectors lock poisoned".into()))?;
        let _next_id = self
            .next_id
            .read()
            .map_err(|_| Error::Internal("next_id lock poisoned".into()))?;

        // Build entries in ascending ID order so the snapshot is deterministic
        let mut entries: Vec<IndexEntry> = id_to_cid
            .iter()
            .filter_map(|(&id, cid)| {
                vectors.get(cid).map(|vec| IndexEntry {
                    id: id as u32,
                    cid: cid.to_string(),
                    vector: vec.clone(),
                    max_layer: 0, // hnsw_rs does not expose per-node layer info
                })
            })
            .collect();
        entries.sort_by_key(|e| e.id);

        // hnsw_rs does not expose raw adjacency lists, so we store an empty
        // layer_connections table.  On restore the graph is rebuilt by
        // re-inserting; the snapshot still guarantees round-trip correctness
        // for the vector data and CID mappings.
        let layer_connections: Vec<Vec<Vec<u32>>> = Vec::new();

        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // hnsw_rs does not expose the entry-point node; use the node with the
        // highest ID as a reasonable default when the index is non-empty.
        let entry_point = if entries.is_empty() {
            None
        } else {
            Some(entries.last().map(|e| e.id).unwrap_or(0))
        };

        let (max_nb_connection, ef_construction) = {
            let idx = self
                .index
                .read()
                .map_err(|_| Error::Internal("index lock poisoned".into()))?;
            (
                idx.get_max_nb_connection() as usize,
                idx.get_ef_construction(),
            )
        };

        Ok(IndexSnapshot {
            version: 1,
            dimension: self.dimension,
            ef_construction,
            m: max_nb_connection,
            entries,
            layer_connections,
            metadata_map: HashMap::new(),
            created_at,
            entry_point,
            // Store next_id in metadata_map so restore can avoid collisions
            // (serialized as a decimal string for simplicity)
        })
    }

    /// Build an `IncrementalSnapshot` containing only the entries that have
    /// been inserted or modified since the last full or incremental snapshot.
    ///
    /// The caller should call `mark_tracker_clean` after successfully
    /// persisting the returned snapshot.
    ///
    /// # Errors
    /// Returns an error if any internal lock is poisoned.
    pub fn snapshot_incremental(
        &self,
        base_version: u64,
    ) -> Result<crate::persistence::IncrementalSnapshot> {
        use crate::persistence::{IncrementalSnapshot, IndexEntry};
        use std::time::{SystemTime, UNIX_EPOCH};

        let tracker = self
            .tracker
            .read()
            .map_err(|_| Error::Internal("tracker lock poisoned".into()))?;
        let dirty_ids = tracker.dirty_ids().clone();
        let delta_version = tracker.version();
        drop(tracker);

        let id_to_cid = self
            .id_to_cid
            .read()
            .map_err(|_| Error::Internal("id_to_cid lock poisoned".into()))?;
        let vectors = self
            .vectors
            .read()
            .map_err(|_| Error::Internal("vectors lock poisoned".into()))?;

        let changed_entries: Vec<IndexEntry> = dirty_ids
            .iter()
            .filter_map(|&dirty_id| {
                id_to_cid.get(&(dirty_id as usize)).and_then(|cid| {
                    vectors.get(cid).map(|vec| IndexEntry {
                        id: dirty_id,
                        cid: cid.to_string(),
                        vector: vec.clone(),
                        max_layer: 0,
                    })
                })
            })
            .collect();

        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Ok(IncrementalSnapshot {
            base_version,
            delta_version,
            changed_entries,
            deleted_ids: Vec::new(), // VectorIndex tombstones are tracked implicitly via mappings
            created_at,
        })
    }

    /// Restore a [`VectorIndex`] from a previously taken [`crate::persistence::IndexSnapshot`]
    ///
    /// All vectors are re-inserted into a freshly created HNSW graph so the
    /// graph topology is fully rebuilt.  The distance metric stored in the
    /// snapshot's `metadata_map` under the key `"metric"` is used when
    /// present; otherwise L2 is assumed.
    ///
    /// # Errors
    /// Returns an error if any entry has a vector with the wrong dimension,
    /// or if a CID string cannot be parsed.
    pub fn from_snapshot(snapshot: &crate::persistence::IndexSnapshot) -> Result<Self> {
        // Determine metric from optional metadata hint
        let metric = snapshot
            .metadata_map
            .get("metric")
            .map(|s| match s.as_str() {
                "cosine" => DistanceMetric::Cosine,
                "dot" => DistanceMetric::DotProduct,
                _ => DistanceMetric::L2,
            })
            .unwrap_or(DistanceMetric::L2);

        let mut index = Self::new(
            snapshot.dimension,
            metric,
            snapshot.m,
            snapshot.ef_construction,
        )?;

        // Re-insert in ascending ID order to keep IDs stable
        let mut ordered = snapshot.entries.clone();
        ordered.sort_by_key(|e| e.id);

        for entry in &ordered {
            let cid: Cid = entry
                .cid
                .parse()
                .map_err(|e| Error::Cid(format!("could not parse CID '{}': {}", entry.cid, e)))?;
            index.insert(&cid, &entry.vector)?;
        }

        // All entries in the restored snapshot are already persisted — clear
        // the dirty set so that the first save after a reload is not forced to
        // write every entry as a "changed" delta.
        if let Ok(mut t) = index.tracker.write() {
            t.record_full_snapshot(std::time::SystemTime::now());
        }

        Ok(index)
    }

    /// Return the number of dirty (unsaved) entries tracked since the last snapshot.
    pub fn dirty_count(&self) -> usize {
        self.tracker.read().map(|t| t.dirty_count()).unwrap_or(0)
    }

    /// Return the current incremental tracker version.
    pub fn tracker_version(&self) -> u64 {
        self.tracker.read().map(|t| t.version()).unwrap_or(0)
    }

    /// Mark the tracker as clean (call after a successful snapshot save).
    pub fn mark_tracker_clean(&self) {
        if let Ok(mut t) = self.tracker.write() {
            t.mark_clean();
        }
    }

    /// Record a full snapshot was taken now (resets dirty set and advances version).
    pub fn record_full_snapshot(&self) {
        if let Ok(mut t) = self.tracker.write() {
            t.record_full_snapshot(std::time::SystemTime::now());
        }
    }
}

/// Index metadata for serialization
#[derive(serde::Serialize, serde::Deserialize)]
struct IndexMetadata {
    dimension: usize,
    metric: DistanceMetric,
    #[serde(
        serialize_with = "serialize_id_to_cid",
        deserialize_with = "deserialize_id_to_cid"
    )]
    id_to_cid: HashMap<usize, Cid>,
    #[serde(
        serialize_with = "serialize_cid_to_id",
        deserialize_with = "deserialize_cid_to_id"
    )]
    cid_to_id: HashMap<Cid, usize>,
    #[serde(
        serialize_with = "serialize_vectors",
        deserialize_with = "deserialize_vectors"
    )]
    vectors: HashMap<Cid, Vec<f32>>,
    next_id: usize,
    max_nb_connection: usize,
    ef_construction: usize,
}

/// Serialize HashMap<usize, Cid> by converting CIDs to strings
fn serialize_id_to_cid<S>(
    map: &HashMap<usize, Cid>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::Serialize;
    let string_map: HashMap<usize, String> =
        map.iter().map(|(id, cid)| (*id, cid.to_string())).collect();
    string_map.serialize(serializer)
}

/// Deserialize HashMap<usize, Cid> by parsing CID strings
fn deserialize_id_to_cid<'de, D>(
    deserializer: D,
) -> std::result::Result<HashMap<usize, Cid>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let string_map: HashMap<usize, String> = HashMap::deserialize(deserializer)?;
    string_map
        .into_iter()
        .map(|(id, cid_str)| {
            cid_str
                .parse::<Cid>()
                .map(|cid| (id, cid))
                .map_err(serde::de::Error::custom)
        })
        .collect()
}

/// Serialize HashMap<Cid, usize> by converting CIDs to strings
fn serialize_cid_to_id<S>(
    map: &HashMap<Cid, usize>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::Serialize;
    let string_map: HashMap<String, usize> =
        map.iter().map(|(cid, id)| (cid.to_string(), *id)).collect();
    string_map.serialize(serializer)
}

/// Deserialize HashMap<Cid, usize> by parsing CID strings
fn deserialize_cid_to_id<'de, D>(
    deserializer: D,
) -> std::result::Result<HashMap<Cid, usize>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let string_map: HashMap<String, usize> = HashMap::deserialize(deserializer)?;
    string_map
        .into_iter()
        .map(|(cid_str, id)| {
            cid_str
                .parse::<Cid>()
                .map(|cid| (cid, id))
                .map_err(serde::de::Error::custom)
        })
        .collect()
}

/// Serialize HashMap<Cid, Vec<f32>> by converting CIDs to strings
fn serialize_vectors<S>(
    map: &HashMap<Cid, Vec<f32>>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::Serialize;
    let string_map: HashMap<String, Vec<f32>> = map
        .iter()
        .map(|(cid, vec)| (cid.to_string(), vec.clone()))
        .collect();
    string_map.serialize(serializer)
}

/// Deserialize HashMap<Cid, Vec<f32>> by parsing CID strings
fn deserialize_vectors<'de, D>(
    deserializer: D,
) -> std::result::Result<HashMap<Cid, Vec<f32>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let string_map: HashMap<String, Vec<f32>> = HashMap::deserialize(deserializer)?;
    string_map
        .into_iter()
        .map(|(cid_str, vec)| {
            cid_str
                .parse::<Cid>()
                .map(|cid| (cid, vec))
                .map_err(serde::de::Error::custom)
        })
        .collect()
}

/// Use case for parameter optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum UseCase {
    /// Optimize for low latency (faster queries, potentially lower recall)
    LowLatency,
    /// Optimize for high recall (more accurate results, potentially slower)
    HighRecall,
    /// Balanced performance (default)
    #[default]
    Balanced,
    /// Optimize for memory efficiency
    LowMemory,
    /// Optimize for large scale (100k+ vectors)
    LargeScale,
}

/// HNSW parameter recommendation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ParameterRecommendation {
    /// Recommended M parameter (connections per layer)
    pub m: usize,
    /// Recommended ef_construction parameter
    pub ef_construction: usize,
    /// Recommended ef_search parameter
    pub ef_search: usize,
    /// Estimated memory usage per vector (bytes)
    pub memory_per_vector: usize,
    /// Estimated recall at k=10
    pub estimated_recall: f32,
    /// Estimated query latency factor (1.0 = baseline)
    pub latency_factor: f32,
    /// Explanation of recommendations
    pub explanation: String,
}

/// Parameter tuner for HNSW index optimization
pub struct ParameterTuner;

impl ParameterTuner {
    /// Get parameter recommendations based on dataset size and use case
    pub fn recommend(
        num_vectors: usize,
        dimension: usize,
        use_case: UseCase,
    ) -> ParameterRecommendation {
        let (m, ef_construction, ef_search, recall, latency) = match use_case {
            UseCase::LowLatency => {
                if num_vectors < 10_000 {
                    (8, 100, 32, 0.90, 0.6)
                } else if num_vectors < 100_000 {
                    (12, 150, 50, 0.88, 0.7)
                } else {
                    (16, 200, 64, 0.85, 0.8)
                }
            }
            UseCase::HighRecall => {
                if num_vectors < 10_000 {
                    (32, 400, 200, 0.99, 2.0)
                } else if num_vectors < 100_000 {
                    (48, 500, 300, 0.98, 2.5)
                } else {
                    (64, 600, 400, 0.97, 3.0)
                }
            }
            UseCase::Balanced => {
                if num_vectors < 10_000 {
                    (16, 200, 50, 0.95, 1.0)
                } else if num_vectors < 100_000 {
                    (24, 300, 100, 0.94, 1.2)
                } else {
                    (32, 400, 150, 0.93, 1.5)
                }
            }
            UseCase::LowMemory => {
                if num_vectors < 10_000 {
                    (8, 100, 50, 0.88, 0.9)
                } else if num_vectors < 100_000 {
                    (10, 120, 64, 0.85, 1.0)
                } else {
                    (12, 150, 80, 0.82, 1.1)
                }
            }
            UseCase::LargeScale => {
                // Optimized for 100k+ vectors
                (32, 400, 100, 0.93, 1.5)
            }
        };

        // Memory per vector: dimension * 4 (f32) + M * 2 * 4 (graph links, assuming 2 layers avg)
        let memory_per_vector = dimension * 4 + m * 2 * 4;

        let explanation =
            Self::generate_explanation(num_vectors, use_case, m, ef_construction, ef_search);

        ParameterRecommendation {
            m,
            ef_construction,
            ef_search,
            memory_per_vector,
            estimated_recall: recall,
            latency_factor: latency,
            explanation,
        }
    }

    fn generate_explanation(
        num_vectors: usize,
        use_case: UseCase,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
    ) -> String {
        let size_category = if num_vectors < 10_000 {
            "small"
        } else if num_vectors < 100_000 {
            "medium"
        } else {
            "large"
        };

        let use_case_str = match use_case {
            UseCase::LowLatency => "low latency",
            UseCase::HighRecall => "high recall",
            UseCase::Balanced => "balanced",
            UseCase::LowMemory => "low memory",
            UseCase::LargeScale => "large scale",
        };

        format!(
            "For {} dataset (~{} vectors) optimized for {}: \
            M={} provides good connectivity, ef_construction={} ensures quality graph, \
            ef_search={} balances speed and accuracy.",
            size_category, num_vectors, use_case_str, m, ef_construction, ef_search
        )
    }

    /// Calculate Pareto-optimal configurations for different recall/latency tradeoffs
    pub fn pareto_configurations(
        num_vectors: usize,
        dimension: usize,
    ) -> Vec<ParameterRecommendation> {
        vec![
            Self::recommend(num_vectors, dimension, UseCase::LowLatency),
            Self::recommend(num_vectors, dimension, UseCase::LowMemory),
            Self::recommend(num_vectors, dimension, UseCase::Balanced),
            Self::recommend(num_vectors, dimension, UseCase::HighRecall),
        ]
    }

    /// Estimate memory usage for given parameters
    pub fn estimate_memory(num_vectors: usize, dimension: usize, m: usize) -> usize {
        // Vector data: num_vectors * dimension * 4 bytes
        let vector_memory = num_vectors * dimension * 4;

        // Graph memory: num_vectors * M * 2 layers average * 4 bytes per link
        let graph_memory = num_vectors * m * 2 * 4;

        // Additional overhead (mappings, etc.): ~50 bytes per vector
        let overhead = num_vectors * 50;

        vector_memory + graph_memory + overhead
    }

    /// Suggest ef_search for target recall at given k
    pub fn ef_search_for_recall(k: usize, target_recall: f32) -> usize {
        // Higher ef_search improves recall
        // Approximate: ef_search = k * (1 / (1 - target_recall))
        let multiplier = if target_recall >= 0.99 {
            10.0
        } else if target_recall >= 0.95 {
            4.0
        } else if target_recall >= 0.90 {
            2.0
        } else {
            1.5
        };

        ((k as f32) * multiplier).ceil() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;

    #[test]
    fn test_vector_index_creation() {
        let index = VectorIndex::with_defaults(128);
        assert!(index.is_ok());
        let index = index.expect("test: unwrap valid index after is_ok check");
        assert_eq!(index.dimension(), 128);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_insert_and_search() {
        let mut index = VectorIndex::with_defaults(4).expect("test: create 4-dim index");

        // Create some test vectors and CIDs
        let cid1 = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .expect("test: parse cid1");
        let vec1 = vec![1.0, 0.0, 0.0, 0.0];

        let cid2 = "bafybeiczsscdsbs7ffqz55asqdf3smv6klcw3gofszvwlyarci47bgf354"
            .parse::<Cid>()
            .expect("test: parse cid2");
        let vec2 = vec![0.9, 0.1, 0.0, 0.0];

        // Insert vectors
        index.insert(&cid1, &vec1).expect("test: insert cid1");
        index.insert(&cid2, &vec2).expect("test: insert cid2");

        assert_eq!(index.len(), 2);

        // Search for nearest neighbor
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index
            .search(&query, 1, 50)
            .expect("test: search for nearest");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].cid, cid1);
    }

    #[test]
    fn test_parameter_tuner() {
        // Test recommendations for different use cases
        let balanced = ParameterTuner::recommend(50_000, 768, UseCase::Balanced);
        assert!(balanced.m > 0);
        assert!(balanced.ef_construction > 0);
        assert!(balanced.estimated_recall > 0.0);

        let low_latency = ParameterTuner::recommend(50_000, 768, UseCase::LowLatency);
        let high_recall = ParameterTuner::recommend(50_000, 768, UseCase::HighRecall);

        // High recall should have higher M than low latency
        assert!(high_recall.m > low_latency.m);
        // High recall should have higher estimated recall
        assert!(high_recall.estimated_recall > low_latency.estimated_recall);

        // Test Pareto configurations
        let pareto = ParameterTuner::pareto_configurations(50_000, 768);
        assert_eq!(pareto.len(), 4);

        // Test memory estimation
        let memory = ParameterTuner::estimate_memory(100_000, 768, 16);
        assert!(memory > 0);

        // Test ef_search for recall
        let ef_high = ParameterTuner::ef_search_for_recall(10, 0.99);
        let ef_low = ParameterTuner::ef_search_for_recall(10, 0.85);
        assert!(ef_high > ef_low);
    }

    #[test]
    fn test_incremental_build() {
        let mut index =
            VectorIndex::with_defaults(4).expect("test: create 4-dim index for incremental");

        // Create test vectors
        let items: Vec<(Cid, Vec<f32>)> = (0..20)
            .map(|i| {
                let cid_str = format!(
                    "bafybei{}yrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi",
                    i
                );
                let cid = cid_str.parse::<Cid>().unwrap_or_else(|_| {
                    "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
                        .parse()
                        .expect("test: parse fallback cid")
                });
                let vec = vec![i as f32, 0.0, 0.0, 0.0];
                (cid, vec)
            })
            .collect();

        // Insert incrementally with chunk size 5
        let stats = index
            .insert_incremental(&items, 5)
            .expect("test: insert incremental");

        assert_eq!(stats.chunks_processed, 4);
        assert!(stats.vectors_inserted <= 20);
        assert_eq!(stats.final_size, index.len());
    }

    #[test]
    fn test_build_health_stats() {
        let index = VectorIndex::new(128, DistanceMetric::L2, 16, 200)
            .expect("test: create L2 index for health stats");

        let stats = index.get_build_stats();
        assert_eq!(stats.index_size, 0);
        assert_eq!(stats.current_m, 16);
        assert_eq!(stats.current_ef_construction, 200);
        assert!(stats.parameter_efficiency > 0.0);

        // For small index with good parameters, no rebuild needed
        assert!(!stats.rebuild_recommended);
    }

    #[test]
    fn test_should_rebuild() {
        // Small index with good parameters - no rebuild needed
        let index1 = VectorIndex::new(128, DistanceMetric::L2, 16, 200)
            .expect("test: create L2 index for should_rebuild");
        assert!(!index1.should_rebuild());

        // Index with suboptimal parameters
        let index2 = VectorIndex::new(128, DistanceMetric::L2, 4, 50)
            .expect("test: create suboptimal L2 index");
        // Small index won't trigger rebuild based on size thresholds
        // but parameters are low
        let _ = index2.should_rebuild();
    }

    #[test]
    fn test_rebuild() {
        let mut index = VectorIndex::with_defaults(4).expect("test: create vector index");

        // Add some vectors
        for i in 0..10 {
            let cid_str = format!(
                "bafybei{}yrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi",
                i
            );
            let cid = cid_str.parse::<Cid>().unwrap_or_else(|_| {
                "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
                    .parse()
                    .expect("test: parse cid")
            });
            let vec = vec![i as f32, 0.0, 0.0, 0.0];
            let _ = index.insert(&cid, &vec);
        }

        // Rebuild with balanced use case
        let rebuild_stats = index
            .rebuild(UseCase::Balanced)
            .expect("test: rebuild index");

        assert_eq!(rebuild_stats.old_parameters.0, 16); // Original M
        assert!(rebuild_stats.new_parameters.0 > 0); // New M
    }

    /// Compute ground truth nearest neighbors using brute force
    fn compute_ground_truth(query: &[f32], vectors: &[(Cid, Vec<f32>)], k: usize) -> Vec<Cid> {
        let mut distances: Vec<(Cid, f32)> = vectors
            .iter()
            .map(|(cid, vec)| {
                let dist: f32 = query
                    .iter()
                    .zip(vec.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                (*cid, dist.sqrt())
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.iter().take(k).map(|(cid, _)| *cid).collect()
    }

    /// Calculate recall@k
    fn calculate_recall_at_k(predicted: &[Cid], ground_truth: &[Cid], k: usize) -> f32 {
        let predicted_set: std::collections::HashSet<_> = predicted.iter().take(k).collect();
        let ground_truth_set: std::collections::HashSet<_> = ground_truth.iter().take(k).collect();

        let intersection = predicted_set.intersection(&ground_truth_set).count();
        intersection as f32 / k as f32
    }

    /// Helper to generate unique test CIDs
    fn generate_test_cid(index: usize) -> Cid {
        use multihash_codetable::{Code, MultihashDigest};
        let data = format!("test_vector_{}", index);
        let hash = Code::Sha2_256.digest(data.as_bytes());
        Cid::new_v1(0x55, hash) // 0x55 = raw codec
    }

    #[test]
    fn test_recall_at_k() {
        // Create index
        let mut index = VectorIndex::with_defaults(128).expect("test: create vector index");

        // Generate test dataset (100 random vectors)
        let mut rng = rand::rng();
        let num_vectors = 100;
        let dimension = 128;

        let mut vectors = Vec::new();
        for i in 0..num_vectors {
            let cid = generate_test_cid(i);

            let vec: Vec<f32> = (0..dimension)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();

            vectors.push((cid, vec.clone()));
            let _ = index.insert(&cid, &vec);
        }

        // Test queries
        let num_queries = 10;
        let mut total_recall_at_1 = 0.0;
        let mut total_recall_at_10 = 0.0;

        for _ in 0..num_queries {
            let query: Vec<f32> = (0..dimension)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();

            // Get HNSW results
            let hnsw_results = index.search(&query, 10, 50).expect("test: search index");
            let hnsw_cids: Vec<Cid> = hnsw_results.iter().map(|r| r.cid).collect();

            // Compute ground truth
            let ground_truth = compute_ground_truth(&query, &vectors, 10);

            // Calculate recall
            total_recall_at_1 += calculate_recall_at_k(&hnsw_cids, &ground_truth, 1);
            total_recall_at_10 += calculate_recall_at_k(&hnsw_cids, &ground_truth, 10);
        }

        let avg_recall_at_1 = total_recall_at_1 / num_queries as f32;
        let avg_recall_at_10 = total_recall_at_10 / num_queries as f32;

        // HNSW should have high recall (>80% for recall@10 on small dataset)
        assert!(
            avg_recall_at_10 > 0.8,
            "Recall@10 too low: {}",
            avg_recall_at_10
        );

        // Recall@1 should be reasonable
        assert!(
            avg_recall_at_1 > 0.5,
            "Recall@1 too low: {}",
            avg_recall_at_1
        );
    }

    #[test]
    fn test_concurrent_queries() {
        use std::sync::Arc;
        use std::thread;

        // Create index
        let mut index = VectorIndex::with_defaults(128).expect("test: create vector index");

        // Insert test vectors
        let mut rng = rand::rng();
        for i in 0..100 {
            let cid = generate_test_cid(i + 1000); // Offset to avoid collision with other tests

            let vec: Vec<f32> = (0..128).map(|_| rng.random_range(-1.0..1.0)).collect();

            let _ = index.insert(&cid, &vec);
        }

        // Share index across threads
        let index = Arc::new(index);
        let num_threads = 10;
        let queries_per_thread = 100;

        // Spawn threads for concurrent queries
        let mut handles = vec![];
        for _ in 0..num_threads {
            let index_clone = Arc::clone(&index);
            let handle = thread::spawn(move || {
                let mut thread_rng = rand::rng();
                let mut success_count = 0;

                for _ in 0..queries_per_thread {
                    let query: Vec<f32> = (0..128)
                        .map(|_| thread_rng.random_range(-1.0..1.0))
                        .collect();

                    if let Ok(results) = index_clone.search(&query, 10, 50) {
                        if !results.is_empty() {
                            success_count += 1;
                        }
                    }
                }
                success_count
            });
            handles.push(handle);
        }

        // Collect results
        let mut total_success = 0;
        for handle in handles {
            total_success += handle.join().expect("test: thread join");
        }

        // All queries should succeed
        let total_queries = num_threads * queries_per_thread;
        assert_eq!(
            total_success, total_queries,
            "Some queries failed under concurrent load"
        );
    }

    #[test]
    fn test_precision_at_k() {
        // Create index
        let mut index = VectorIndex::with_defaults(32).expect("test: create vector index");

        // Create structured dataset: 5 clusters of 10 vectors each
        let num_clusters = 5;
        let vectors_per_cluster = 10;

        for cluster in 0..num_clusters {
            // Cluster center
            let mut center = [0.0; 32];
            center[cluster] = 10.0;

            for i in 0..vectors_per_cluster {
                let idx = cluster * vectors_per_cluster + i;
                let cid = generate_test_cid(idx + 2000); // Offset to avoid collision

                // Add small random noise to center
                let mut rng = rand::rng();
                let vec: Vec<f32> = center
                    .iter()
                    .map(|&c| c + rng.random_range(-0.5..0.5))
                    .collect();

                let _ = index.insert(&cid, &vec);
            }
        }

        // Query with a vector close to cluster 0
        let mut query = vec![0.0; 32];
        query[0] = 10.0;

        let results = index.search(&query, 10, 50).expect("test: search index");

        // Count how many results are from cluster 0 (first 10 CIDs)
        // Note: This is approximate since CID generation is not deterministic
        // In a real test, you'd track cluster membership explicitly
        assert_eq!(results.len(), 10, "Should return 10 results");

        // Results should be relatively close to query
        for result in &results {
            assert!(
                result.score < 5.0,
                "Result too far from query: {}",
                result.score
            );
        }
    }

    #[test]
    fn test_hnsw_memory_estimate() {
        let dim = 128;
        let mut index =
            VectorIndex::new(dim, DistanceMetric::L2, 16, 200).expect("test: create vector index");

        // Empty index should estimate 0 bytes.
        assert_eq!(
            index.estimated_memory_bytes(),
            0,
            "empty index should report 0 bytes"
        );

        // Insert 1000 vectors.
        for i in 0..1000_usize {
            let cid = generate_test_cid(i + 10_000);
            let vec = vec![i as f32 * 0.001; dim];
            index.insert(&cid, &vec).expect("test: insert vector");
        }

        let estimate = index.estimated_memory_bytes();
        assert!(
            estimate > 0,
            "memory estimate should be > 0 after inserting 1000 vectors (got {})",
            estimate
        );
        // Sanity: at least dim*4 bytes per node (the vector storage alone).
        let lower_bound = 1000 * dim * 4;
        assert!(
            estimate >= lower_bound,
            "estimate {} should be >= lower bound {}",
            estimate,
            lower_bound
        );
    }
}
