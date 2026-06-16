//! Dynamic embedding updates for evolving embedding spaces
//!
//! This module provides mechanisms for:
//! - Online embedding updates
//! - Version migration
//! - Incremental fine-tuning
//! - Embedding space evolution tracking

use crate::VectorIndex;
use ipfrs_core::{Cid, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Version of an embedding model
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelVersion {
    /// Major version
    pub major: u32,
    /// Minor version
    pub minor: u32,
    /// Patch version
    pub patch: u32,
    /// Optional tag (e.g., "alpha", "beta")
    pub tag: Option<String>,
}

impl ModelVersion {
    /// Create a new model version
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            tag: None,
        }
    }

    /// Create a version with a tag
    pub fn with_tag(mut self, tag: String) -> Self {
        self.tag = Some(tag);
        self
    }

    /// Check if this version is compatible with another (same major version)
    pub fn is_compatible_with(&self, other: &ModelVersion) -> bool {
        self.major == other.major
    }
}

impl std::fmt::Display for ModelVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(tag) = &self.tag {
            write!(f, "-{}", tag)?;
        }
        Ok(())
    }
}

/// Embedding transformation for migrating between versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingTransform {
    /// Source version
    pub from_version: ModelVersion,
    /// Target version
    pub to_version: ModelVersion,
    /// Transformation matrix (if dimensions change)
    pub transform_matrix: Option<Vec<Vec<f32>>>,
    /// Bias vector
    pub bias: Option<Vec<f32>>,
}

impl EmbeddingTransform {
    /// Create an identity transform (no change)
    pub fn identity(version: ModelVersion) -> Self {
        Self {
            from_version: version.clone(),
            to_version: version,
            transform_matrix: None,
            bias: None,
        }
    }

    /// Create a linear transformation
    pub fn linear(
        from_version: ModelVersion,
        to_version: ModelVersion,
        matrix: Vec<Vec<f32>>,
    ) -> Self {
        Self {
            from_version,
            to_version,
            transform_matrix: Some(matrix),
            bias: None,
        }
    }

    /// Apply transformation to an embedding
    pub fn apply(&self, embedding: &[f32]) -> Vec<f32> {
        let mut result = embedding.to_vec();

        // Apply matrix transformation if present
        if let Some(matrix) = &self.transform_matrix {
            let out_dim = matrix[0].len();
            let mut transformed = vec![0.0; out_dim];

            for (i, row) in matrix.iter().enumerate() {
                if i >= embedding.len() {
                    break;
                }
                for (j, &val) in row.iter().enumerate() {
                    transformed[j] += embedding[i] * val;
                }
            }

            result = transformed;
        }

        // Apply bias if present
        if let Some(bias) = &self.bias {
            for (i, &b) in bias.iter().enumerate() {
                if i < result.len() {
                    result[i] += b;
                }
            }
        }

        result
    }
}

/// Dynamic index that supports multiple embedding versions
pub struct DynamicIndex {
    /// Indices for each version
    indices: Arc<RwLock<HashMap<ModelVersion, VectorIndex>>>,
    /// Current active version
    active_version: Arc<RwLock<ModelVersion>>,
    /// Transformations between versions
    transforms: Arc<RwLock<HashMap<(ModelVersion, ModelVersion), EmbeddingTransform>>>,
    /// Embedding dimension
    dimension: usize,
}

impl DynamicIndex {
    /// Create a new dynamic index
    pub fn new(initial_version: ModelVersion, dimension: usize) -> Result<Self> {
        let mut indices = HashMap::new();
        let index = VectorIndex::with_defaults(dimension)?;
        indices.insert(initial_version.clone(), index);

        Ok(Self {
            indices: Arc::new(RwLock::new(indices)),
            active_version: Arc::new(RwLock::new(initial_version)),
            transforms: Arc::new(RwLock::new(HashMap::new())),
            dimension,
        })
    }

    /// Get the current active version
    pub fn active_version(&self) -> ModelVersion {
        self.active_version
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Add a new version with optional transform from previous version
    pub fn add_version(
        &self,
        version: ModelVersion,
        transform: Option<EmbeddingTransform>,
    ) -> Result<()> {
        let mut indices = self.indices.write().unwrap_or_else(|e| e.into_inner());

        if indices.contains_key(&version) {
            return Err(Error::InvalidInput(format!(
                "Version {} already exists",
                version
            )));
        }

        let index = VectorIndex::with_defaults(self.dimension)?;
        indices.insert(version.clone(), index);

        // Add transform if provided
        if let Some(t) = transform {
            let mut transforms = self.transforms.write().unwrap_or_else(|e| e.into_inner());
            transforms.insert((t.from_version.clone(), t.to_version.clone()), t);
        }

        Ok(())
    }

    /// Set the active version
    pub fn set_active_version(&self, version: ModelVersion) -> Result<()> {
        let indices = self.indices.read().unwrap_or_else(|e| e.into_inner());

        if !indices.contains_key(&version) {
            return Err(Error::InvalidInput(format!(
                "Version {} does not exist",
                version
            )));
        }

        let mut active = self
            .active_version
            .write()
            .unwrap_or_else(|e| e.into_inner());
        *active = version;

        Ok(())
    }

    /// Insert an embedding for a specific version
    pub fn insert(
        &self,
        cid: &Cid,
        embedding: &[f32],
        version: Option<ModelVersion>,
    ) -> Result<()> {
        let version = version.unwrap_or_else(|| self.active_version());

        let mut indices = self.indices.write().unwrap_or_else(|e| e.into_inner());
        let index = indices
            .get_mut(&version)
            .ok_or_else(|| Error::InvalidInput(format!("Version {} does not exist", version)))?;

        index.insert(cid, embedding)?;
        Ok(())
    }

    /// Update an existing embedding
    pub fn update(
        &self,
        cid: &Cid,
        new_embedding: &[f32],
        version: Option<ModelVersion>,
    ) -> Result<()> {
        let version = version.unwrap_or_else(|| self.active_version());

        let mut indices = self.indices.write().unwrap_or_else(|e| e.into_inner());
        let index = indices
            .get_mut(&version)
            .ok_or_else(|| Error::InvalidInput(format!("Version {} does not exist", version)))?;

        // First delete the old embedding
        index.delete(cid)?;
        // Then insert the new one
        index.insert(cid, new_embedding)?;

        Ok(())
    }

    /// Migrate embeddings from one version to another
    pub fn migrate(&self, from: &ModelVersion, to: &ModelVersion) -> Result<usize> {
        let transforms = self.transforms.read().unwrap_or_else(|e| e.into_inner());
        let transform = transforms
            .get(&(from.clone(), to.clone()))
            .ok_or_else(|| Error::InvalidInput(format!("No transform from {} to {}", from, to)))?
            .clone();
        drop(transforms);

        // Get all embeddings from source version
        let indices = self.indices.read().unwrap_or_else(|e| e.into_inner());
        let source_index = indices.get(from).ok_or_else(|| {
            Error::InvalidInput(format!("Source version {} does not exist", from))
        })?;

        // Ensure target version exists
        if !indices.contains_key(to) {
            return Err(Error::InvalidInput(format!(
                "Target version {} does not exist",
                to
            )));
        }

        // Get all embeddings from source index
        let embeddings = source_index.get_all_embeddings();
        drop(indices);

        // Apply transformation and insert into target index
        let mut migrated_count = 0;
        for (cid, embedding) in embeddings {
            // Apply transformation
            let transformed = transform.apply(&embedding);

            // Insert into target index
            let mut indices = self.indices.write().unwrap_or_else(|e| e.into_inner());
            if let Some(target_index) = indices.get_mut(to) {
                // Only insert if not already present
                if !target_index.contains(&cid) {
                    target_index.insert(&cid, &transformed)?;
                    migrated_count += 1;
                }
            }
            drop(indices);
        }

        Ok(migrated_count)
    }

    /// Get statistics for all versions
    pub fn version_stats(&self) -> HashMap<ModelVersion, VersionStats> {
        let indices = self.indices.read().unwrap_or_else(|e| e.into_inner());

        indices
            .iter()
            .map(|(version, index)| {
                let stats = VersionStats {
                    version: version.clone(),
                    num_embeddings: index.len(),
                    is_active: version == &self.active_version(),
                };
                (version.clone(), stats)
            })
            .collect()
    }
}

/// Statistics for a specific version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionStats {
    /// Version identifier
    pub version: ModelVersion,
    /// Number of embeddings in this version
    pub num_embeddings: usize,
    /// Whether this is the active version
    pub is_active: bool,
}

/// Online updater for incremental fine-tuning
pub struct OnlineUpdater {
    /// Learning rate for updates
    learning_rate: f32,
    /// Momentum factor
    momentum: f32,
    /// Velocity (for momentum)
    velocity: Arc<RwLock<HashMap<Cid, Vec<f32>>>>,
}

impl OnlineUpdater {
    /// Create a new online updater
    pub fn new(learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Update an embedding with a gradient
    pub fn update(&self, cid: &Cid, embedding: &[f32], gradient: &[f32]) -> Vec<f32> {
        let mut velocity = self.velocity.write().unwrap_or_else(|e| e.into_inner());

        // Get or initialize velocity for this CID
        let v = velocity
            .entry(*cid)
            .or_insert_with(|| vec![0.0; embedding.len()]);

        // Update velocity with momentum
        for i in 0..embedding.len().min(gradient.len()) {
            v[i] = self.momentum * v[i] - self.learning_rate * gradient[i];
        }

        // Apply velocity to embedding
        embedding
            .iter()
            .zip(v.iter())
            .map(|(&e, &vel)| e + vel)
            .collect()
    }

    /// Clear velocity history
    pub fn reset(&self) {
        let mut velocity = self.velocity.write().unwrap_or_else(|e| e.into_inner());
        velocity.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> OnlineUpdaterStats {
        let velocity = self.velocity.read().unwrap_or_else(|e| e.into_inner());

        OnlineUpdaterStats {
            learning_rate: self.learning_rate,
            momentum: self.momentum,
            num_tracked: velocity.len(),
        }
    }
}

/// Statistics for online updater
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineUpdaterStats {
    /// Learning rate
    pub learning_rate: f32,
    /// Momentum
    pub momentum: f32,
    /// Number of tracked embeddings
    pub num_tracked: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_version() {
        let v1 = ModelVersion::new(1, 0, 0);
        let v2 = ModelVersion::new(1, 1, 0);
        let v3 = ModelVersion::new(2, 0, 0);

        assert!(v1.is_compatible_with(&v2));
        assert!(!v1.is_compatible_with(&v3));

        assert_eq!(v1.to_string(), "1.0.0");
        assert_eq!(v1.with_tag("alpha".into()).to_string(), "1.0.0-alpha");
    }

    #[test]
    fn test_embedding_transform() {
        let v1 = ModelVersion::new(1, 0, 0);
        let v2 = ModelVersion::new(1, 1, 0);

        // Identity transform
        let identity = EmbeddingTransform::identity(v1.clone());
        let embedding = vec![1.0, 2.0, 3.0];
        let result = identity.apply(&embedding);
        assert_eq!(result, embedding);

        // Linear transform (2x2 -> 2x2)
        let matrix = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
        let transform = EmbeddingTransform::linear(v1, v2, matrix);
        let embedding = vec![1.0, 2.0];
        let result = transform.apply(&embedding);
        assert_eq!(result, vec![1.0, 4.0]);
    }

    #[test]
    fn test_dynamic_index_creation() {
        let version = ModelVersion::new(1, 0, 0);
        let index = DynamicIndex::new(version.clone(), 128)
            .expect("test: DynamicIndex creation should succeed");

        assert_eq!(index.active_version(), version);
    }

    #[test]
    fn test_add_version() {
        let v1 = ModelVersion::new(1, 0, 0);
        let v2 = ModelVersion::new(1, 1, 0);

        let index =
            DynamicIndex::new(v1.clone(), 128).expect("test: DynamicIndex creation should succeed");
        index
            .add_version(v2.clone(), None)
            .expect("test: add_version should succeed");

        let stats = index.version_stats();
        assert_eq!(stats.len(), 2);
        assert!(stats.contains_key(&v1));
        assert!(stats.contains_key(&v2));
    }

    #[test]
    fn test_set_active_version() {
        let v1 = ModelVersion::new(1, 0, 0);
        let v2 = ModelVersion::new(1, 1, 0);

        let index =
            DynamicIndex::new(v1.clone(), 128).expect("test: DynamicIndex creation should succeed");
        index
            .add_version(v2.clone(), None)
            .expect("test: add_version should succeed");

        assert_eq!(index.active_version(), v1);

        index
            .set_active_version(v2.clone())
            .expect("test: set_active_version should succeed");
        assert_eq!(index.active_version(), v2);
    }

    #[test]
    fn test_insert_and_update() {
        use multihash_codetable::{Code, MultihashDigest};

        let version = ModelVersion::new(1, 0, 0);
        let index =
            DynamicIndex::new(version, 3).expect("test: DynamicIndex creation should succeed");

        let data = "test_embedding";
        let hash = Code::Sha2_256.digest(data.as_bytes());
        let cid = Cid::new_v1(0x55, hash);

        let embedding = vec![1.0, 2.0, 3.0];
        index
            .insert(&cid, &embedding, None)
            .expect("test: insert should succeed");

        let stats = index.version_stats();
        assert_eq!(
            stats
                .values()
                .next()
                .expect("test: stats should have at least one entry")
                .num_embeddings,
            1
        );

        // Update the embedding
        let new_embedding = vec![4.0, 5.0, 6.0];
        index
            .update(&cid, &new_embedding, None)
            .expect("test: update should succeed");

        let stats = index.version_stats();
        assert_eq!(
            stats
                .values()
                .next()
                .expect("test: stats should have at least one entry")
                .num_embeddings,
            1
        );
    }

    #[test]
    fn test_online_updater() {
        use multihash_codetable::{Code, MultihashDigest};

        let updater = OnlineUpdater::new(0.1, 0.9);

        let data = "test";
        let hash = Code::Sha2_256.digest(data.as_bytes());
        let cid = Cid::new_v1(0x55, hash);

        let embedding = vec![1.0, 1.0, 1.0];
        let gradient = vec![0.1, 0.1, 0.1];

        let updated = updater.update(&cid, &embedding, &gradient);

        // With learning_rate=0.1, gradient should decrease embedding
        assert!(updated[0] < 1.0);
        assert_eq!(updated.len(), 3);

        let stats = updater.stats();
        assert_eq!(stats.num_tracked, 1);
    }

    #[test]
    fn test_updater_momentum() {
        use multihash_codetable::{Code, MultihashDigest};

        let updater = OnlineUpdater::new(0.1, 0.9);

        let data = "test";
        let hash = Code::Sha2_256.digest(data.as_bytes());
        let cid = Cid::new_v1(0x55, hash);

        let embedding = vec![1.0];
        let gradient = vec![0.1];

        // First update
        let updated1 = updater.update(&cid, &embedding, &gradient);

        // Second update with same gradient
        let updated2 = updater.update(&cid, &updated1, &gradient);

        // With momentum, second update should have larger magnitude
        let delta1 = (embedding[0] - updated1[0]).abs();
        let delta2 = (updated1[0] - updated2[0]).abs();
        assert!(delta2 > delta1);
    }
}
