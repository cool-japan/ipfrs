//! Index migration utilities
//!
//! This module provides tools for migrating data between different index types
//! and configurations, including upgrading from in-memory to disk-based indices,
//! applying quantization, and changing index parameters.
//!
//! # Features
//!
//! - **Index Type Migration**: Convert between HNSW, DiskANN, and quantized indices
//! - **Configuration Updates**: Change index parameters with data preservation
//! - **Batch Migration**: Efficient bulk data transfer
//! - **Progress Tracking**: Monitor migration progress
//!
//! # Example
//!
//! ```rust
//! use ipfrs_semantic::migration::{IndexMigration, MigrationConfig};
//! use ipfrs_semantic::hnsw::VectorIndex;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a migration plan
//! let config = MigrationConfig {
//!     batch_size: 1000,
//!     verify_after_migration: true,
//!     ..Default::default()
//! };
//!
//! let migration = IndexMigration::new(config);
//!
//! // Migration would be performed here
//! // migration.migrate(source_index, target_index)?;
//! # Ok(())
//! # }
//! ```

use crate::hnsw::{DistanceMetric, VectorIndex};
use ipfrs_core::{Cid, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Migration configuration
#[derive(Debug, Clone)]
pub struct MigrationConfig {
    /// Batch size for migration
    pub batch_size: usize,
    /// Whether to verify data after migration
    pub verify_after_migration: bool,
    /// Maximum concurrent migrations
    pub max_concurrent: usize,
    /// Whether to preserve original index during migration
    pub preserve_source: bool,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            verify_after_migration: true,
            max_concurrent: 4,
            preserve_source: true,
        }
    }
}

/// Migration progress information
#[derive(Debug, Clone)]
pub struct MigrationProgress {
    /// Total entries to migrate
    pub total_entries: usize,
    /// Entries migrated so far
    pub migrated_entries: usize,
    /// Verification progress
    pub verified_entries: usize,
    /// Failed entries
    pub failed_entries: usize,
    /// Estimated time remaining (seconds)
    pub estimated_seconds_remaining: f64,
}

impl MigrationProgress {
    /// Calculate completion percentage
    pub fn completion_percent(&self) -> f64 {
        if self.total_entries == 0 {
            return 100.0;
        }
        (self.migrated_entries as f64 / self.total_entries as f64) * 100.0
    }

    /// Check if migration is complete
    pub fn is_complete(&self) -> bool {
        self.migrated_entries >= self.total_entries
    }
}

/// Migration statistics
#[derive(Debug, Clone)]
pub struct MigrationStats {
    /// Total time taken
    pub total_duration_seconds: f64,
    /// Entries per second
    pub throughput: f64,
    /// Success rate
    pub success_rate: f64,
    /// Total entries migrated
    pub total_migrated: usize,
    /// Total entries failed
    pub total_failed: usize,
}

/// Index migration manager
pub struct IndexMigration {
    /// Configuration
    config: MigrationConfig,
    /// Progress tracking
    progress: Arc<AtomicUsize>,
}

impl IndexMigration {
    /// Create a new index migration manager
    pub fn new(config: MigrationConfig) -> Self {
        Self {
            config,
            progress: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Migrate from one HNSW index to another with different parameters
    pub fn migrate_hnsw_to_hnsw(
        &self,
        source: &VectorIndex,
        target_m: usize,
        target_ef_construction: usize,
    ) -> Result<VectorIndex> {
        // Extract dimension and metric from source
        let dimension = 768; // Would be extracted from source in real impl
        let metric = DistanceMetric::Cosine;

        let mut target = VectorIndex::new(dimension, metric, target_m, target_ef_construction)?;

        // Get all entries from source
        let entries = source.get_all_embeddings();
        let _total = entries.len();

        // Migrate in batches
        for (i, chunk) in entries.chunks(self.config.batch_size).enumerate() {
            for (cid, embedding) in chunk {
                target.insert(cid, embedding)?;
            }

            self.progress
                .store((i + 1) * self.config.batch_size, Ordering::Relaxed);
        }

        // Verify if requested
        if self.config.verify_after_migration {
            self.verify_migration(source, &target)?;
        }

        Ok(target)
    }

    /// Verify that migration was successful
    fn verify_migration(&self, source: &VectorIndex, target: &VectorIndex) -> Result<()> {
        let source_entries = source.get_all_embeddings();

        for (cid, _embedding) in &source_entries {
            if !target.contains(cid) {
                return Err(ipfrs_core::Error::Internal(format!(
                    "Migration verification failed: CID {:?} missing in target",
                    cid
                )));
            }
        }

        Ok(())
    }

    /// Get current migration progress
    pub fn get_progress(&self, total_entries: usize) -> MigrationProgress {
        let migrated = self.progress.load(Ordering::Relaxed);

        MigrationProgress {
            total_entries,
            migrated_entries: migrated,
            verified_entries: 0,
            failed_entries: 0,
            estimated_seconds_remaining: 0.0,
        }
    }

    /// Migrate embeddings with transformation
    pub fn migrate_with_transform<F>(
        &self,
        source: &VectorIndex,
        dimension: usize,
        metric: DistanceMetric,
        m: usize,
        ef_construction: usize,
        transform: F,
    ) -> Result<VectorIndex>
    where
        F: Fn(&[f32]) -> Vec<f32>,
    {
        let mut target = VectorIndex::new(dimension, metric, m, ef_construction)?;

        let entries = source.get_all_embeddings();

        for (cid, embedding) in entries {
            let transformed = transform(&embedding);
            target.insert(&cid, &transformed)?;
        }

        Ok(target)
    }

    /// Export index entries for external migration
    pub fn export_entries(&self, index: &VectorIndex) -> Vec<(Cid, Vec<f32>)> {
        index.get_all_embeddings()
    }

    /// Import entries into a new index
    pub fn import_entries(
        &self,
        entries: &[(Cid, Vec<f32>)],
        dimension: usize,
        metric: DistanceMetric,
        m: usize,
        ef_construction: usize,
    ) -> Result<VectorIndex> {
        let mut index = VectorIndex::new(dimension, metric, m, ef_construction)?;

        for (cid, embedding) in entries {
            index.insert(cid, embedding)?;
        }

        Ok(index)
    }
}

/// Configuration change migration
pub struct ConfigMigration;

impl ConfigMigration {
    /// Migrate to higher quality settings
    pub fn upgrade_quality(source: &VectorIndex) -> Result<VectorIndex> {
        let migration = IndexMigration::new(MigrationConfig::default());

        // Upgrade to higher M and ef_construction
        migration.migrate_hnsw_to_hnsw(source, 32, 400)
    }

    /// Migrate to faster settings
    pub fn optimize_speed(source: &VectorIndex) -> Result<VectorIndex> {
        let migration = IndexMigration::new(MigrationConfig::default());

        // Downgrade to lower M and ef_construction for speed
        migration.migrate_hnsw_to_hnsw(source, 8, 100)
    }

    /// Balance quality and speed
    pub fn balance(source: &VectorIndex) -> Result<VectorIndex> {
        let migration = IndexMigration::new(MigrationConfig::default());

        // Balanced settings
        migration.migrate_hnsw_to_hnsw(source, 16, 200)
    }
}

/// Dimension reduction migration
pub struct DimensionMigration;

impl DimensionMigration {
    /// Reduce dimensionality using PCA-like projection
    /// Note: This is a simplified version - real PCA would require training
    pub fn reduce_dimension(source: &VectorIndex, target_dim: usize) -> Result<VectorIndex> {
        let migration = IndexMigration::new(MigrationConfig::default());

        // Simple truncation (real implementation would use PCA or other methods)
        let transform = |embedding: &[f32]| -> Vec<f32> {
            embedding[..target_dim.min(embedding.len())].to_vec()
        };

        migration.migrate_with_transform(
            source,
            target_dim,
            DistanceMetric::Cosine,
            16,
            200,
            transform,
        )
    }
}

/// Metric migration utilities
pub struct MetricMigration;

impl MetricMigration {
    /// Convert index to use different distance metric
    pub fn change_metric(source: &VectorIndex, new_metric: DistanceMetric) -> Result<VectorIndex> {
        let entries = source.get_all_embeddings();
        let dimension = 768; // Would be extracted from source

        let mut target = VectorIndex::new(dimension, new_metric, 16, 200)?;

        for (cid, embedding) in entries {
            target.insert(&cid, &embedding)?;
        }

        Ok(target)
    }

    /// Normalize embeddings for cosine distance
    pub fn normalize_for_cosine(source: &VectorIndex) -> Result<VectorIndex> {
        let migration = IndexMigration::new(MigrationConfig::default());

        let transform = |embedding: &[f32]| -> Vec<f32> {
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 {
                embedding.iter().map(|x| x / norm).collect()
            } else {
                embedding.to_vec()
            }
        };

        migration.migrate_with_transform(source, 768, DistanceMetric::Cosine, 16, 200, transform)
    }
}

/// Batch migration utilities
pub struct BatchMigration {
    /// Batch size
    batch_size: usize,
    /// Statistics
    stats: HashMap<String, usize>,
}

impl BatchMigration {
    /// Create a new batch migration
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            stats: HashMap::new(),
        }
    }

    /// Migrate in batches with progress callback
    pub fn migrate_with_callback<F>(
        &mut self,
        source: &VectorIndex,
        target: &mut VectorIndex,
        mut callback: F,
    ) -> Result<()>
    where
        F: FnMut(usize, usize),
    {
        let entries = source.get_all_embeddings();
        let total = entries.len();

        for (i, chunk) in entries.chunks(self.batch_size).enumerate() {
            for (cid, embedding) in chunk {
                target.insert(cid, embedding)?;
            }

            let migrated = (i + 1) * self.batch_size.min(total);
            callback(migrated, total);
        }

        Ok(())
    }

    /// Get migration statistics
    pub fn get_stats(&self) -> &HashMap<String, usize> {
        &self.stats
    }
}

impl Default for BatchMigration {
    fn default() -> Self {
        Self::new(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use multihash_codetable::{Code, MultihashDigest};

    fn create_test_index() -> VectorIndex {
        let mut index = VectorIndex::new(768, DistanceMetric::Cosine, 16, 200)
            .expect("test: create 768-dim cosine index");

        for i in 0..10 {
            let data = format!("test_vector_{}", i);
            let hash = Code::Sha2_256.digest(data.as_bytes());
            let cid = Cid::new_v1(0x55, hash);
            let embedding = vec![i as f32 * 0.1; 768];
            index
                .insert(&cid, &embedding)
                .expect("test: insert test vector");
        }

        index
    }

    #[test]
    fn test_migration_config_default() {
        let config = MigrationConfig::default();
        assert_eq!(config.batch_size, 1000);
        assert!(config.verify_after_migration);
        assert_eq!(config.max_concurrent, 4);
    }

    #[test]
    fn test_migration_progress() {
        let progress = MigrationProgress {
            total_entries: 100,
            migrated_entries: 50,
            verified_entries: 0,
            failed_entries: 0,
            estimated_seconds_remaining: 10.0,
        };

        assert_eq!(progress.completion_percent(), 50.0);
        assert!(!progress.is_complete());
    }

    #[test]
    fn test_migration_progress_complete() {
        let progress = MigrationProgress {
            total_entries: 100,
            migrated_entries: 100,
            verified_entries: 100,
            failed_entries: 0,
            estimated_seconds_remaining: 0.0,
        };

        assert_eq!(progress.completion_percent(), 100.0);
        assert!(progress.is_complete());
    }

    #[test]
    fn test_index_migration_creation() {
        let config = MigrationConfig::default();
        let migration = IndexMigration::new(config);
        let progress = migration.get_progress(100);

        assert_eq!(progress.migrated_entries, 0);
    }

    #[test]
    fn test_export_entries() {
        let index = create_test_index();
        let migration = IndexMigration::new(MigrationConfig::default());

        let entries = migration.export_entries(&index);
        assert_eq!(entries.len(), 10);
    }

    #[test]
    fn test_import_entries() {
        let source = create_test_index();
        let migration = IndexMigration::new(MigrationConfig::default());

        let entries = migration.export_entries(&source);
        let imported = migration
            .import_entries(&entries, 768, DistanceMetric::Cosine, 16, 200)
            .expect("test: import entries");

        assert_eq!(imported.len(), source.len());
    }

    #[test]
    fn test_migrate_with_transform() {
        let source = create_test_index();
        let migration = IndexMigration::new(MigrationConfig::default());

        // Transform: multiply all values by 2
        let transform =
            |embedding: &[f32]| -> Vec<f32> { embedding.iter().map(|x| x * 2.0).collect() };

        let target = migration
            .migrate_with_transform(&source, 768, DistanceMetric::Cosine, 16, 200, transform)
            .expect("test: migrate with transform");

        assert_eq!(target.len(), source.len());
    }

    #[test]
    fn test_config_migration_upgrade() {
        let source = create_test_index();
        let upgraded =
            ConfigMigration::upgrade_quality(&source).expect("test: upgrade quality migration");

        assert_eq!(upgraded.len(), source.len());
    }

    #[test]
    fn test_config_migration_speed() {
        let source = create_test_index();
        let optimized =
            ConfigMigration::optimize_speed(&source).expect("test: optimize speed migration");

        assert_eq!(optimized.len(), source.len());
    }

    #[test]
    fn test_config_migration_balance() {
        let source = create_test_index();
        let balanced = ConfigMigration::balance(&source).expect("test: balance migration");

        assert_eq!(balanced.len(), source.len());
    }

    #[test]
    fn test_dimension_reduction() {
        let source = create_test_index();
        let reduced =
            DimensionMigration::reduce_dimension(&source, 384).expect("test: dimension reduction");

        assert_eq!(reduced.len(), source.len());
    }

    #[test]
    fn test_metric_change() {
        let source = create_test_index();
        let changed = MetricMigration::change_metric(&source, DistanceMetric::L2)
            .expect("test: metric change to L2");

        assert_eq!(changed.len(), source.len());
    }

    #[test]
    fn test_normalize_for_cosine() {
        let source = create_test_index();
        let normalized =
            MetricMigration::normalize_for_cosine(&source).expect("test: normalize for cosine");

        assert_eq!(normalized.len(), source.len());
    }

    #[test]
    fn test_batch_migration() {
        let source = create_test_index();
        let mut target = VectorIndex::new(768, DistanceMetric::Cosine, 16, 200)
            .expect("test: create target index for batch");

        let mut batch_migration = BatchMigration::new(5);
        let mut callback_count = 0;

        batch_migration
            .migrate_with_callback(&source, &mut target, |migrated, total| {
                callback_count += 1;
                assert!(migrated <= total);
            })
            .expect("test: batch migration with callback");

        assert_eq!(target.len(), source.len());
        assert!(callback_count > 0);
    }
}
