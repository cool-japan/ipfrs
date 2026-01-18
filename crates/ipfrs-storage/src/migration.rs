//! Storage migration utilities
//!
//! This module provides utilities for migrating data between different storage backends,
//! enabling seamless transitions in production deployments.

use crate::traits::BlockStore;
use ipfrs_core::{Cid, Result};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Migration statistics
#[derive(Debug, Clone, Default)]
pub struct MigrationStats {
    /// Total blocks migrated
    pub blocks_migrated: u64,
    /// Total bytes migrated
    pub bytes_migrated: u64,
    /// Number of blocks skipped (already present in destination)
    pub blocks_skipped: u64,
    /// Number of errors encountered
    pub errors: u64,
    /// Migration duration
    pub duration: Duration,
    /// Migration throughput in blocks per second
    pub blocks_per_second: f64,
    /// Migration throughput in bytes per second
    pub bytes_per_second: f64,
}

impl MigrationStats {
    /// Calculate throughput metrics
    fn calculate_throughput(&mut self, duration: Duration) {
        let seconds = duration.as_secs_f64();
        if seconds > 0.0 {
            self.blocks_per_second = self.blocks_migrated as f64 / seconds;
            self.bytes_per_second = self.bytes_migrated as f64 / seconds;
        }
    }
}

/// Migration configuration
#[derive(Debug, Clone)]
pub struct MigrationConfig {
    /// Batch size for bulk operations
    pub batch_size: usize,
    /// Whether to skip blocks that already exist in destination
    pub skip_existing: bool,
    /// Whether to verify each block after migration
    pub verify: bool,
    /// Maximum number of concurrent operations
    pub concurrency: usize,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            skip_existing: true,
            verify: false,
            concurrency: 4,
        }
    }
}

/// Progress callback type
pub type ProgressCallback = Arc<dyn Fn(u64, u64) + Send + Sync>;

/// Storage migrator
pub struct StorageMigrator<S: BlockStore, D: BlockStore> {
    source: Arc<S>,
    destination: Arc<D>,
    config: MigrationConfig,
    progress_callback: Option<ProgressCallback>,
}

impl<S: BlockStore, D: BlockStore> StorageMigrator<S, D> {
    /// Create a new migrator
    pub fn new(source: Arc<S>, destination: Arc<D>) -> Self {
        Self {
            source,
            destination,
            config: MigrationConfig::default(),
            progress_callback: None,
        }
    }

    /// Create with custom configuration
    pub fn with_config(source: Arc<S>, destination: Arc<D>, config: MigrationConfig) -> Self {
        Self {
            source,
            destination,
            config,
            progress_callback: None,
        }
    }

    /// Set progress callback
    pub fn with_progress_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(u64, u64) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Arc::new(callback));
        self
    }

    /// Migrate all blocks from source to destination
    pub async fn migrate_all(&self) -> Result<MigrationStats> {
        let start = Instant::now();

        let blocks_migrated = AtomicU64::new(0);
        let bytes_migrated = AtomicU64::new(0);
        let blocks_skipped = AtomicU64::new(0);
        let errors = AtomicU64::new(0);

        // Get all CIDs from source
        let all_cids = self.source.list_cids()?;
        let total_blocks = all_cids.len() as u64;

        // Migrate in batches
        for batch in all_cids.chunks(self.config.batch_size) {
            // Check which blocks already exist in destination if skip_existing is enabled
            let cids_to_migrate = if self.config.skip_existing {
                let exists = self.destination.has_many(batch).await?;
                batch
                    .iter()
                    .zip(exists.iter())
                    .filter_map(|(cid, exists)| {
                        if *exists {
                            blocks_skipped.fetch_add(1, Ordering::Relaxed);
                            None
                        } else {
                            Some(*cid)
                        }
                    })
                    .collect::<Vec<_>>()
            } else {
                batch.to_vec()
            };

            if cids_to_migrate.is_empty() {
                continue;
            }

            // Get blocks from source
            let blocks_result = self.source.get_many(&cids_to_migrate).await?;

            // Filter out None values and collect valid blocks
            let mut valid_blocks = Vec::new();
            for block_opt in blocks_result {
                if let Some(block) = block_opt {
                    bytes_migrated.fetch_add(block.data().len() as u64, Ordering::Relaxed);
                    valid_blocks.push(block);
                } else {
                    errors.fetch_add(1, Ordering::Relaxed);
                }
            }

            // Put blocks to destination
            if !valid_blocks.is_empty() {
                match self.destination.put_many(&valid_blocks).await {
                    Ok(_) => {
                        blocks_migrated.fetch_add(valid_blocks.len() as u64, Ordering::Relaxed);

                        // Verify if enabled
                        if self.config.verify {
                            let cids: Vec<Cid> = valid_blocks.iter().map(|b| *b.cid()).collect();
                            let verified = self.destination.has_many(&cids).await?;
                            let failed = verified.iter().filter(|&&exists| !exists).count();
                            if failed > 0 {
                                errors.fetch_add(failed as u64, Ordering::Relaxed);
                            }
                        }
                    }
                    Err(_) => {
                        errors.fetch_add(valid_blocks.len() as u64, Ordering::Relaxed);
                    }
                }
            }

            // Call progress callback
            if let Some(ref callback) = self.progress_callback {
                let migrated = blocks_migrated.load(Ordering::Relaxed);
                callback(migrated, total_blocks);
            }
        }

        let mut stats = MigrationStats {
            blocks_migrated: blocks_migrated.load(Ordering::Relaxed),
            bytes_migrated: bytes_migrated.load(Ordering::Relaxed),
            blocks_skipped: blocks_skipped.load(Ordering::Relaxed),
            errors: errors.load(Ordering::Relaxed),
            duration: start.elapsed(),
            blocks_per_second: 0.0,
            bytes_per_second: 0.0,
        };

        stats.calculate_throughput(stats.duration);

        Ok(stats)
    }

    /// Migrate specific CIDs
    pub async fn migrate_cids(&self, cids: &[Cid]) -> Result<MigrationStats> {
        let start = Instant::now();

        let blocks_migrated = AtomicU64::new(0);
        let bytes_migrated = AtomicU64::new(0);
        let blocks_skipped = AtomicU64::new(0);
        let errors = AtomicU64::new(0);

        // Migrate in batches
        for batch in cids.chunks(self.config.batch_size) {
            // Check which blocks already exist
            let cids_to_migrate = if self.config.skip_existing {
                let exists = self.destination.has_many(batch).await?;
                batch
                    .iter()
                    .zip(exists.iter())
                    .filter_map(|(cid, exists)| {
                        if *exists {
                            blocks_skipped.fetch_add(1, Ordering::Relaxed);
                            None
                        } else {
                            Some(*cid)
                        }
                    })
                    .collect::<Vec<_>>()
            } else {
                batch.to_vec()
            };

            if cids_to_migrate.is_empty() {
                continue;
            }

            // Get and migrate blocks
            let blocks_result = self.source.get_many(&cids_to_migrate).await?;
            let mut valid_blocks = Vec::new();

            for block_opt in blocks_result {
                if let Some(block) = block_opt {
                    bytes_migrated.fetch_add(block.data().len() as u64, Ordering::Relaxed);
                    valid_blocks.push(block);
                } else {
                    errors.fetch_add(1, Ordering::Relaxed);
                }
            }

            if !valid_blocks.is_empty() {
                match self.destination.put_many(&valid_blocks).await {
                    Ok(_) => {
                        blocks_migrated.fetch_add(valid_blocks.len() as u64, Ordering::Relaxed);
                    }
                    Err(_) => {
                        errors.fetch_add(valid_blocks.len() as u64, Ordering::Relaxed);
                    }
                }
            }
        }

        let mut stats = MigrationStats {
            blocks_migrated: blocks_migrated.load(Ordering::Relaxed),
            bytes_migrated: bytes_migrated.load(Ordering::Relaxed),
            blocks_skipped: blocks_skipped.load(Ordering::Relaxed),
            errors: errors.load(Ordering::Relaxed),
            duration: start.elapsed(),
            blocks_per_second: 0.0,
            bytes_per_second: 0.0,
        };

        stats.calculate_throughput(stats.duration);

        Ok(stats)
    }
}

/// Helper function to migrate between stores
pub async fn migrate_storage<S: BlockStore, D: BlockStore>(
    source: Arc<S>,
    destination: Arc<D>,
) -> Result<MigrationStats> {
    let migrator = StorageMigrator::new(source, destination);
    migrator.migrate_all().await
}

/// Helper function to migrate with progress reporting
pub async fn migrate_storage_with_progress<S: BlockStore, D: BlockStore, F>(
    source: Arc<S>,
    destination: Arc<D>,
    progress_callback: F,
) -> Result<MigrationStats>
where
    F: Fn(u64, u64) + Send + Sync + 'static,
{
    let migrator =
        StorageMigrator::new(source, destination).with_progress_callback(progress_callback);
    migrator.migrate_all().await
}

/// Migrate with custom batch size for optimal performance
pub async fn migrate_storage_batched<S: BlockStore, D: BlockStore>(
    source: Arc<S>,
    destination: Arc<D>,
    batch_size: usize,
) -> Result<MigrationStats> {
    let config = MigrationConfig {
        batch_size,
        ..Default::default()
    };
    let migrator = StorageMigrator::with_config(source, destination, config);
    migrator.migrate_all().await
}

/// Migrate with verification enabled (slower but safer)
pub async fn migrate_storage_verified<S: BlockStore, D: BlockStore>(
    source: Arc<S>,
    destination: Arc<D>,
) -> Result<MigrationStats> {
    let config = MigrationConfig {
        verify: true,
        ..Default::default()
    };
    let migrator = StorageMigrator::with_config(source, destination, config);
    migrator.migrate_all().await
}

/// Estimate migration time and space requirements
#[derive(Debug, Clone)]
pub struct MigrationEstimate {
    /// Total blocks to migrate
    pub total_blocks: usize,
    /// Total bytes to migrate
    pub total_bytes: u64,
    /// Estimated duration at 100 blocks/sec
    pub estimated_duration_low: Duration,
    /// Estimated duration at 1000 blocks/sec
    pub estimated_duration_high: Duration,
    /// Space required in destination
    pub space_required: u64,
}

/// Estimate migration requirements
pub async fn estimate_migration<S: BlockStore>(source: Arc<S>) -> Result<MigrationEstimate> {
    let all_cids = source.list_cids()?;
    let total_blocks = all_cids.len();

    // Sample first 100 blocks to estimate average size
    let sample_size = total_blocks.min(100);
    let sample_cids: Vec<_> = all_cids.iter().take(sample_size).copied().collect();

    let blocks = source.get_many(&sample_cids).await?;
    let sample_bytes: u64 = blocks
        .iter()
        .filter_map(|b| b.as_ref())
        .map(|b| b.data().len() as u64)
        .sum();

    let avg_block_size = if sample_size > 0 {
        sample_bytes / sample_size as u64
    } else {
        0
    };

    let total_bytes = avg_block_size * total_blocks as u64;

    // Estimate durations (conservative: 100 blocks/sec, optimistic: 1000 blocks/sec)
    let estimated_duration_low = Duration::from_secs(total_blocks as u64 / 100);
    let estimated_duration_high = Duration::from_secs(total_blocks as u64 / 1000);

    Ok(MigrationEstimate {
        total_blocks,
        total_bytes,
        estimated_duration_low,
        estimated_duration_high,
        space_required: total_bytes,
    })
}

/// Migration validation - verify both stores have identical content
pub async fn validate_migration<S: BlockStore, D: BlockStore>(
    source: Arc<S>,
    destination: Arc<D>,
) -> Result<bool> {
    let source_cids = source.list_cids()?;
    let dest_cids = destination.list_cids()?;

    // Check if same number of blocks
    if source_cids.len() != dest_cids.len() {
        return Ok(false);
    }

    // Check all source CIDs exist in destination
    let exists = destination.has_many(&source_cids).await?;
    Ok(exists.iter().all(|&e| e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryBlockStore;
    use bytes::Bytes;
    use ipfrs_core::Block;

    #[tokio::test]
    async fn test_basic_migration() {
        let source = Arc::new(MemoryBlockStore::new());
        let destination = Arc::new(MemoryBlockStore::new());

        // Add some blocks to source
        for i in 0..10 {
            let block = Block::new(Bytes::from(format!("block {}", i))).unwrap();
            source.put(&block).await.unwrap();
        }

        assert_eq!(source.len(), 10);
        assert_eq!(destination.len(), 0);

        // Migrate
        let stats = migrate_storage(source.clone(), destination.clone())
            .await
            .unwrap();

        assert_eq!(stats.blocks_migrated, 10);
        assert_eq!(stats.blocks_skipped, 0);
        assert_eq!(stats.errors, 0);
        assert_eq!(destination.len(), 10);
    }

    #[tokio::test]
    async fn test_migration_skip_existing() {
        let source = Arc::new(MemoryBlockStore::new());
        let destination = Arc::new(MemoryBlockStore::new());

        // Add blocks to both stores
        let mut blocks = Vec::new();
        for i in 0..10 {
            let block = Block::new(Bytes::from(format!("block {}", i))).unwrap();
            blocks.push(block);
        }

        // Add all to source
        for block in &blocks {
            source.put(block).await.unwrap();
        }

        // Add first 5 to destination
        for block in blocks.iter().take(5) {
            destination.put(block).await.unwrap();
        }

        // Migrate with skip_existing
        let config = MigrationConfig {
            skip_existing: true,
            ..Default::default()
        };
        let migrator = StorageMigrator::with_config(source, destination.clone(), config);
        let stats = migrator.migrate_all().await.unwrap();

        assert_eq!(stats.blocks_migrated, 5); // Only new blocks
        assert_eq!(stats.blocks_skipped, 5); // Existing blocks
        assert_eq!(destination.len(), 10);
    }

    #[tokio::test]
    async fn test_migration_with_progress() {
        let source = Arc::new(MemoryBlockStore::new());
        let destination = Arc::new(MemoryBlockStore::new());

        // Add blocks
        for i in 0..20 {
            let block = Block::new(Bytes::from(format!("block {}", i))).unwrap();
            source.put(&block).await.unwrap();
        }

        let progress_called = Arc::new(AtomicU64::new(0));
        let progress_called_clone = progress_called.clone();

        let stats = migrate_storage_with_progress(source, destination, move |_current, _total| {
            progress_called_clone.fetch_add(1, Ordering::Relaxed);
        })
        .await
        .unwrap();

        assert_eq!(stats.blocks_migrated, 20);
        assert!(progress_called.load(Ordering::Relaxed) > 0);
    }

    #[tokio::test]
    async fn test_migrate_storage_batched() {
        let source = Arc::new(MemoryBlockStore::new());
        let destination = Arc::new(MemoryBlockStore::new());

        // Add blocks
        for i in 0..50 {
            let block = Block::new(Bytes::from(format!("block {}", i))).unwrap();
            source.put(&block).await.unwrap();
        }

        let stats = migrate_storage_batched(source, destination.clone(), 10)
            .await
            .unwrap();

        assert_eq!(stats.blocks_migrated, 50);
        assert_eq!(destination.len(), 50);
    }

    #[tokio::test]
    async fn test_estimate_migration() {
        let source = Arc::new(MemoryBlockStore::new());

        // Add blocks with unique data (so they have unique CIDs)
        for i in 0..100 {
            let block = Block::new(Bytes::from(format!("block {}", i))).unwrap();
            source.put(&block).await.unwrap();
        }

        let estimate = estimate_migration(source).await.unwrap();

        assert_eq!(estimate.total_blocks, 100);
        assert!(estimate.total_bytes > 0);
        assert!(estimate.space_required > 0);
    }

    #[tokio::test]
    async fn test_validate_migration() {
        let source = Arc::new(MemoryBlockStore::new());
        let destination = Arc::new(MemoryBlockStore::new());

        // Add same blocks to both
        for i in 0..10 {
            let block = Block::new(Bytes::from(format!("block {}", i))).unwrap();
            source.put(&block).await.unwrap();
            destination.put(&block).await.unwrap();
        }

        let valid = validate_migration(source.clone(), destination.clone())
            .await
            .unwrap();

        assert!(valid);

        // Add one more block to source only
        let extra_block = Block::new(Bytes::from("extra")).unwrap();
        source.put(&extra_block).await.unwrap();

        let valid = validate_migration(source, destination).await.unwrap();
        assert!(!valid);
    }
}
