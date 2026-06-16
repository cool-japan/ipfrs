//! Repository management and analysis operations for Node

use ipfrs_core::{Cid, Result};
use ipfrs_storage::{OrphanGarbageCollector, OrphanGcConfig};

use crate::fsck::{FilesystemChecker, FsckConfig};
use crate::gc::{GarbageCollector, GcConfig};

use super::{FsckResult, GcResult, Node, OrphanGcResult};

impl Node {
    /// Run garbage collection
    pub async fn repo_gc(&self, dry_run: bool) -> Result<GcResult> {
        self.repo_gc_with_options(dry_run, 3600).await
    }

    /// Run orphan block garbage collection using [`OrphanGarbageCollector`].
    ///
    /// Collects blocks that are present in the store but are not referenced by
    /// any pinned CID.  When `dry_run` is `true` the orphans are reported but
    /// not actually deleted.  The `min_age_secs` parameter is stored in the
    /// config for future use when per-block timestamps become available.
    pub async fn gc_blocks(&self, dry_run: bool, min_age_secs: u64) -> Result<OrphanGcResult> {
        let storage = self.storage()?;

        // Build the pinned CID set from the local pin manager.
        let pinned_cids: std::collections::HashSet<String> = self
            .pin_manager
            .list()
            .into_iter()
            .map(|pin| pin.cid.to_string())
            .collect();

        let config = OrphanGcConfig {
            dry_run,
            min_age_secs,
            batch_size: 1000,
        };
        let collector = OrphanGarbageCollector::new(config);
        let result = collector.collect(storage.as_ref(), &pinned_cids).await?;
        Ok(result)
    }

    /// Run garbage collection with explicit age filter.
    ///
    /// `min_age_secs` – blocks younger than this many seconds are spared
    /// (passed through to the underlying GC config).
    pub async fn repo_gc_with_options(&self, dry_run: bool, min_age_secs: u64) -> Result<GcResult> {
        let storage = self.storage()?;
        let gc = GarbageCollector::new(storage.clone(), self.pin_manager.clone());

        let config = GcConfig {
            dry_run,
            min_age_seconds: min_age_secs,
            ..Default::default()
        };

        let stats = gc.collect(config).await?;

        Ok(GcResult {
            blocks_collected: stats.blocks_collected,
            bytes_freed: stats.bytes_freed,
            blocks_marked: stats.blocks_marked,
            blocks_scanned: stats.blocks_scanned,
            duration: stats.duration,
            cancelled: stats.cancelled,
        })
    }

    /// Verify repository integrity
    pub async fn repo_fsck(&self) -> Result<FsckResult> {
        let storage = self.storage()?;
        let fsck = FilesystemChecker::new(storage.clone());

        let config = FsckConfig::default();
        let result = fsck.check(config).await?;

        Ok(FsckResult {
            blocks_checked: result.blocks_checked,
            blocks_valid: result.blocks_valid,
            blocks_corrupt: result.blocks_corrupt,
            blocks_missing: result.blocks_missing,
        })
    }

    /// Run quick filesystem check (only verify CIDs match content)
    pub async fn repo_fsck_quick(&self) -> Result<FsckResult> {
        let storage = self.storage()?;
        let fsck = FilesystemChecker::new(storage.clone());

        let result = fsck.quick_check().await?;

        Ok(FsckResult {
            blocks_checked: result.blocks_checked,
            blocks_valid: result.blocks_valid,
            blocks_corrupt: result.blocks_corrupt,
            blocks_missing: result.blocks_missing,
        })
    }

    /// Get garbage collection statistics (without running GC)
    pub fn gc_stats(&self) -> Result<(usize, usize)> {
        let storage = self.storage()?;
        let gc = GarbageCollector::new(storage.clone(), self.pin_manager.clone());

        let unpinned_count = gc.count_unpinned()?;
        let pinned_count = self.pin_manager.count();

        Ok((pinned_count, unpinned_count))
    }

    /// Get comprehensive repository statistics
    pub async fn repo_stat(&self) -> Result<crate::repo::RepoStats> {
        let storage = self.storage()?;
        let analyzer = crate::repo::RepoAnalyzer::new(storage.clone(), self.pin_manager.clone());
        analyzer.analyze().await
    }

    /// Get block size distribution
    pub async fn block_distribution(&self) -> Result<crate::repo::BlockDistribution> {
        let storage = self.storage()?;
        let analyzer = crate::repo::RepoAnalyzer::new(storage.clone(), self.pin_manager.clone());
        analyzer.block_distribution().await
    }

    /// Find duplicate blocks (same content, different CIDs)
    pub async fn find_duplicates(&self) -> Result<Vec<Vec<Cid>>> {
        let storage = self.storage()?;
        let analyzer = crate::repo::RepoAnalyzer::new(storage.clone(), self.pin_manager.clone());
        analyzer.find_duplicates().await
    }

    /// Get largest blocks in the repository
    pub async fn largest_blocks(&self, limit: usize) -> Result<Vec<(Cid, u64)>> {
        let storage = self.storage()?;
        let analyzer = crate::repo::RepoAnalyzer::new(storage.clone(), self.pin_manager.clone());
        analyzer.largest_blocks(limit).await
    }

    /// Find orphaned blocks (not pinned and not referenced)
    pub async fn find_orphaned(&self) -> Result<Vec<Cid>> {
        let storage = self.storage()?;
        let analyzer = crate::repo::RepoAnalyzer::new(storage.clone(), self.pin_manager.clone());
        analyzer.find_orphaned_blocks().await
    }
}
