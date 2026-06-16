//! Partial HNSW sync with dirty region tracking.
//!
//! This module implements delta-based synchronisation of HNSW graph state
//! between peers.  Rather than shipping the entire index on every gossip
//! round, it tracks which nodes have changed since the last successful sync
//! and ships only those deltas.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │  DirtyRegionTracker                                          │
//! │   ┌─────────────────────────────────────────────────────┐   │
//! │   │  dirty_nodes: HashMap<layer, HashSet<node_id>>      │   │
//! │   │  generation:  AtomicU64                             │   │
//! │   └─────────────────────────────────────────────────────┘   │
//! │                        ▲  ▼                                  │
//! │  PartialSyncManager                                          │
//! │   ┌─────────────────────────────────────────────────────┐   │
//! │   │  record_change / build_delta / apply_delta          │   │
//! │   │  pending_deltas: Vec<EmbeddingDelta> (ack tracking) │   │
//! │   └─────────────────────────────────────────────────────┘   │
//! └──────────────────────────────────────────────────────────────┘
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

// ──────────────────────────────────────────────────────────────────────────────
// EmbeddingRegion
// ──────────────────────────────────────────────────────────────────────────────

/// A region of the HNSW graph identified by layer + node ID range.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingRegion {
    /// HNSW layer (0 = bottom, highest connectivity).
    pub layer: usize,
    /// First node ID included in the region (inclusive).
    pub node_start: u64,
    /// First node ID **not** included in the region (exclusive).
    pub node_end: u64,
}

impl EmbeddingRegion {
    /// Create a new region for `layer` covering `[node_start, node_end)`.
    pub fn new(layer: usize, node_start: u64, node_end: u64) -> Self {
        Self {
            layer,
            node_start,
            node_end,
        }
    }

    /// Return `true` when `node_id` falls inside this region.
    #[inline]
    pub fn contains(&self, node_id: u64) -> bool {
        node_id >= self.node_start && node_id < self.node_end
    }

    /// Number of node IDs spanned by this region.
    #[inline]
    pub fn size(&self) -> u64 {
        self.node_end.saturating_sub(self.node_start)
    }

    /// Return `true` when the two regions share at least one node ID on the
    /// same HNSW layer.
    pub fn overlaps(&self, other: &EmbeddingRegion) -> bool {
        if self.layer != other.layer {
            return false;
        }
        // Two intervals [a, b) and [c, d) overlap iff a < d && c < b.
        self.node_start < other.node_end && other.node_start < self.node_end
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// EmbeddingDelta
// ──────────────────────────────────────────────────────────────────────────────

/// A delta snapshot of changed embeddings in a region.
///
/// `changed_ids` and `vectors` are parallel slices: `changed_ids[i]` is the
/// node ID whose new vector is `vectors[i]`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingDelta {
    /// The region this delta covers.
    pub region: EmbeddingRegion,
    /// Node IDs that changed (parallel to `vectors`).
    pub changed_ids: Vec<u64>,
    /// Corresponding new vectors (parallel to `changed_ids`).
    pub vectors: Vec<Vec<f32>>,
    /// Monotonically increasing version counter.
    pub generation: u64,
    /// Originating peer identifier.
    pub source_peer: String,
    /// Wall-clock timestamp (milliseconds since Unix epoch) at creation.
    pub created_at_ms: u64,
}

impl EmbeddingDelta {
    /// Construct an empty delta for `region` at `generation`.
    pub fn new(
        region: EmbeddingRegion,
        generation: u64,
        source_peer: impl Into<String>,
        now_ms: u64,
    ) -> Self {
        Self {
            region,
            changed_ids: Vec::new(),
            vectors: Vec::new(),
            generation,
            source_peer: source_peer.into(),
            created_at_ms: now_ms,
        }
    }

    /// Append a single changed node.
    pub fn add_change(&mut self, node_id: u64, vector: Vec<f32>) {
        self.changed_ids.push(node_id);
        self.vectors.push(vector);
    }

    /// Number of changed nodes recorded in this delta.
    #[inline]
    pub fn change_count(&self) -> usize {
        self.changed_ids.len()
    }

    /// `true` when no changes have been recorded.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.changed_ids.is_empty()
    }

    /// Rough byte estimate for gossip bandwidth accounting.
    ///
    /// Formula: `4 * Σ(dim_i) + 8 * n` where `n` is the number of changed
    /// nodes and `dim_i` is the length of the i-th vector.
    pub fn estimated_bytes(&self) -> usize {
        let vector_bytes: usize = self.vectors.iter().map(|v| 4 * v.len()).sum();
        let id_bytes = 8 * self.changed_ids.len();
        vector_bytes + id_bytes
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DirtyRegionTracker
// ──────────────────────────────────────────────────────────────────────────────

/// Tracks which HNSW nodes are dirty (changed since the last successful sync).
///
/// Thread-safe: all public methods take `&self` and use interior mutability.
pub struct DirtyRegionTracker {
    /// Per-layer sets of dirty node IDs.
    dirty_nodes: RwLock<HashMap<usize, HashSet<u64>>>,
    /// Monotonically increasing sync-generation counter.
    generation: AtomicU64,
    /// Upper bound on how many dirty nodes we track before the ratio saturates
    /// at 1.0.  Does *not* enforce a hard cap — it is a denominator for
    /// `dirty_ratio`.
    max_dirty_nodes: usize,
}

impl DirtyRegionTracker {
    /// Create a new tracker.
    ///
    /// `max_dirty_nodes` is used only as the denominator when computing
    /// `dirty_ratio`; it does not enforce a hard limit on dirty-set growth.
    pub fn new(max_dirty_nodes: usize) -> Self {
        Self {
            dirty_nodes: RwLock::new(HashMap::new()),
            generation: AtomicU64::new(0),
            max_dirty_nodes,
        }
    }

    /// Mark `node_id` as dirty in `layer`.
    pub fn mark_dirty(&self, layer: usize, node_id: u64) {
        let mut guard = self.dirty_nodes.write();
        guard.entry(layer).or_default().insert(node_id);
    }

    /// Mark all nodes in `node_ids` as dirty in `layer`.
    pub fn mark_dirty_batch(&self, layer: usize, node_ids: &[u64]) {
        if node_ids.is_empty() {
            return;
        }
        let mut guard = self.dirty_nodes.write();
        let set = guard.entry(layer).or_default();
        for &id in node_ids {
            set.insert(id);
        }
    }

    /// Remove all dirty entries whose layer and node ID fall inside `region`.
    ///
    /// Returns the number of entries removed.
    pub fn clear_region(&self, region: &EmbeddingRegion) -> usize {
        let mut guard = self.dirty_nodes.write();
        let set = match guard.get_mut(&region.layer) {
            Some(s) => s,
            None => return 0,
        };
        let before = set.len();
        set.retain(|&id| !region.contains(id));
        let removed = before - set.len();
        // Tidy up empty sets so dirty_layers() stays accurate.
        if set.is_empty() {
            guard.remove(&region.layer);
        }
        removed
    }

    /// Return all dirty node IDs whose layer and node ID fall inside `region`.
    pub fn dirty_in_region(&self, region: &EmbeddingRegion) -> Vec<u64> {
        let guard = self.dirty_nodes.read();
        match guard.get(&region.layer) {
            Some(set) => set
                .iter()
                .filter(|&&id| region.contains(id))
                .copied()
                .collect(),
            None => Vec::new(),
        }
    }

    /// Total number of dirty nodes across all layers.
    pub fn total_dirty(&self) -> usize {
        self.dirty_nodes.read().values().map(HashSet::len).sum()
    }

    /// Fraction of `max_dirty_nodes` that are currently dirty.
    ///
    /// Clamped to `[0.0, 1.0]`.
    pub fn dirty_ratio(&self) -> f64 {
        if self.max_dirty_nodes == 0 {
            return 0.0;
        }
        let ratio = self.total_dirty() as f64 / self.max_dirty_nodes as f64;
        ratio.min(1.0)
    }

    /// `true` when at least one node is dirty.
    pub fn has_dirty(&self) -> bool {
        self.dirty_nodes.read().values().any(|s| !s.is_empty())
    }

    /// Current generation counter value.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Atomically increment the generation counter and return the new value.
    ///
    /// Call this after a sync round completes to invalidate stale deltas.
    pub fn advance_generation(&self) -> u64 {
        self.generation.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Layers that have at least one dirty node, in ascending order.
    pub fn dirty_layers(&self) -> Vec<usize> {
        let guard = self.dirty_nodes.read();
        let mut layers: Vec<usize> = guard
            .iter()
            .filter(|(_, s)| !s.is_empty())
            .map(|(&l, _)| l)
            .collect();
        layers.sort_unstable();
        layers
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// PartialSyncManager
// ──────────────────────────────────────────────────────────────────────────────

/// Manages partial sync of HNSW graph changes between peers.
///
/// Wraps a [`DirtyRegionTracker`] and adds outbound delta bookkeeping so the
/// caller can detect when a peer has not yet acknowledged a delta.
pub struct PartialSyncManager {
    tracker: Arc<DirtyRegionTracker>,
    /// Deltas sent to peers that have not yet been acknowledged.
    pending_deltas: RwLock<Vec<EmbeddingDelta>>,
    /// Maximum number of unacknowledged deltas we keep.
    max_pending: usize,
    /// Identifier for the local peer (used as `source_peer` in new deltas).
    local_peer_id: String,
}

impl PartialSyncManager {
    /// Create a new manager.
    ///
    /// - `max_dirty_nodes`: denominator for `dirty_ratio` (forwarded to the
    ///   inner [`DirtyRegionTracker`]).
    /// - `max_pending`: maximum number of un-acked deltas held in memory.
    /// - `local_peer_id`: identifier stamped onto outbound deltas.
    pub fn new(
        max_dirty_nodes: usize,
        max_pending: usize,
        local_peer_id: impl Into<String>,
    ) -> Self {
        Self {
            tracker: Arc::new(DirtyRegionTracker::new(max_dirty_nodes)),
            pending_deltas: RwLock::new(Vec::new()),
            max_pending,
            local_peer_id: local_peer_id.into(),
        }
    }

    /// Record that the vector for `node_id` in `layer` has changed.
    pub fn record_change(&self, layer: usize, node_id: u64) {
        self.tracker.mark_dirty(layer, node_id);
    }

    /// Build a delta containing all dirty nodes that fall inside `region`.
    ///
    /// `get_vector` is called for each dirty node ID to retrieve its current
    /// embedding.  If `get_vector` returns `None` for a node the node is
    /// skipped (it may have been deleted).
    pub fn build_delta<F>(
        &self,
        region: &EmbeddingRegion,
        get_vector: F,
        now_ms: u64,
    ) -> EmbeddingDelta
    where
        F: Fn(u64) -> Option<Vec<f32>>,
    {
        let generation = self.tracker.generation();
        let mut delta = EmbeddingDelta::new(
            region.clone(),
            generation,
            self.local_peer_id.clone(),
            now_ms,
        );

        let dirty_ids = self.tracker.dirty_in_region(region);
        for node_id in dirty_ids {
            if let Some(vec) = get_vector(node_id) {
                delta.add_change(node_id, vec);
            }
        }
        delta
    }

    /// Build one delta per dirty layer, covering all dirty nodes in that layer.
    ///
    /// `get_vector` receives `(layer, node_id)` and should return the current
    /// embedding for that node, or `None` if the node no longer exists.
    ///
    /// Layers with no dirty nodes are omitted from the result.
    pub fn build_all_deltas<F>(&self, get_vector: F, now_ms: u64) -> Vec<EmbeddingDelta>
    where
        F: Fn(usize, u64) -> Option<Vec<f32>>,
    {
        let generation = self.tracker.generation();

        // Snapshot dirty state under the read lock.
        let layer_dirty: Vec<(usize, Vec<u64>)> = {
            let guard = self.tracker.dirty_nodes.read();
            guard
                .iter()
                .filter(|(_, s)| !s.is_empty())
                .map(|(&layer, ids)| (layer, ids.iter().copied().collect()))
                .collect()
        };

        let mut deltas = Vec::with_capacity(layer_dirty.len());
        for (layer, ids) in layer_dirty {
            // Region that covers the entire layer (u64::MAX exclusive upper bound).
            let region = EmbeddingRegion::new(layer, 0, u64::MAX);
            let mut delta =
                EmbeddingDelta::new(region, generation, self.local_peer_id.clone(), now_ms);
            for node_id in ids {
                if let Some(vec) = get_vector(layer, node_id) {
                    delta.add_change(node_id, vec);
                }
            }
            if !delta.is_empty() {
                deltas.push(delta);
            }
        }
        deltas
    }

    /// Apply an incoming delta from a peer.
    ///
    /// Clears the local dirty status for all nodes mentioned in the delta (the
    /// peer has the authoritative version of those nodes).
    pub fn apply_delta(&self, delta: &EmbeddingDelta) {
        if delta.changed_ids.is_empty() {
            return;
        }
        let region = &delta.region;
        let mut guard = self.tracker.dirty_nodes.write();
        if let Some(set) = guard.get_mut(&region.layer) {
            for &node_id in &delta.changed_ids {
                set.remove(&node_id);
            }
            if set.is_empty() {
                guard.remove(&region.layer);
            }
        }
    }

    /// Push a delta onto the pending list (waiting for peer acknowledgement).
    ///
    /// Returns `false` when the pending list is already at capacity; the
    /// caller should ack older deltas first.
    pub fn push_pending(&self, delta: EmbeddingDelta) -> bool {
        let mut guard = self.pending_deltas.write();
        if guard.len() >= self.max_pending {
            return false;
        }
        guard.push(delta);
        true
    }

    /// Remove all pending deltas whose `generation` is ≤ `generation`.
    ///
    /// Returns the number of deltas removed.
    pub fn ack_generation(&self, generation: u64) -> usize {
        let mut guard = self.pending_deltas.write();
        let before = guard.len();
        guard.retain(|d| d.generation > generation);
        before - guard.len()
    }

    /// Number of pending (un-acked) deltas.
    pub fn pending_count(&self) -> usize {
        self.pending_deltas.read().len()
    }

    /// Access the underlying [`DirtyRegionTracker`].
    pub fn tracker(&self) -> &DirtyRegionTracker {
        &self.tracker
    }

    /// Convenience wrapper: `dirty_ratio` of the inner tracker.
    pub fn dirty_ratio(&self) -> f64 {
        self.tracker.dirty_ratio()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── EmbeddingRegion ───────────────────────────────────────────────────────

    #[test]
    fn test_region_contains() {
        let r = EmbeddingRegion::new(0, 10, 20);
        // Left boundary inclusive.
        assert!(r.contains(10));
        // Interior.
        assert!(r.contains(15));
        // Right boundary exclusive.
        assert!(!r.contains(20));
        // Below start.
        assert!(!r.contains(9));
        // Far above end.
        assert!(!r.contains(100));
    }

    #[test]
    fn test_region_overlaps() {
        let a = EmbeddingRegion::new(0, 10, 20);
        // Same region overlaps with itself.
        assert!(a.overlaps(&a));
        // Partial overlap from the right.
        let b = EmbeddingRegion::new(0, 15, 25);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
        // Adjacent but non-overlapping.
        let c = EmbeddingRegion::new(0, 20, 30);
        assert!(!a.overlaps(&c));
        // Different layers never overlap.
        let d = EmbeddingRegion::new(1, 10, 20);
        assert!(!a.overlaps(&d));
        // Entirely contained.
        let e = EmbeddingRegion::new(0, 12, 18);
        assert!(a.overlaps(&e));
        // Entirely disjoint (left).
        let f = EmbeddingRegion::new(0, 0, 10);
        assert!(!a.overlaps(&f));
    }

    #[test]
    fn test_region_size() {
        let r = EmbeddingRegion::new(0, 5, 15);
        assert_eq!(r.size(), 10);
        // Zero-size region.
        let z = EmbeddingRegion::new(0, 7, 7);
        assert_eq!(z.size(), 0);
        // Saturating sub: node_end < node_start should give 0, not overflow.
        let inv = EmbeddingRegion {
            layer: 0,
            node_start: 100,
            node_end: 10,
        };
        assert_eq!(inv.size(), 0);
    }

    // ── EmbeddingDelta ────────────────────────────────────────────────────────

    #[test]
    fn test_embedding_delta_add_change() {
        let region = EmbeddingRegion::new(0, 0, 100);
        let mut delta = EmbeddingDelta::new(region, 1, "peer-a", 42_000);
        assert!(delta.is_empty());
        assert_eq!(delta.change_count(), 0);

        delta.add_change(5, vec![1.0, 2.0, 3.0]);
        delta.add_change(7, vec![4.0, 5.0]);
        assert!(!delta.is_empty());
        assert_eq!(delta.change_count(), 2);
        assert_eq!(delta.changed_ids, vec![5, 7]);
        assert_eq!(delta.vectors[0], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_embedding_delta_estimated_bytes() {
        let region = EmbeddingRegion::new(0, 0, 100);
        let mut delta = EmbeddingDelta::new(region, 1, "peer-a", 0);
        // No changes → 0 bytes.
        assert_eq!(delta.estimated_bytes(), 0);

        // One node with a 4-element vector:  4*4 + 8*1 = 24
        delta.add_change(1, vec![1.0; 4]);
        assert_eq!(delta.estimated_bytes(), 24);

        // Second node with 8-element vector: 24 + 4*8 + 8 = 64
        delta.add_change(2, vec![0.5; 8]);
        assert_eq!(delta.estimated_bytes(), 64);
    }

    // ── DirtyRegionTracker ────────────────────────────────────────────────────

    #[test]
    fn test_dirty_tracker_mark_and_query() {
        let t = DirtyRegionTracker::new(100);
        assert!(!t.has_dirty());
        assert_eq!(t.total_dirty(), 0);

        t.mark_dirty(0, 42);
        t.mark_dirty(0, 43);
        t.mark_dirty(1, 10);

        assert!(t.has_dirty());
        assert_eq!(t.total_dirty(), 3);

        let region = EmbeddingRegion::new(0, 40, 50);
        let mut ids = t.dirty_in_region(&region);
        ids.sort_unstable();
        assert_eq!(ids, vec![42, 43]);

        // Layer 1 not visible in a layer-0 region.
        let region1 = EmbeddingRegion::new(1, 0, 100);
        let mut ids1 = t.dirty_in_region(&region1);
        ids1.sort_unstable();
        assert_eq!(ids1, vec![10]);
    }

    #[test]
    fn test_dirty_tracker_mark_batch() {
        let t = DirtyRegionTracker::new(100);
        t.mark_dirty_batch(0, &[1, 2, 3, 4, 5]);
        assert_eq!(t.total_dirty(), 5);
        // Inserting the same IDs again should not grow the set.
        t.mark_dirty_batch(0, &[1, 2, 3]);
        assert_eq!(t.total_dirty(), 5);
        // Empty slice is a no-op.
        t.mark_dirty_batch(0, &[]);
        assert_eq!(t.total_dirty(), 5);
    }

    #[test]
    fn test_dirty_tracker_clear_region() {
        let t = DirtyRegionTracker::new(100);
        t.mark_dirty_batch(0, &[10, 11, 12, 50]);
        t.mark_dirty(1, 10);

        let region = EmbeddingRegion::new(0, 10, 13);
        let cleared = t.clear_region(&region);
        assert_eq!(cleared, 3);

        // Node 50 in layer 0 remains.
        let remaining = t.dirty_in_region(&EmbeddingRegion::new(0, 0, 100));
        assert_eq!(remaining, vec![50]);
        // Layer 1 untouched.
        assert_eq!(
            t.dirty_in_region(&EmbeddingRegion::new(1, 0, 100)),
            vec![10]
        );
    }

    #[test]
    fn test_dirty_tracker_dirty_ratio() {
        let t = DirtyRegionTracker::new(10);
        assert_eq!(t.dirty_ratio(), 0.0);
        t.mark_dirty_batch(0, &[0, 1, 2, 3, 4]);
        assert!((t.dirty_ratio() - 0.5).abs() < 1e-9);
        // Exceeding max should clamp at 1.0.
        t.mark_dirty_batch(0, &[5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(t.dirty_ratio(), 1.0);
    }

    #[test]
    fn test_dirty_tracker_advance_generation() {
        let t = DirtyRegionTracker::new(100);
        assert_eq!(t.generation(), 0);
        let g1 = t.advance_generation();
        assert_eq!(g1, 1);
        assert_eq!(t.generation(), 1);
        let g2 = t.advance_generation();
        assert_eq!(g2, 2);
    }

    #[test]
    fn test_dirty_tracker_dirty_layers() {
        let t = DirtyRegionTracker::new(100);
        assert!(t.dirty_layers().is_empty());
        t.mark_dirty(2, 5);
        t.mark_dirty(0, 1);
        t.mark_dirty(1, 3);
        assert_eq!(t.dirty_layers(), vec![0, 1, 2]);

        // Clearing all nodes from layer 1 removes it from dirty_layers.
        t.clear_region(&EmbeddingRegion::new(1, 0, u64::MAX));
        assert_eq!(t.dirty_layers(), vec![0, 2]);
    }

    // ── PartialSyncManager ────────────────────────────────────────────────────

    #[test]
    fn test_partial_sync_record_change() {
        let mgr = PartialSyncManager::new(100, 16, "local");
        mgr.record_change(0, 7);
        mgr.record_change(0, 8);
        mgr.record_change(1, 3);
        assert_eq!(mgr.tracker().total_dirty(), 3);
        assert!(mgr.tracker().has_dirty());
    }

    #[test]
    fn test_partial_sync_build_delta() {
        let mgr = PartialSyncManager::new(100, 16, "local");
        mgr.record_change(0, 10);
        mgr.record_change(0, 20);
        mgr.record_change(0, 30);

        let region = EmbeddingRegion::new(0, 0, 100);
        let delta = mgr.build_delta(&region, |node_id| Some(vec![node_id as f32; 4]), 1000);

        assert_eq!(delta.change_count(), 3);
        assert_eq!(delta.source_peer, "local");
        assert_eq!(delta.created_at_ms, 1000);
        // All three node IDs should be present.
        let mut ids = delta.changed_ids.clone();
        ids.sort_unstable();
        assert_eq!(ids, vec![10, 20, 30]);
    }

    #[test]
    fn test_partial_sync_build_all_deltas() {
        let mgr = PartialSyncManager::new(100, 16, "local");
        // Dirty in layer 0.
        mgr.record_change(0, 1);
        mgr.record_change(0, 2);
        // Dirty in layer 1.
        mgr.record_change(1, 99);

        let deltas = mgr.build_all_deltas(|_layer, node_id| Some(vec![node_id as f32; 3]), 2000);

        // One delta per dirty layer.
        assert_eq!(deltas.len(), 2);

        // Find each layer's delta and verify counts.
        let delta_l0 = deltas
            .iter()
            .find(|d| d.region.layer == 0)
            .expect("layer 0 delta");
        let delta_l1 = deltas
            .iter()
            .find(|d| d.region.layer == 1)
            .expect("layer 1 delta");
        assert_eq!(delta_l0.change_count(), 2);
        assert_eq!(delta_l1.change_count(), 1);
    }

    #[test]
    fn test_partial_sync_apply_delta_clears_dirty() {
        let mgr = PartialSyncManager::new(100, 16, "local");
        mgr.record_change(0, 5);
        mgr.record_change(0, 6);
        mgr.record_change(0, 7);

        // Build and apply an incoming delta that covers nodes 5 and 6.
        let region = EmbeddingRegion::new(0, 0, 100);
        let mut incoming = EmbeddingDelta::new(region, 1, "remote-peer", 999);
        incoming.add_change(5, vec![1.0; 3]);
        incoming.add_change(6, vec![2.0; 3]);

        mgr.apply_delta(&incoming);

        // Node 7 should still be dirty; 5 and 6 cleared.
        assert_eq!(mgr.tracker().total_dirty(), 1);
        let remaining = mgr
            .tracker()
            .dirty_in_region(&EmbeddingRegion::new(0, 0, 100));
        assert_eq!(remaining, vec![7]);
    }

    #[test]
    fn test_partial_sync_ack_generation() {
        let mgr = PartialSyncManager::new(100, 16, "local");
        let region = EmbeddingRegion::new(0, 0, 100);

        // Push deltas at generations 1, 2, 3, 4.
        for gen in 1u64..=4 {
            let delta = EmbeddingDelta::new(region.clone(), gen, "local", gen * 1000);
            assert!(mgr.push_pending(delta));
        }
        assert_eq!(mgr.pending_count(), 4);

        // Ack up to generation 2 → removes deltas with gen ≤ 2.
        let removed = mgr.ack_generation(2);
        assert_eq!(removed, 2);
        assert_eq!(mgr.pending_count(), 2);

        // Ack everything.
        let removed2 = mgr.ack_generation(u64::MAX);
        assert_eq!(removed2, 2);
        assert_eq!(mgr.pending_count(), 0);
    }
}
