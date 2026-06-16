//! Semantic Cluster Manager — online k-means-style document clustering over embeddings.
//!
//! Manages semantic embedding clusters, assigns documents to clusters, updates centroids
//! incrementally via exponential moving average, and detects cluster drift over time.

/// Assignment of a document to a cluster.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterAssignment {
    /// Unique identifier of the document.
    pub doc_id: u64,
    /// The cluster this document was assigned to.
    pub cluster_id: usize,
    /// Euclidean distance from the document embedding to the cluster centroid.
    pub distance: f32,
}

/// A single semantic cluster with a centroid maintained via incremental updates.
#[derive(Debug, Clone)]
pub struct SemanticCluster {
    /// Cluster index (0-based).
    pub cluster_id: usize,
    /// Current centroid vector (mean embedding direction).
    pub centroid: Vec<f32>,
    /// Number of documents that have been assigned to and updated this cluster.
    pub member_count: u64,
    /// Cumulative Euclidean shift of the centroid across all updates.
    pub total_drift: f32,
}

impl SemanticCluster {
    /// Compute Euclidean distance between the centroid and `embedding`.
    ///
    /// Returns `0.0` if either vector is empty or if the dimensions differ.
    pub fn euclidean_distance(&self, embedding: &[f32]) -> f32 {
        if self.centroid.is_empty() || embedding.is_empty() {
            return 0.0;
        }
        if self.centroid.len() != embedding.len() {
            return 0.0;
        }
        let sum_sq: f32 = self
            .centroid
            .iter()
            .zip(embedding.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        sum_sq.sqrt()
    }

    /// Update the centroid toward `new_embedding` using an exponential moving average.
    ///
    /// Formula: `centroid[i] = (1 - learning_rate) * centroid[i] + learning_rate * new_embedding[i]`
    ///
    /// Computes the drift (Euclidean shift of the centroid) and adds it to `total_drift`.
    /// Increments `member_count` by one.
    ///
    /// This is a no-op if the dimensions of `new_embedding` do not match the centroid.
    pub fn update_centroid(&mut self, new_embedding: &[f32], learning_rate: f32) {
        if self.centroid.len() != new_embedding.len() {
            return;
        }
        if self.centroid.is_empty() {
            return;
        }

        let old_centroid = self.centroid.clone();

        for (c, &e) in self.centroid.iter_mut().zip(new_embedding.iter()) {
            *c = (1.0 - learning_rate) * (*c) + learning_rate * e;
        }

        // Compute drift = Euclidean distance between old and new centroid.
        let drift: f32 = old_centroid
            .iter()
            .zip(self.centroid.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();

        self.total_drift += drift;
        self.member_count += 1;
    }
}

/// Configuration for [`SemanticClusterManager`].
#[derive(Debug, Clone)]
pub struct ClusterManagerConfig {
    /// Number of clusters to maintain.
    pub n_clusters: usize,
    /// Centroid update learning rate (EMA weight for the new embedding).
    pub learning_rate: f32,
    /// Total drift threshold above which a cluster is considered unstable.
    pub drift_threshold: f32,
}

impl Default for ClusterManagerConfig {
    fn default() -> Self {
        Self {
            n_clusters: 8,
            learning_rate: 0.1,
            drift_threshold: 0.5,
        }
    }
}

/// Aggregate statistics over all clusters managed by [`SemanticClusterManager`].
#[derive(Debug, Clone)]
pub struct ClusterManagerStats {
    /// Total number of clusters.
    pub total_clusters: usize,
    /// Total number of document assignments performed.
    pub total_assignments: u64,
    /// Mean member count across all clusters.
    pub avg_cluster_size: f64,
    /// `cluster_id` of the cluster with the highest `total_drift`, or `None` if all have zero drift.
    pub most_drifted_cluster: Option<usize>,
    /// Number of clusters whose `total_drift` exceeds the configured threshold.
    pub unstable_clusters: usize,
}

/// Online k-means-style semantic cluster manager.
///
/// Documents are assigned to the nearest initialised cluster (by Euclidean distance) and each
/// assignment incrementally shifts the cluster centroid toward the document embedding.
pub struct SemanticClusterManager {
    /// The managed clusters (length == `config.n_clusters`).
    pub clusters: Vec<SemanticCluster>,
    /// Manager configuration.
    pub config: ClusterManagerConfig,
    /// Total number of successful assignments since creation or last reset.
    pub total_assignments: u64,
}

impl SemanticClusterManager {
    /// Create a new manager with `n_clusters` empty (uninitialized) clusters.
    pub fn new(config: ClusterManagerConfig) -> Self {
        let n = config.n_clusters;
        let clusters = (0..n)
            .map(|i| SemanticCluster {
                cluster_id: i,
                centroid: Vec::new(),
                member_count: 0,
                total_drift: 0.0,
            })
            .collect();
        Self {
            clusters,
            config,
            total_assignments: 0,
        }
    }

    /// Initialise cluster centroids from the provided list.
    ///
    /// Only up to `min(centroids.len(), n_clusters)` clusters are initialised.
    pub fn initialize_centroids(&mut self, centroids: Vec<Vec<f32>>) {
        let limit = centroids.len().min(self.config.n_clusters);
        for (i, centroid) in centroids.into_iter().take(limit).enumerate() {
            self.clusters[i].centroid = centroid;
        }
    }

    /// Assign `doc_id` to the nearest initialised cluster and update the centroid.
    ///
    /// A cluster is considered initialised when its centroid is non-empty **and** its
    /// dimension matches that of `embedding`.
    ///
    /// Returns `None` when no valid cluster exists.
    pub fn assign(&mut self, doc_id: u64, embedding: &[f32]) -> Option<ClusterAssignment> {
        // Find the nearest valid cluster.
        let mut best_cluster_id: Option<usize> = None;
        let mut best_distance = f32::MAX;

        for cluster in &self.clusters {
            if cluster.centroid.is_empty() || cluster.centroid.len() != embedding.len() {
                continue;
            }
            let dist = cluster.euclidean_distance(embedding);
            if dist < best_distance {
                best_distance = dist;
                best_cluster_id = Some(cluster.cluster_id);
            }
        }

        let cluster_id = best_cluster_id?;

        // Update centroid in place.
        let lr = self.config.learning_rate;
        self.clusters[cluster_id].update_centroid(embedding, lr);

        self.total_assignments += 1;

        Some(ClusterAssignment {
            doc_id,
            cluster_id,
            distance: best_distance,
        })
    }

    /// Return the `cluster_id` of the nearest initialised cluster to `embedding` without
    /// mutating any state.
    pub fn nearest_cluster(&self, embedding: &[f32]) -> Option<usize> {
        let mut best_cluster_id: Option<usize> = None;
        let mut best_distance = f32::MAX;

        for cluster in &self.clusters {
            if cluster.centroid.is_empty() || cluster.centroid.len() != embedding.len() {
                continue;
            }
            let dist = cluster.euclidean_distance(embedding);
            if dist < best_distance {
                best_distance = dist;
                best_cluster_id = Some(cluster.cluster_id);
            }
        }

        best_cluster_id
    }

    /// Return a reference to a cluster by its `cluster_id`, or `None` if out of bounds.
    pub fn cluster(&self, cluster_id: usize) -> Option<&SemanticCluster> {
        self.clusters.get(cluster_id)
    }

    /// Compute aggregate statistics over all clusters.
    pub fn stats(&self) -> ClusterManagerStats {
        let total_clusters = self.clusters.len();

        let avg_cluster_size = if total_clusters == 0 {
            0.0
        } else {
            let total_members: u64 = self.clusters.iter().map(|c| c.member_count).sum();
            total_members as f64 / total_clusters as f64
        };

        // Find the cluster with the highest total_drift.
        let most_drifted_cluster = self
            .clusters
            .iter()
            .max_by(|a, b| {
                a.total_drift
                    .partial_cmp(&b.total_drift)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .and_then(|c| {
                if c.total_drift > 0.0 {
                    Some(c.cluster_id)
                } else {
                    None
                }
            });

        let threshold = self.config.drift_threshold;
        let unstable_clusters = self
            .clusters
            .iter()
            .filter(|c| c.total_drift > threshold)
            .count();

        ClusterManagerStats {
            total_clusters,
            total_assignments: self.total_assignments,
            avg_cluster_size,
            most_drifted_cluster,
            unstable_clusters,
        }
    }

    /// Reset `total_drift` to `0.0` for all clusters.
    pub fn reset_drift(&mut self) {
        for cluster in &mut self.clusters {
            cluster.total_drift = 0.0;
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(n_clusters: usize) -> ClusterManagerConfig {
        ClusterManagerConfig {
            n_clusters,
            learning_rate: 0.1,
            drift_threshold: 0.5,
        }
    }

    fn make_manager(n_clusters: usize) -> SemanticClusterManager {
        SemanticClusterManager::new(make_config(n_clusters))
    }

    // ── SemanticCluster helpers ──────────────────────────────────────────────

    #[test]
    fn test_euclidean_distance_basic() {
        let cluster = SemanticCluster {
            cluster_id: 0,
            centroid: vec![0.0, 0.0, 0.0],
            member_count: 0,
            total_drift: 0.0,
        };
        let dist = cluster.euclidean_distance(&[3.0, 4.0, 0.0]);
        assert!((dist - 5.0).abs() < 1e-5, "expected 5.0, got {dist}");
    }

    #[test]
    fn test_euclidean_distance_zero_when_same() {
        let cluster = SemanticCluster {
            cluster_id: 0,
            centroid: vec![1.0, 2.0, 3.0],
            member_count: 0,
            total_drift: 0.0,
        };
        let dist = cluster.euclidean_distance(&[1.0, 2.0, 3.0]);
        assert!(dist.abs() < 1e-6, "expected 0.0, got {dist}");
    }

    #[test]
    fn test_euclidean_distance_empty_centroid_returns_zero() {
        let cluster = SemanticCluster {
            cluster_id: 0,
            centroid: vec![],
            member_count: 0,
            total_drift: 0.0,
        };
        let dist = cluster.euclidean_distance(&[1.0, 2.0]);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_euclidean_distance_empty_embedding_returns_zero() {
        let cluster = SemanticCluster {
            cluster_id: 0,
            centroid: vec![1.0, 2.0],
            member_count: 0,
            total_drift: 0.0,
        };
        let dist = cluster.euclidean_distance(&[]);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_euclidean_distance_dimension_mismatch_returns_zero() {
        let cluster = SemanticCluster {
            cluster_id: 0,
            centroid: vec![1.0, 2.0],
            member_count: 0,
            total_drift: 0.0,
        };
        let dist = cluster.euclidean_distance(&[1.0, 2.0, 3.0]);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_update_centroid_shifts_toward_embedding() {
        let mut cluster = SemanticCluster {
            cluster_id: 0,
            centroid: vec![0.0, 0.0],
            member_count: 0,
            total_drift: 0.0,
        };
        // With lr=1.0 the centroid must equal the new embedding exactly.
        cluster.update_centroid(&[10.0, 10.0], 1.0);
        assert!((cluster.centroid[0] - 10.0).abs() < 1e-5);
        assert!((cluster.centroid[1] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_update_centroid_ema_formula() {
        let mut cluster = SemanticCluster {
            cluster_id: 0,
            centroid: vec![0.0],
            member_count: 0,
            total_drift: 0.0,
        };
        cluster.update_centroid(&[1.0], 0.1);
        // centroid[0] = 0.9*0.0 + 0.1*1.0 = 0.1
        assert!((cluster.centroid[0] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_update_centroid_increments_member_count() {
        let mut cluster = SemanticCluster {
            cluster_id: 0,
            centroid: vec![0.0],
            member_count: 5,
            total_drift: 0.0,
        };
        cluster.update_centroid(&[1.0], 0.1);
        assert_eq!(cluster.member_count, 6);
    }

    #[test]
    fn test_update_centroid_accumulates_drift() {
        let mut cluster = SemanticCluster {
            cluster_id: 0,
            centroid: vec![0.0, 0.0],
            member_count: 0,
            total_drift: 0.0,
        };
        cluster.update_centroid(&[10.0, 0.0], 0.1);
        // drift after first update = |new_centroid - old_centroid| = |[1.0, 0.0] - [0.0, 0.0]| = 1.0
        assert!(
            (cluster.total_drift - 1.0).abs() < 1e-5,
            "drift={}",
            cluster.total_drift
        );

        cluster.update_centroid(&[10.0, 0.0], 0.1);
        // centroid now [1.0, 0.0] → [1.9, 0.0], drift addition = 0.9
        assert!(cluster.total_drift > 1.0);
    }

    #[test]
    fn test_update_centroid_noop_on_dimension_mismatch() {
        let mut cluster = SemanticCluster {
            cluster_id: 0,
            centroid: vec![1.0, 2.0],
            member_count: 3,
            total_drift: 0.5,
        };
        cluster.update_centroid(&[9.0, 9.0, 9.0], 0.5);
        // Nothing should change.
        assert_eq!(cluster.member_count, 3);
        assert!((cluster.total_drift - 0.5).abs() < 1e-6);
        assert!((cluster.centroid[0] - 1.0).abs() < 1e-6);
    }

    // ── SemanticClusterManager ───────────────────────────────────────────────

    #[test]
    fn test_new_creates_correct_number_of_clusters() {
        let mgr = make_manager(5);
        assert_eq!(mgr.clusters.len(), 5);
        for (i, c) in mgr.clusters.iter().enumerate() {
            assert_eq!(c.cluster_id, i);
            assert!(c.centroid.is_empty());
            assert_eq!(c.member_count, 0);
        }
    }

    #[test]
    fn test_initialize_centroids_sets_centroids() {
        let mut mgr = make_manager(3);
        let centroids = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        mgr.initialize_centroids(centroids.clone());
        for (i, c) in centroids.iter().enumerate() {
            assert_eq!(&mgr.clusters[i].centroid, c);
        }
    }

    #[test]
    fn test_initialize_centroids_more_than_n_clusters_is_clamped() {
        let mut mgr = make_manager(2);
        let centroids = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        mgr.initialize_centroids(centroids);
        // Only first 2 should be set.
        assert_eq!(mgr.clusters[0].centroid, vec![1.0]);
        assert_eq!(mgr.clusters[1].centroid, vec![2.0]);
    }

    #[test]
    fn test_assign_returns_none_when_no_initialised_clusters() {
        let mut mgr = make_manager(3);
        let result = mgr.assign(42, &[0.5, 0.5]);
        assert!(result.is_none());
    }

    #[test]
    fn test_assign_returns_none_on_dimension_mismatch() {
        let mut mgr = make_manager(2);
        mgr.initialize_centroids(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        // Embedding has different dimension.
        let result = mgr.assign(1, &[0.5, 0.5, 0.5]);
        assert!(result.is_none());
    }

    #[test]
    fn test_assign_returns_nearest_cluster() {
        let mut mgr = make_manager(2);
        // Cluster 0 centroid at (1, 0), cluster 1 at (0, 1).
        mgr.initialize_centroids(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);

        // Embedding close to cluster 0.
        let result = mgr.assign(1, &[0.9, 0.1]).expect("should assign");
        assert_eq!(result.cluster_id, 0);

        // Embedding close to cluster 1.
        let result2 = mgr.assign(2, &[0.1, 0.9]).expect("should assign");
        assert_eq!(result2.cluster_id, 1);
    }

    #[test]
    fn test_assign_distance_is_correct() {
        let mut mgr = make_manager(1);
        mgr.initialize_centroids(vec![vec![0.0, 0.0, 0.0]]);
        let result = mgr.assign(1, &[3.0, 4.0, 0.0]).expect("should assign");
        // Euclidean distance from origin = 5.
        assert!(
            (result.distance - 5.0).abs() < 1e-5,
            "dist={}",
            result.distance
        );
    }

    #[test]
    fn test_assign_increments_total_assignments() {
        let mut mgr = make_manager(1);
        mgr.initialize_centroids(vec![vec![0.0]]);
        assert_eq!(mgr.total_assignments, 0);
        mgr.assign(1, &[1.0]);
        assert_eq!(mgr.total_assignments, 1);
        mgr.assign(2, &[1.0]);
        assert_eq!(mgr.total_assignments, 2);
    }

    #[test]
    fn test_assign_increments_cluster_member_count() {
        let mut mgr = make_manager(1);
        mgr.initialize_centroids(vec![vec![0.0]]);
        mgr.assign(1, &[1.0]);
        mgr.assign(2, &[2.0]);
        assert_eq!(mgr.clusters[0].member_count, 2);
    }

    #[test]
    fn test_assign_updates_centroid() {
        let mut mgr = make_manager(1);
        mgr.initialize_centroids(vec![vec![0.0]]);
        mgr.assign(1, &[1.0]);
        // centroid[0] = 0.9*0.0 + 0.1*1.0 = 0.1
        assert!((mgr.clusters[0].centroid[0] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_nearest_cluster_no_mutation() {
        let mut mgr = make_manager(2);
        mgr.initialize_centroids(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        let before_count = mgr.clusters[0].member_count;
        let id = mgr.nearest_cluster(&[0.9, 0.1]).expect("should find");
        assert_eq!(id, 0);
        assert_eq!(
            mgr.clusters[0].member_count, before_count,
            "nearest_cluster must not mutate"
        );
        assert_eq!(mgr.total_assignments, 0);
    }

    #[test]
    fn test_nearest_cluster_returns_none_for_uninitialised() {
        let mgr = make_manager(3);
        let result = mgr.nearest_cluster(&[0.5]);
        assert!(result.is_none());
    }

    #[test]
    fn test_nearest_cluster_returns_correct_id() {
        let mut mgr = make_manager(3);
        mgr.initialize_centroids(vec![vec![10.0, 0.0], vec![0.0, 10.0], vec![5.0, 5.0]]);
        // Closest to cluster 2.
        let id = mgr.nearest_cluster(&[5.1, 5.1]).expect("should find");
        assert_eq!(id, 2);
    }

    #[test]
    fn test_cluster_getter() {
        let mut mgr = make_manager(3);
        mgr.initialize_centroids(vec![vec![7.0]]);
        let c = mgr.cluster(0).expect("cluster 0 exists");
        assert_eq!(c.centroid, vec![7.0]);
        assert!(mgr.cluster(100).is_none());
    }

    #[test]
    fn test_stats_total_clusters() {
        let mgr = make_manager(4);
        assert_eq!(mgr.stats().total_clusters, 4);
    }

    #[test]
    fn test_stats_total_assignments() {
        let mut mgr = make_manager(1);
        mgr.initialize_centroids(vec![vec![0.0]]);
        mgr.assign(1, &[1.0]);
        mgr.assign(2, &[2.0]);
        assert_eq!(mgr.stats().total_assignments, 2);
    }

    #[test]
    fn test_stats_avg_cluster_size() {
        let mut mgr = make_manager(2);
        mgr.initialize_centroids(vec![vec![0.0], vec![10.0]]);
        // 3 assignments to cluster 0, 1 to cluster 1.
        mgr.assign(1, &[0.1]);
        mgr.assign(2, &[0.2]);
        mgr.assign(3, &[0.3]);
        mgr.assign(4, &[9.9]);
        let stats = mgr.stats();
        // avg = (3 + 1) / 2 = 2.0
        assert!(
            (stats.avg_cluster_size - 2.0).abs() < 1e-6,
            "avg={}",
            stats.avg_cluster_size
        );
    }

    #[test]
    fn test_stats_most_drifted_cluster() {
        let mut mgr = make_manager(2);
        mgr.initialize_centroids(vec![vec![0.0], vec![100.0]]);
        // Make cluster 1 drift more by sending a very different embedding.
        for _ in 0..10 {
            mgr.assign(99, &[0.0]); // near cluster 0, low drift on cluster 0
        }
        // Force large drift on cluster 1 by manually setting after the test assignments.
        mgr.clusters[1].total_drift = 99.0;
        let stats = mgr.stats();
        assert_eq!(stats.most_drifted_cluster, Some(1));
    }

    #[test]
    fn test_stats_most_drifted_cluster_none_when_all_zero() {
        let mgr = make_manager(3);
        let stats = mgr.stats();
        assert!(stats.most_drifted_cluster.is_none());
    }

    #[test]
    fn test_stats_unstable_clusters() {
        let mut mgr = make_manager(3);
        mgr.initialize_centroids(vec![vec![0.0], vec![5.0], vec![10.0]]);
        mgr.clusters[0].total_drift = 0.1;
        mgr.clusters[1].total_drift = 0.6; // above 0.5
        mgr.clusters[2].total_drift = 1.5; // above 0.5
        let stats = mgr.stats();
        assert_eq!(stats.unstable_clusters, 2);
    }

    #[test]
    fn test_reset_drift_zeroes_all() {
        let mut mgr = make_manager(3);
        mgr.clusters[0].total_drift = 1.0;
        mgr.clusters[1].total_drift = 2.0;
        mgr.clusters[2].total_drift = 3.0;
        mgr.reset_drift();
        for c in &mgr.clusters {
            assert_eq!(c.total_drift, 0.0);
        }
    }

    #[test]
    fn test_default_config() {
        let cfg = ClusterManagerConfig::default();
        assert_eq!(cfg.n_clusters, 8);
        assert!((cfg.learning_rate - 0.1).abs() < 1e-6);
        assert!((cfg.drift_threshold - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_assign_doc_id_preserved() {
        let mut mgr = make_manager(1);
        mgr.initialize_centroids(vec![vec![0.0]]);
        let result = mgr.assign(12345, &[0.5]).expect("should assign");
        assert_eq!(result.doc_id, 12345);
    }

    #[test]
    fn test_total_drift_accumulates_over_multiple_assigns() {
        let mut mgr = make_manager(1);
        mgr.initialize_centroids(vec![vec![0.0, 0.0]]);
        mgr.assign(1, &[10.0, 10.0]);
        mgr.assign(2, &[10.0, 10.0]);
        mgr.assign(3, &[10.0, 10.0]);
        assert!(mgr.clusters[0].total_drift > 0.0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Batch K-Means Semantic Cluster Manager
// ═══════════════════════════════════════════════════════════════════════════════

use std::collections::HashMap;

/// Configuration for batch k-means clustering.
#[derive(Debug, Clone)]
pub struct BatchClusterConfig {
    /// Number of clusters (k). Default: 8.
    pub num_clusters: usize,
    /// Maximum number of k-means iterations. Default: 100.
    pub max_iterations: usize,
    /// Convergence threshold for centroid movement (L2). Default: 1e-6.
    pub convergence_threshold: f64,
    /// Dimensionality of embedding vectors.
    pub embedding_dim: usize,
}

impl Default for BatchClusterConfig {
    fn default() -> Self {
        Self {
            num_clusters: 8,
            max_iterations: 100,
            convergence_threshold: 1e-6,
            embedding_dim: 0,
        }
    }
}

/// A cluster produced by batch k-means.
#[derive(Debug, Clone)]
pub struct BatchCluster {
    /// Cluster index (0-based).
    pub id: usize,
    /// Centroid vector (mean of member embeddings).
    pub centroid: Vec<f64>,
    /// Number of members assigned to this cluster.
    pub member_count: usize,
    /// Sum of squared Euclidean distances from each member to the centroid (inertia).
    pub inertia: f64,
}

/// Aggregate statistics for [`BatchSemanticClusterManager`].
#[derive(Debug, Clone)]
pub struct BatchClusterManagerStats {
    /// Number of clusters.
    pub num_clusters: usize,
    /// Total number of members across all clusters.
    pub total_members: usize,
    /// Number of iterations the last `fit` executed.
    pub iterations_run: usize,
    /// Whether the last `fit` converged before reaching `max_iterations`.
    pub converged: bool,
    /// Sum of inertias across all clusters.
    pub total_inertia: f64,
}

/// Batch k-means semantic cluster manager.
///
/// Runs Lloyd's k-means algorithm over a set of named embeddings, then supports
/// prediction (nearest centroid lookup), membership queries, and quality metrics.
pub struct BatchSemanticClusterManager {
    config: BatchClusterConfig,
    clusters: Vec<BatchCluster>,
    assignments: HashMap<String, usize>,
    iterations_run: usize,
    converged: bool,
}

/// Compute Euclidean distance between two vectors of equal length.
///
/// Returns 0.0 if lengths differ or either is empty.
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Compute element-wise mean of a set of vectors.
///
/// Returns an empty vector if the input is empty.
pub fn vec_mean(vectors: &[&[f64]]) -> Vec<f64> {
    if vectors.is_empty() {
        return Vec::new();
    }
    let dim = vectors[0].len();
    let n = vectors.len() as f64;
    let mut mean = vec![0.0; dim];
    for v in vectors {
        for (i, &val) in v.iter().enumerate() {
            if i < dim {
                mean[i] += val;
            }
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    mean
}

impl BatchSemanticClusterManager {
    /// Create a new (unfitted) batch cluster manager.
    pub fn new(config: BatchClusterConfig) -> Self {
        Self {
            config,
            clusters: Vec::new(),
            assignments: HashMap::new(),
            iterations_run: 0,
            converged: false,
        }
    }

    /// Run k-means clustering on the provided embeddings.
    ///
    /// Embeddings are `(doc_id, vector)` pairs.  The method initialises centroids
    /// from the first `k` distinct embeddings and then iterates assignment + update
    /// steps until convergence or `max_iterations` is reached.
    ///
    /// Calling `fit` again resets all previous state.
    pub fn fit(&mut self, embeddings: &[(String, Vec<f64>)]) -> Result<(), String> {
        let k = self.config.num_clusters;

        if embeddings.is_empty() {
            return Err("cannot fit on empty embeddings".to_string());
        }
        if embeddings.len() < k {
            return Err(format!(
                "need at least {} embeddings but got {}",
                k,
                embeddings.len()
            ));
        }

        // Validate dimensions.
        let dim = if self.config.embedding_dim > 0 {
            self.config.embedding_dim
        } else {
            embeddings[0].1.len()
        };
        if dim == 0 {
            return Err("embedding dimension is zero".to_string());
        }
        for (id, v) in embeddings {
            if v.len() != dim {
                return Err(format!(
                    "embedding '{}' has dimension {} but expected {}",
                    id,
                    v.len(),
                    dim
                ));
            }
        }

        // Reset state.
        self.clusters.clear();
        self.assignments.clear();
        self.iterations_run = 0;
        self.converged = false;

        // Initialise centroids from first k distinct embeddings.
        let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
        for (_id, v) in embeddings {
            let is_dup = centroids.iter().any(|c| {
                c.iter()
                    .zip(v.iter())
                    .all(|(a, b)| (a - b).abs() < f64::EPSILON)
            });
            if !is_dup {
                centroids.push(v.clone());
            }
            if centroids.len() == k {
                break;
            }
        }
        if centroids.len() < k {
            return Err(format!(
                "need at least {} distinct embeddings but found only {}",
                k,
                centroids.len()
            ));
        }

        let mut assignments: Vec<usize> = vec![0; embeddings.len()];

        for iter in 0..self.config.max_iterations {
            // ── Assignment step ──────────────────────────────────────────────
            for (idx, (_id, v)) in embeddings.iter().enumerate() {
                let mut best_cluster = 0;
                let mut best_dist = f64::MAX;
                for (ci, c) in centroids.iter().enumerate() {
                    let d = euclidean_distance(v, c);
                    if d < best_dist {
                        best_dist = d;
                        best_cluster = ci;
                    }
                }
                assignments[idx] = best_cluster;
            }

            // ── Update step ──────────────────────────────────────────────────
            let mut new_centroids = vec![vec![0.0; dim]; k];
            let mut counts = vec![0usize; k];

            for (idx, (_id, v)) in embeddings.iter().enumerate() {
                let ci = assignments[idx];
                counts[ci] += 1;
                for (j, &val) in v.iter().enumerate() {
                    new_centroids[ci][j] += val;
                }
            }

            for ci in 0..k {
                if counts[ci] > 0 {
                    let n = counts[ci] as f64;
                    for val in new_centroids[ci].iter_mut() {
                        *val /= n;
                    }
                } else {
                    // Keep old centroid for empty clusters.
                    new_centroids[ci] = centroids[ci].clone();
                }
            }

            // ── Check convergence ────────────────────────────────────────────
            let max_movement = centroids
                .iter()
                .zip(new_centroids.iter())
                .map(|(old, new)| euclidean_distance(old, new))
                .fold(0.0_f64, f64::max);

            centroids = new_centroids;
            self.iterations_run = iter + 1;

            if max_movement < self.config.convergence_threshold {
                self.converged = true;
                break;
            }
        }

        // ── Build final clusters and assignments ─────────────────────────────
        // Re-run assignment with final centroids to ensure consistency.
        for (idx, (_id, v)) in embeddings.iter().enumerate() {
            let mut best_cluster = 0;
            let mut best_dist = f64::MAX;
            for (ci, c) in centroids.iter().enumerate() {
                let d = euclidean_distance(v, c);
                if d < best_dist {
                    best_dist = d;
                    best_cluster = ci;
                }
            }
            assignments[idx] = best_cluster;
        }

        // Compute inertia per cluster.
        let mut inertias = vec![0.0_f64; k];
        let mut member_counts = vec![0usize; k];
        for (idx, (_id, v)) in embeddings.iter().enumerate() {
            let ci = assignments[idx];
            member_counts[ci] += 1;
            let d = euclidean_distance(v, &centroids[ci]);
            inertias[ci] += d * d;
        }

        self.clusters = (0..k)
            .map(|ci| BatchCluster {
                id: ci,
                centroid: centroids[ci].clone(),
                member_count: member_counts[ci],
                inertia: inertias[ci],
            })
            .collect();

        for (idx, (id, _v)) in embeddings.iter().enumerate() {
            self.assignments.insert(id.clone(), assignments[idx]);
        }

        Ok(())
    }

    /// Predict which cluster a new embedding belongs to (nearest centroid).
    ///
    /// Returns an error if the manager has not been fitted yet.
    pub fn predict(&self, embedding: &[f64]) -> Result<usize, String> {
        if self.clusters.is_empty() {
            return Err("not fitted yet".to_string());
        }

        let mut best_cluster = 0;
        let mut best_dist = f64::MAX;
        for cluster in &self.clusters {
            let d = euclidean_distance(embedding, &cluster.centroid);
            if d < best_dist {
                best_dist = d;
                best_cluster = cluster.id;
            }
        }
        Ok(best_cluster)
    }

    /// Return a reference to a cluster by id, or `None` if out of range.
    pub fn get_cluster(&self, id: usize) -> Option<&BatchCluster> {
        self.clusters.get(id)
    }

    /// Return the cluster assignment for a document, or `None` if unknown.
    pub fn get_assignment(&self, doc_id: &str) -> Option<usize> {
        self.assignments.get(doc_id).copied()
    }

    /// Return the list of document IDs assigned to a given cluster.
    pub fn cluster_members(&self, cluster_id: usize) -> Vec<String> {
        self.assignments
            .iter()
            .filter(|(_id, &cid)| cid == cluster_id)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Total inertia (sum of squared distances) across all clusters.
    pub fn total_inertia(&self) -> f64 {
        self.clusters.iter().map(|c| c.inertia).sum()
    }

    /// Approximate silhouette score.
    ///
    /// For each point, computes `(b - a) / max(a, b)` where:
    /// - `a` = average Euclidean distance to other members of the same cluster
    /// - `b` = average distance to the nearest **other** cluster centroid
    ///
    /// Returns the mean silhouette across all points.
    pub fn silhouette_score_approx(
        &self,
        embeddings: &[(String, Vec<f64>)],
    ) -> Result<f64, String> {
        if self.clusters.is_empty() {
            return Err("not fitted yet".to_string());
        }
        if embeddings.is_empty() {
            return Err("no embeddings provided".to_string());
        }
        if self.clusters.len() < 2 {
            // Silhouette is undefined for a single cluster.
            return Ok(0.0);
        }

        // Build per-cluster embedding lists.
        let k = self.clusters.len();
        let mut cluster_vecs: Vec<Vec<&[f64]>> = vec![Vec::new(); k];
        for (id, v) in embeddings {
            if let Some(&ci) = self.assignments.get(id) {
                if ci < k {
                    cluster_vecs[ci].push(v.as_slice());
                }
            }
        }

        let mut total_sil = 0.0_f64;
        let mut count = 0usize;

        for (id, v) in embeddings {
            let ci = match self.assignments.get(id) {
                Some(&c) => c,
                None => continue,
            };

            // a = average distance to own cluster members (excluding self).
            let own_members = &cluster_vecs[ci];
            let a = if own_members.len() <= 1 {
                0.0
            } else {
                let sum: f64 = own_members.iter().map(|m| euclidean_distance(v, m)).sum();
                // sum includes distance to self (0), so denominator is len-1.
                sum / (own_members.len() - 1) as f64
            };

            // b = min over other clusters of distance to that cluster's centroid.
            let mut b = f64::MAX;
            for cluster in &self.clusters {
                if cluster.id == ci {
                    continue;
                }
                let d = euclidean_distance(v, &cluster.centroid);
                if d < b {
                    b = d;
                }
            }
            if b == f64::MAX {
                b = 0.0;
            }

            let max_ab = a.max(b);
            let sil = if max_ab > 0.0 { (b - a) / max_ab } else { 0.0 };

            total_sil += sil;
            count += 1;
        }

        if count == 0 {
            return Ok(0.0);
        }
        Ok(total_sil / count as f64)
    }

    /// Return aggregate statistics about the current clustering state.
    pub fn stats(&self) -> BatchClusterManagerStats {
        BatchClusterManagerStats {
            num_clusters: self.clusters.len(),
            total_members: self.clusters.iter().map(|c| c.member_count).sum(),
            iterations_run: self.iterations_run,
            converged: self.converged,
            total_inertia: self.total_inertia(),
        }
    }
}

// ─── Batch K-Means Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod batch_tests {
    use super::*;

    fn default_config(k: usize, dim: usize) -> BatchClusterConfig {
        BatchClusterConfig {
            num_clusters: k,
            max_iterations: 100,
            convergence_threshold: 1e-6,
            embedding_dim: dim,
        }
    }

    fn well_separated_2d(n_per_cluster: usize) -> Vec<(String, Vec<f64>)> {
        // Two clusters centred at (0,0) and (100,100).
        let mut data = Vec::new();
        for i in 0..n_per_cluster {
            data.push((
                format!("a{}", i),
                vec![0.0 + i as f64 * 0.01, 0.0 + i as f64 * 0.01],
            ));
        }
        for i in 0..n_per_cluster {
            data.push((
                format!("b{}", i),
                vec![100.0 + i as f64 * 0.01, 100.0 + i as f64 * 0.01],
            ));
        }
        data
    }

    // ── 1. Fit with well-separated clusters converges ───────────────────────

    #[test]
    fn test_batch_fit_well_separated_converges() {
        let data = well_separated_2d(20);
        let mut mgr = BatchSemanticClusterManager::new(default_config(2, 2));
        mgr.fit(&data).expect("fit should succeed");
        assert!(mgr.converged, "should converge on well-separated data");
    }

    // ── 2. Predict returns correct cluster for points near centroids ────────

    #[test]
    fn test_batch_predict_near_centroids() {
        let data = well_separated_2d(20);
        let mut mgr = BatchSemanticClusterManager::new(default_config(2, 2));
        mgr.fit(&data).expect("fit should succeed");

        let c0 = mgr.predict(&[0.05, 0.05]).expect("predict");
        let c1 = mgr.predict(&[100.05, 100.05]).expect("predict");
        assert_ne!(c0, c1, "should assign to different clusters");
    }

    // ── 3. Empty embeddings returns error ───────────────────────────────────

    #[test]
    fn test_batch_fit_empty_returns_error() {
        let mut mgr = BatchSemanticClusterManager::new(default_config(2, 2));
        let result = mgr.fit(&[]);
        assert!(result.is_err());
        assert!(result.expect_err("should be error").contains("empty"),);
    }

    // ── 4. Fewer embeddings than k returns error ────────────────────────────

    #[test]
    fn test_batch_fit_too_few_embeddings() {
        let data = vec![("a".to_string(), vec![1.0, 2.0])];
        let mut mgr = BatchSemanticClusterManager::new(default_config(3, 2));
        let result = mgr.fit(&data);
        assert!(result.is_err());
    }

    // ── 5. Convergence flag set when movement < threshold ───────────────────

    #[test]
    fn test_batch_convergence_flag() {
        // Single cluster with identical points → immediate convergence.
        let data: Vec<(String, Vec<f64>)> = (0..5)
            .map(|i| (format!("d{}", i), vec![1.0, 1.0]))
            .collect();
        let mut mgr = BatchSemanticClusterManager::new(default_config(1, 2));
        mgr.fit(&data).expect("fit should succeed");
        assert!(mgr.converged);
    }

    // ── 6. Max iterations respected ─────────────────────────────────────────

    #[test]
    fn test_batch_max_iterations_respected() {
        let data = well_separated_2d(20);
        let mut cfg = default_config(2, 2);
        cfg.max_iterations = 3;
        cfg.convergence_threshold = 0.0; // never converge
        let mut mgr = BatchSemanticClusterManager::new(cfg);
        mgr.fit(&data).expect("fit should succeed");
        assert_eq!(mgr.iterations_run, 3);
        assert!(!mgr.converged);
    }

    // ── 7. Cluster members match assignments ────────────────────────────────

    #[test]
    fn test_batch_cluster_members_match() {
        let data = well_separated_2d(10);
        let mut mgr = BatchSemanticClusterManager::new(default_config(2, 2));
        mgr.fit(&data).expect("fit should succeed");

        let mut total = 0;
        for ci in 0..2 {
            let members = mgr.cluster_members(ci);
            for mid in &members {
                assert_eq!(
                    mgr.get_assignment(mid),
                    Some(ci),
                    "member {} should be in cluster {}",
                    mid,
                    ci
                );
            }
            total += members.len();
        }
        assert_eq!(total, data.len());
    }

    // ── 8. Total inertia is non-negative ────────────────────────────────────

    #[test]
    fn test_batch_total_inertia_non_negative() {
        let data = well_separated_2d(10);
        let mut mgr = BatchSemanticClusterManager::new(default_config(2, 2));
        mgr.fit(&data).expect("fit");
        assert!(mgr.total_inertia() >= 0.0);
    }

    // ── 9. Inertia decreases with more clusters ─────────────────────────────

    #[test]
    fn test_batch_inertia_decreases_with_more_clusters() {
        let data = well_separated_2d(20);

        let mut mgr1 = BatchSemanticClusterManager::new(default_config(1, 2));
        mgr1.fit(&data).expect("fit k=1");
        let inertia1 = mgr1.total_inertia();

        let mut mgr2 = BatchSemanticClusterManager::new(default_config(2, 2));
        mgr2.fit(&data).expect("fit k=2");
        let inertia2 = mgr2.total_inertia();

        assert!(
            inertia2 <= inertia1,
            "k=2 inertia ({}) should be <= k=1 inertia ({})",
            inertia2,
            inertia1
        );
    }

    // ── 10. Single-dimension embeddings work ────────────────────────────────

    #[test]
    fn test_batch_single_dimension() {
        let data: Vec<(String, Vec<f64>)> = vec![
            ("a".to_string(), vec![0.0]),
            ("b".to_string(), vec![1.0]),
            ("c".to_string(), vec![100.0]),
            ("d".to_string(), vec![101.0]),
        ];
        let mut mgr = BatchSemanticClusterManager::new(default_config(2, 1));
        mgr.fit(&data).expect("fit");
        let ca = mgr.get_assignment("a").expect("a assigned");
        let cb = mgr.get_assignment("b").expect("b assigned");
        let cc = mgr.get_assignment("c").expect("c assigned");
        let cd = mgr.get_assignment("d").expect("d assigned");
        assert_eq!(ca, cb, "a and b should be in same cluster");
        assert_eq!(cc, cd, "c and d should be in same cluster");
        assert_ne!(ca, cc, "groups should be different");
    }

    // ── 11. Multiple fits reset state ───────────────────────────────────────

    #[test]
    fn test_batch_multiple_fits_reset_state() {
        let data1 = well_separated_2d(10);
        let mut mgr = BatchSemanticClusterManager::new(default_config(2, 2));
        mgr.fit(&data1).expect("fit 1");
        let inertia1 = mgr.total_inertia();

        // Re-fit with different data.
        let data2: Vec<(String, Vec<f64>)> = (0..10)
            .map(|i| (format!("x{}", i), vec![i as f64, i as f64]))
            .collect();
        mgr.fit(&data2).expect("fit 2");

        // Old assignments should be gone.
        assert!(mgr.get_assignment("a0").is_none());
        // New assignments present.
        assert!(mgr.get_assignment("x0").is_some());
        // Inertia likely different.
        let _inertia2 = mgr.total_inertia();
        let _ = inertia1; // used above
    }

    // ── 12. get_assignment returns None for unknown doc ──────────────────────

    #[test]
    fn test_batch_get_assignment_unknown() {
        let data = well_separated_2d(5);
        let mut mgr = BatchSemanticClusterManager::new(default_config(2, 2));
        mgr.fit(&data).expect("fit");
        assert!(mgr.get_assignment("nonexistent").is_none());
    }

    // ── 13. Stats reflect current state ─────────────────────────────────────

    #[test]
    fn test_batch_stats_reflect_state() {
        let data = well_separated_2d(10);
        let mut mgr = BatchSemanticClusterManager::new(default_config(2, 2));
        mgr.fit(&data).expect("fit");

        let stats = mgr.stats();
        assert_eq!(stats.num_clusters, 2);
        assert_eq!(stats.total_members, 20);
        assert!(stats.iterations_run > 0);
        assert!(stats.total_inertia >= 0.0);
    }

    // ── 14. Predict errors when not fitted ──────────────────────────────────

    #[test]
    fn test_batch_predict_not_fitted() {
        let mgr = BatchSemanticClusterManager::new(default_config(2, 2));
        let result = mgr.predict(&[1.0, 2.0]);
        assert!(result.is_err());
    }

    // ── 15. get_cluster returns correct data ────────────────────────────────

    #[test]
    fn test_batch_get_cluster() {
        let data = well_separated_2d(10);
        let mut mgr = BatchSemanticClusterManager::new(default_config(2, 2));
        mgr.fit(&data).expect("fit");

        let c = mgr.get_cluster(0).expect("cluster 0 exists");
        assert_eq!(c.id, 0);
        assert!(!c.centroid.is_empty());
        assert!(mgr.get_cluster(999).is_none());
    }

    // ── 16. Silhouette score in valid range ─────────────────────────────────

    #[test]
    fn test_batch_silhouette_score_range() {
        let data = well_separated_2d(20);
        let mut mgr = BatchSemanticClusterManager::new(default_config(2, 2));
        mgr.fit(&data).expect("fit");

        let sil = mgr.silhouette_score_approx(&data).expect("silhouette");
        assert!(
            (-1.0..=1.0).contains(&sil),
            "silhouette {} out of [-1,1]",
            sil
        );
    }

    // ── 17. Silhouette score high for well-separated clusters ───────────────

    #[test]
    fn test_batch_silhouette_score_high_for_separated() {
        let data = well_separated_2d(20);
        let mut mgr = BatchSemanticClusterManager::new(default_config(2, 2));
        mgr.fit(&data).expect("fit");

        let sil = mgr.silhouette_score_approx(&data).expect("silhouette");
        assert!(
            sil > 0.5,
            "expected high silhouette for separated clusters, got {}",
            sil
        );
    }

    // ── 18. Silhouette errors when not fitted ───────────────────────────────

    #[test]
    fn test_batch_silhouette_not_fitted() {
        let mgr = BatchSemanticClusterManager::new(default_config(2, 2));
        let result = mgr.silhouette_score_approx(&[]);
        assert!(result.is_err());
    }

    // ── 19. Cluster members returns empty for unused cluster ────────────────

    #[test]
    fn test_batch_cluster_members_empty_cluster() {
        // With well-separated data, one of the clusters beyond 2 will be empty.
        let data = well_separated_2d(10);
        let mut mgr = BatchSemanticClusterManager::new(default_config(2, 2));
        mgr.fit(&data).expect("fit");

        let members = mgr.cluster_members(999);
        assert!(members.is_empty());
    }

    // ── 20. Euclidean distance helper ───────────────────────────────────────

    #[test]
    fn test_euclidean_distance_f64() {
        let d = euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_distance_same_point() {
        let d = euclidean_distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        assert!(d.abs() < 1e-15);
    }

    #[test]
    fn test_euclidean_distance_mismatch() {
        let d = euclidean_distance(&[1.0], &[1.0, 2.0]);
        assert_eq!(d, 0.0);
    }

    // ── 23. vec_mean helper ─────────────────────────────────────────────────

    #[test]
    fn test_vec_mean_basic() {
        let v1 = vec![2.0, 4.0];
        let v2 = vec![4.0, 8.0];
        let mean = vec_mean(&[v1.as_slice(), v2.as_slice()]);
        assert!((mean[0] - 3.0).abs() < 1e-10);
        assert!((mean[1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec_mean_empty() {
        let mean = vec_mean(&[]);
        assert!(mean.is_empty());
    }

    // ── 25. Dimension mismatch in fit ───────────────────────────────────────

    #[test]
    fn test_batch_fit_dimension_mismatch() {
        let data = vec![
            ("a".to_string(), vec![1.0, 2.0]),
            ("b".to_string(), vec![3.0, 4.0, 5.0]),
        ];
        let mut mgr = BatchSemanticClusterManager::new(default_config(2, 2));
        let result = mgr.fit(&data);
        assert!(result.is_err());
    }

    // ── 26. Zero-dim config detected ────────────────────────────────────────

    #[test]
    fn test_batch_fit_zero_dim_vectors() {
        let data: Vec<(String, Vec<f64>)> =
            vec![("a".to_string(), vec![]), ("b".to_string(), vec![])];
        let mut mgr = BatchSemanticClusterManager::new(default_config(2, 0));
        let result = mgr.fit(&data);
        assert!(result.is_err());
    }

    // ── 27. Three clusters ──────────────────────────────────────────────────

    #[test]
    fn test_batch_three_clusters() {
        let mut data: Vec<(String, Vec<f64>)> = Vec::new();
        for i in 0..10 {
            data.push((format!("g0_{}", i), vec![0.0 + i as f64 * 0.001, 0.0]));
        }
        for i in 0..10 {
            data.push((format!("g1_{}", i), vec![100.0 + i as f64 * 0.001, 0.0]));
        }
        for i in 0..10 {
            data.push((format!("g2_{}", i), vec![0.0, 100.0 + i as f64 * 0.001]));
        }

        let mut mgr = BatchSemanticClusterManager::new(default_config(3, 2));
        mgr.fit(&data).expect("fit 3 clusters");

        // All g0 members should be in same cluster, distinct from g1, g2.
        let c0 = mgr.get_assignment("g0_0").expect("g0_0");
        let c1 = mgr.get_assignment("g1_0").expect("g1_0");
        let c2 = mgr.get_assignment("g2_0").expect("g2_0");
        assert_ne!(c0, c1);
        assert_ne!(c0, c2);
        assert_ne!(c1, c2);
    }

    // ── 28. Stats after fresh construction ──────────────────────────────────

    #[test]
    fn test_batch_stats_unfitted() {
        let mgr = BatchSemanticClusterManager::new(default_config(4, 3));
        let stats = mgr.stats();
        assert_eq!(stats.num_clusters, 0);
        assert_eq!(stats.total_members, 0);
        assert_eq!(stats.iterations_run, 0);
        assert!(!stats.converged);
        assert_eq!(stats.total_inertia, 0.0);
    }

    // ── 29. Predict consistency ─────────────────────────────────────────────

    #[test]
    fn test_batch_predict_consistency() {
        let data = well_separated_2d(20);
        let mut mgr = BatchSemanticClusterManager::new(default_config(2, 2));
        mgr.fit(&data).expect("fit");

        // Predict on a training point should match its assignment.
        for (id, v) in &data {
            let predicted = mgr.predict(v).expect("predict");
            let assigned = mgr.get_assignment(id).expect("assignment");
            assert_eq!(predicted, assigned, "predict({}) != assignment({})", id, id);
        }
    }

    // ── 30. Identical embeddings fewer than k ───────────────────────────────

    #[test]
    fn test_batch_identical_embeddings_fewer_distinct_than_k() {
        // All identical → only 1 distinct, but k=2 → error.
        let data: Vec<(String, Vec<f64>)> = (0..10)
            .map(|i| (format!("d{}", i), vec![5.0, 5.0]))
            .collect();
        let mut mgr = BatchSemanticClusterManager::new(default_config(2, 2));
        let result = mgr.fit(&data);
        assert!(result.is_err());
        assert!(result.expect_err("err").contains("distinct"),);
    }
}
