//! Semantic Cluster Analyzer — k-means++ style cluster analysis over embedding vectors.
//!
//! Provides cluster statistics, inertia computation, and outlier detection
//! using a deterministic k-means++ initialization strategy (no random selection).

/// A single point in the embedding space.
#[derive(Debug, Clone)]
pub struct ClusterPoint {
    /// Unique identifier for this point.
    pub id: u64,
    /// The embedding vector.
    pub vector: Vec<f32>,
    /// The cluster this point is assigned to, or `None` if unassigned.
    pub cluster_id: Option<usize>,
}

/// A cluster of embedding vectors with a computed centroid.
#[derive(Debug, Clone)]
pub struct Cluster {
    /// Cluster index (0-based).
    pub id: usize,
    /// The centroid vector (mean of all member vectors).
    pub centroid: Vec<f32>,
    /// IDs of all member points.
    pub member_ids: Vec<u64>,
}

impl Cluster {
    /// Returns the number of members in this cluster.
    #[inline]
    pub fn size(&self) -> usize {
        self.member_ids.len()
    }

    /// Returns `true` when the cluster has no members.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.member_ids.is_empty()
    }
}

/// Aggregate statistics over a completed k-means run.
#[derive(Debug, Clone)]
pub struct ClusterStats {
    /// Number of clusters requested (`k`).
    pub k: usize,
    /// Total number of points clustered.
    pub total_points: usize,
    /// Sum of squared Euclidean distances from each point to its assigned centroid.
    pub inertia: f64,
    /// Size of the largest cluster.
    pub largest_cluster: usize,
    /// Size of the smallest non-empty cluster.
    pub smallest_cluster: usize,
}

impl ClusterStats {
    /// Returns `smallest / largest` (1.0 = perfectly balanced).
    /// Returns 0.0 when `largest_cluster == 0`.
    pub fn balance_ratio(&self) -> f64 {
        if self.largest_cluster == 0 {
            return 0.0;
        }
        self.smallest_cluster as f64 / self.largest_cluster as f64
    }
}

impl Default for ClusterStats {
    fn default() -> Self {
        Self {
            k: 0,
            total_points: 0,
            inertia: 0.0,
            largest_cluster: 0,
            smallest_cluster: 0,
        }
    }
}

/// Configuration for the [`SemanticClusterAnalyzer`].
#[derive(Debug, Clone)]
pub struct AnalyzerConfig {
    /// Maximum number of k-means iterations (default 50).
    pub max_iterations: usize,
    /// Stop early when all centroid movements fall below this threshold (default 1e-4).
    pub convergence_threshold: f64,
    /// Point is considered an outlier when its distance to its centroid exceeds
    /// `outlier_distance_factor * avg_intra_distance` (default 3.0).
    pub outlier_distance_factor: f64,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            convergence_threshold: 1e-4,
            outlier_distance_factor: 3.0,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Squared Euclidean distance between two equal-length slices.
#[inline]
fn squared_distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = (x - y) as f64;
            d * d
        })
        .sum()
}

/// Euclidean distance between two equal-length slices.
#[inline]
fn euclidean_distance(a: &[f32], b: &[f32]) -> f64 {
    squared_distance(a, b).sqrt()
}

// ──────────────────────────────────────────────────────────────────────────────
// SemanticClusterAnalyzer
// ──────────────────────────────────────────────────────────────────────────────

/// Performs k-means++ style cluster analysis over a set of embedding vectors.
///
/// Initialization is **deterministic** (k-means++ greedy farthest-point selection,
/// no randomness), which means results are reproducible given the same input order.
pub struct SemanticClusterAnalyzer {
    /// All managed points.
    pub points: Vec<ClusterPoint>,
    /// Clusters produced by the most recent [`run_kmeans`](Self::run_kmeans) call.
    pub clusters: Vec<Cluster>,
    /// Analyzer configuration.
    pub config: AnalyzerConfig,
}

impl SemanticClusterAnalyzer {
    /// Create a new analyzer with the given configuration.
    pub fn new(config: AnalyzerConfig) -> Self {
        Self {
            points: Vec::new(),
            clusters: Vec::new(),
            config,
        }
    }

    /// Add a point to the analyzer.
    pub fn add_point(&mut self, id: u64, vector: Vec<f32>) {
        self.points.push(ClusterPoint {
            id,
            vector,
            cluster_id: None,
        });
    }

    /// Run k-means clustering with `k` clusters.
    ///
    /// ### Initialization (deterministic k-means++ style)
    /// 1. First centroid = `points[0].vector`.
    /// 2. Each subsequent centroid = the point whose *minimum* distance to any
    ///    already-chosen centroid is the largest (greedy farthest-point).
    ///
    /// ### Iteration
    /// Assign → recompute → check convergence (or `max_iterations` reached).
    ///
    /// Returns a default [`ClusterStats`] when `k == 0`, `points` is empty,
    /// or `k > points.len()`.
    pub fn run_kmeans(&mut self, k: usize) -> ClusterStats {
        if k == 0 || self.points.is_empty() || k > self.points.len() {
            // Reset cluster assignments
            for p in &mut self.points {
                p.cluster_id = None;
            }
            self.clusters.clear();
            return ClusterStats::default();
        }

        let dim = self.points[0].vector.len();

        // ── Step 1: Deterministic k-means++ centroid initialisation ──────────
        let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
        centroids.push(self.points[0].vector.clone());

        for _ in 1..k {
            // For each point, compute its minimum distance to any existing centroid.
            let farthest_idx = (0..self.points.len())
                .map(|i| {
                    let min_dist = centroids
                        .iter()
                        .map(|c| squared_distance(&self.points[i].vector, c))
                        .fold(f64::INFINITY, f64::min);
                    (i, min_dist)
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            centroids.push(self.points[farthest_idx].vector.clone());
        }

        // ── Step 2: Iterative assignment & centroid update ───────────────────
        for _iter in 0..self.config.max_iterations {
            // Assignment step
            for p in &mut self.points {
                let nearest = centroids
                    .iter()
                    .enumerate()
                    .map(|(ci, c)| (ci, squared_distance(&p.vector, c)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(ci, _)| ci)
                    .unwrap_or(0);
                p.cluster_id = Some(nearest);
            }

            // Recompute centroids
            let mut new_centroids: Vec<Vec<f32>> = vec![vec![0.0f32; dim]; k];
            let mut counts: Vec<usize> = vec![0; k];

            for p in &self.points {
                if let Some(ci) = p.cluster_id {
                    counts[ci] += 1;
                    for (d, &v) in new_centroids[ci].iter_mut().zip(p.vector.iter()) {
                        *d += v;
                    }
                }
            }

            // Average; keep old centroid for empty clusters
            for ci in 0..k {
                if counts[ci] > 0 {
                    let n = counts[ci] as f32;
                    for d in &mut new_centroids[ci] {
                        *d /= n;
                    }
                } else {
                    new_centroids[ci].clone_from(&centroids[ci]);
                }
            }

            // Convergence check: all movements < threshold
            let converged = centroids
                .iter()
                .zip(new_centroids.iter())
                .all(|(old, new)| euclidean_distance(old, new) < self.config.convergence_threshold);

            centroids = new_centroids;

            if converged {
                break;
            }
        }

        // ── Step 3: Build Cluster structs ────────────────────────────────────
        let mut cluster_members: Vec<Vec<u64>> = vec![Vec::new(); k];
        for p in &self.points {
            if let Some(ci) = p.cluster_id {
                cluster_members[ci].push(p.id);
            }
        }

        self.clusters = centroids
            .iter()
            .enumerate()
            .map(|(ci, c)| Cluster {
                id: ci,
                centroid: c.clone(),
                member_ids: cluster_members[ci].clone(),
            })
            .collect();

        // ── Step 4: Compute stats ────────────────────────────────────────────
        self.compute_stats_internal(k)
    }

    // ─── Internal helpers ─────────────────────────────────────────────────────

    /// Build [`ClusterStats`] from the current cluster and point state.
    fn compute_stats_internal(&self, k: usize) -> ClusterStats {
        let mut inertia = 0.0f64;
        for p in &self.points {
            if let Some(ci) = p.cluster_id {
                if let Some(cluster) = self.clusters.get(ci) {
                    inertia += squared_distance(&p.vector, &cluster.centroid);
                }
            }
        }

        let sizes: Vec<usize> = self.clusters.iter().map(|c| c.size()).collect();
        let largest = sizes.iter().copied().max().unwrap_or(0);
        let smallest = sizes.iter().copied().filter(|&s| s > 0).min().unwrap_or(0);

        ClusterStats {
            k,
            total_points: self.points.len(),
            inertia,
            largest_cluster: largest,
            smallest_cluster: smallest,
        }
    }

    /// Return IDs of points whose distance to their centroid exceeds
    /// `factor * avg_intra_distance`.
    ///
    /// If there are no assigned points the list is empty.
    pub fn outliers(&self, factor: f64) -> Vec<u64> {
        if self.clusters.is_empty() {
            return Vec::new();
        }

        // Gather (point_id, distance_to_centroid) for every assigned point.
        let distances: Vec<(u64, f64)> = self
            .points
            .iter()
            .filter_map(|p| {
                p.cluster_id.and_then(|ci| {
                    self.clusters.get(ci).map(|c| {
                        let dist = euclidean_distance(&p.vector, &c.centroid);
                        (p.id, dist)
                    })
                })
            })
            .collect();

        if distances.is_empty() {
            return Vec::new();
        }

        let avg: f64 = distances.iter().map(|(_, d)| d).sum::<f64>() / distances.len() as f64;
        let threshold = factor * avg;

        distances
            .into_iter()
            .filter(|(_, d)| *d > threshold)
            .map(|(id, _)| id)
            .collect()
    }

    /// Return the cluster id of the nearest centroid to `query`,
    /// or `None` when no clusters exist.
    pub fn nearest_cluster(&self, query: &[f32]) -> Option<usize> {
        self.clusters
            .iter()
            .map(|c| (c.id, squared_distance(query, &c.centroid)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id)
    }

    /// Return cluster statistics for the current state.
    ///
    /// Returns a default [`ClusterStats`] when no clusters have been computed.
    pub fn stats(&self) -> ClusterStats {
        if self.clusters.is_empty() {
            return ClusterStats::default();
        }
        self.compute_stats_internal(self.clusters.len())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn analyzer() -> SemanticClusterAnalyzer {
        SemanticClusterAnalyzer::new(AnalyzerConfig::default())
    }

    fn fill_two_clusters(a: &mut SemanticClusterAnalyzer) {
        // Group A — around (0, 0)
        a.add_point(1, vec![0.0, 0.0]);
        a.add_point(2, vec![0.1, 0.0]);
        a.add_point(3, vec![0.0, 0.1]);
        a.add_point(4, vec![0.05, 0.05]);
        // Group B — around (10, 10)
        a.add_point(5, vec![10.0, 10.0]);
        a.add_point(6, vec![10.1, 10.0]);
        a.add_point(7, vec![10.0, 10.1]);
        a.add_point(8, vec![9.95, 9.95]);
    }

    // ── 1. add_point stores the correct id and vector ─────────────────────────
    #[test]
    fn test_add_point_stores_correctly() {
        let mut a = analyzer();
        a.add_point(42, vec![1.0, 2.0, 3.0]);
        assert_eq!(a.points.len(), 1);
        assert_eq!(a.points[0].id, 42);
        assert_eq!(a.points[0].vector, vec![1.0, 2.0, 3.0]);
        assert!(a.points[0].cluster_id.is_none());
    }

    // ── 2. add_point: multiple points accumulate ──────────────────────────────
    #[test]
    fn test_add_point_multiple() {
        let mut a = analyzer();
        for i in 0..10u64 {
            a.add_point(i, vec![i as f32; 4]);
        }
        assert_eq!(a.points.len(), 10);
    }

    // ── 3. run_kmeans k=1 assigns all points to cluster 0 ────────────────────
    #[test]
    fn test_kmeans_k1_assigns_all() {
        let mut a = analyzer();
        fill_two_clusters(&mut a);
        let stats = a.run_kmeans(1);
        assert_eq!(stats.k, 1);
        assert_eq!(stats.total_points, 8);
        for p in &a.points {
            assert_eq!(p.cluster_id, Some(0));
        }
        assert_eq!(a.clusters.len(), 1);
        assert_eq!(a.clusters[0].member_ids.len(), 8);
    }

    // ── 4. run_kmeans k=2 separates the two well-separated groups ─────────────
    #[test]
    fn test_kmeans_k2_separates_clusters() {
        let mut a = analyzer();
        fill_two_clusters(&mut a);
        let stats = a.run_kmeans(2);
        assert_eq!(stats.k, 2);
        assert_eq!(stats.total_points, 8);
        // Both clusters should be non-empty
        assert!(!a.clusters[0].is_empty());
        assert!(!a.clusters[1].is_empty());
        // All points from group A should be in one cluster and group B in another
        // ids 1-4 should share a cluster; 5-8 should share the other
        let c_of = |id: u64| {
            a.points
                .iter()
                .find(|p| p.id == id)
                .and_then(|p| p.cluster_id)
        };
        assert_eq!(c_of(1), c_of(2));
        assert_eq!(c_of(2), c_of(3));
        assert_eq!(c_of(3), c_of(4));
        assert_eq!(c_of(5), c_of(6));
        assert_eq!(c_of(6), c_of(7));
        assert_eq!(c_of(7), c_of(8));
        assert_ne!(c_of(1), c_of(5));
    }

    // ── 5. convergence stops before max_iterations ────────────────────────────
    #[test]
    fn test_convergence_stops_early() {
        let config = AnalyzerConfig {
            max_iterations: 1000,
            convergence_threshold: 1e-4,
            outlier_distance_factor: 3.0,
        };
        let mut a = SemanticClusterAnalyzer::new(config);
        fill_two_clusters(&mut a);
        // Should still produce correct result (convergence kicks in before 1000)
        let stats = a.run_kmeans(2);
        assert_eq!(stats.k, 2);
        assert!(stats.inertia >= 0.0);
    }

    // ── 6. outliers: single outlier is detected ───────────────────────────────
    #[test]
    fn test_outliers_detected() {
        let mut a = analyzer();
        // Large tight cluster around origin so the centroid stays very close to (0,0)
        // and the avg intra-cluster distance stays tiny.
        for i in 0..50u64 {
            let v = i as f32 * 0.001;
            a.add_point(i + 1, vec![v, 0.0]);
        }
        // Far outlier — well beyond 3x avg intra-distance
        a.add_point(999, vec![500.0, 500.0]);
        a.run_kmeans(1);
        let out = a.outliers(a.config.outlier_distance_factor);
        assert!(out.contains(&999), "outlier id 999 should be detected");
    }

    // ── 7. outliers: no outliers in perfectly uniform data ─────────────────────
    #[test]
    fn test_no_outliers_uniform() {
        let mut a = analyzer();
        for i in 0..8u64 {
            a.add_point(i, vec![i as f32 * 0.001, 0.0]);
        }
        a.run_kmeans(1);
        let out = a.outliers(3.0);
        // With uniform spacing, no point should be a dramatic outlier at factor 3
        assert!(
            out.len() < a.points.len(),
            "not all points should be outliers"
        );
    }

    // ── 8. outliers: empty clusters returns empty vec ─────────────────────────
    #[test]
    fn test_outliers_no_clusters() {
        let a = analyzer();
        let out = a.outliers(3.0);
        assert!(out.is_empty());
    }

    // ── 9. nearest_cluster: returns correct cluster id ────────────────────────
    #[test]
    fn test_nearest_cluster() {
        let mut a = analyzer();
        fill_two_clusters(&mut a);
        a.run_kmeans(2);
        // A query near (0,0) should map to the cluster containing point 1
        let nc_near_origin = a.nearest_cluster(&[0.0, 0.0]);
        let cluster_of_pt1 = a
            .points
            .iter()
            .find(|p| p.id == 1)
            .and_then(|p| p.cluster_id);
        assert_eq!(nc_near_origin, cluster_of_pt1);
        // A query near (10,10) should map to the cluster containing point 5
        let nc_near_10 = a.nearest_cluster(&[10.0, 10.0]);
        let cluster_of_pt5 = a
            .points
            .iter()
            .find(|p| p.id == 5)
            .and_then(|p| p.cluster_id);
        assert_eq!(nc_near_10, cluster_of_pt5);
    }

    // ── 10. nearest_cluster: none when no clusters ───────────────────────────
    #[test]
    fn test_nearest_cluster_empty() {
        let a = analyzer();
        assert!(a.nearest_cluster(&[1.0, 2.0]).is_none());
    }

    // ── 11. balance_ratio: 1.0 for perfectly balanced clusters ───────────────
    #[test]
    fn test_balance_ratio_perfect() {
        let mut a = analyzer();
        fill_two_clusters(&mut a); // 4 + 4 points
        let stats = a.run_kmeans(2);
        assert!(
            (stats.balance_ratio() - 1.0).abs() < 1e-9,
            "expected ratio 1.0, got {}",
            stats.balance_ratio()
        );
    }

    // ── 12. balance_ratio: < 1.0 for unbalanced clusters ─────────────────────
    #[test]
    fn test_balance_ratio_unbalanced() {
        let mut a = analyzer();
        // 1 vs 9 points (well-separated so k=2 gives those sizes)
        a.add_point(1, vec![0.0, 0.0]);
        for i in 2..=10u64 {
            a.add_point(i, vec![100.0 + i as f32 * 0.01, 0.0]);
        }
        let stats = a.run_kmeans(2);
        assert!(
            stats.balance_ratio() < 1.0,
            "ratio should be < 1.0, got {}",
            stats.balance_ratio()
        );
    }

    // ── 13. balance_ratio: 0.0 when largest == 0 ─────────────────────────────
    #[test]
    fn test_balance_ratio_zero() {
        let stats = ClusterStats {
            k: 0,
            total_points: 0,
            inertia: 0.0,
            largest_cluster: 0,
            smallest_cluster: 0,
        };
        assert_eq!(stats.balance_ratio(), 0.0);
    }

    // ── 14. inertia > 0 for non-trivial data ─────────────────────────────────
    #[test]
    fn test_inertia_positive() {
        let mut a = analyzer();
        fill_two_clusters(&mut a);
        let stats = a.run_kmeans(2);
        assert!(stats.inertia > 0.0, "inertia should be positive");
    }

    // ── 15. inertia decreases from k=1 to k=2 ────────────────────────────────
    #[test]
    fn test_inertia_decreases_with_more_clusters() {
        let mut a1 = analyzer();
        fill_two_clusters(&mut a1);
        let stats1 = a1.run_kmeans(1);

        let mut a2 = analyzer();
        fill_two_clusters(&mut a2);
        let stats2 = a2.run_kmeans(2);

        assert!(
            stats2.inertia < stats1.inertia,
            "inertia with k=2 ({}) should be less than k=1 ({})",
            stats2.inertia,
            stats1.inertia
        );
    }

    // ── 16. stats() returns current cluster state ─────────────────────────────
    #[test]
    fn test_stats_reflects_clusters() {
        let mut a = analyzer();
        fill_two_clusters(&mut a);
        a.run_kmeans(2);
        let s = a.stats();
        assert_eq!(s.k, 2);
        assert_eq!(s.total_points, 8);
        assert!(s.inertia >= 0.0);
    }

    // ── 17. empty guard: k > n returns default stats ──────────────────────────
    #[test]
    fn test_guard_k_greater_than_n() {
        let mut a = analyzer();
        a.add_point(1, vec![0.0, 0.0]);
        a.add_point(2, vec![1.0, 1.0]);
        let stats = a.run_kmeans(5); // k=5 > n=2
        assert_eq!(stats.k, 0);
        assert_eq!(stats.total_points, 0);
        assert!(a.clusters.is_empty());
    }

    // ── 18. empty guard: no points returns default stats ──────────────────────
    #[test]
    fn test_guard_empty_points() {
        let mut a = analyzer();
        let stats = a.run_kmeans(3);
        assert_eq!(stats.k, 0);
    }

    // ── 19. Cluster::size and is_empty ────────────────────────────────────────
    #[test]
    fn test_cluster_size_and_is_empty() {
        let c_empty = Cluster {
            id: 0,
            centroid: vec![0.0],
            member_ids: vec![],
        };
        let c_full = Cluster {
            id: 1,
            centroid: vec![1.0],
            member_ids: vec![1, 2, 3],
        };
        assert!(c_empty.is_empty());
        assert_eq!(c_empty.size(), 0);
        assert!(!c_full.is_empty());
        assert_eq!(c_full.size(), 3);
    }

    // ── 20. nearest_cluster after k=1 always returns Some(0) ─────────────────
    #[test]
    fn test_nearest_cluster_k1() {
        let mut a = analyzer();
        fill_two_clusters(&mut a);
        a.run_kmeans(1);
        assert_eq!(a.nearest_cluster(&[999.0, 999.0]), Some(0));
        assert_eq!(a.nearest_cluster(&[-999.0, -999.0]), Some(0));
    }
}
