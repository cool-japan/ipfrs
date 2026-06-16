//! Embedding Cluster Analyzer — comprehensive cluster analysis for embedding spaces.
//!
//! Provides cluster quality metrics (silhouette, Davies-Bouldin, Calinski-Harabász),
//! outlier detection with multiple strategies, local density estimation, and
//! cluster evolution tracking between analysis snapshots.

// ─── Types ────────────────────────────────────────────────────────────────────

/// Newtype wrapper around a cluster index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClusterId(pub usize);

/// A point in the embedding space with optional cluster assignment.
#[derive(Debug, Clone)]
pub struct EcaClusterPoint {
    /// Unique string identifier for this point.
    pub id: String,
    /// The embedding vector.
    pub embedding: Vec<f64>,
    /// Which cluster this point belongs to, if known.
    pub cluster: Option<ClusterId>,
    /// Euclidean distance from this point to its assigned cluster centroid.
    pub distance_to_centroid: f64,
}

/// Description of a single cluster including geometric properties.
#[derive(Debug, Clone)]
pub struct ClusterDescriptor {
    /// Cluster index.
    pub id: ClusterId,
    /// The centroid of this cluster.
    pub centroid: Vec<f64>,
    /// Radius: maximum distance from centroid to any member point.
    pub radius: f64,
    /// Density: number of members per unit volume (simplified to member count).
    pub density: f64,
    /// Number of member points.
    pub point_count: usize,
    /// Optional human-readable label for this cluster.
    pub label: Option<String>,
}

/// Score indicating how much of an outlier a point is.
#[derive(Debug, Clone)]
pub struct OutlierScore {
    /// ID of the outlying point.
    pub point_id: String,
    /// Outlier score (higher = more anomalous).
    pub score: f64,
    /// The reason this point was flagged.
    pub reason: OutlierReason,
}

/// Reason a point was identified as an outlier.
#[derive(Debug, Clone)]
pub enum OutlierReason {
    /// Point is too far from its cluster centroid (z-score exceeded threshold).
    FarFromCentroid {
        /// Actual distance to centroid.
        distance: f64,
        /// The threshold distance that was exceeded.
        threshold: f64,
    },
    /// Point resides in a low-density region.
    LowDensityRegion {
        /// Estimated local density at this point.
        local_density: f64,
    },
    /// Point's cluster has too few members to be reliable.
    IsolatedPoint,
}

/// Configuration for the `EmbeddingClusterAnalyzer`.
#[derive(Debug, Clone)]
pub struct EcaAnalyzerConfig {
    /// Number of standard deviations beyond the mean for a point to be an outlier.
    pub outlier_threshold_sigma: f64,
    /// Minimum number of points in a cluster; smaller clusters produce `IsolatedPoint` outliers.
    pub min_cluster_size: usize,
    /// Radius used for local density estimation.
    pub density_radius: f64,
    /// Maximum fraction of total points that may be reported as outliers (0..1).
    pub max_outlier_fraction: f64,
}

impl Default for EcaAnalyzerConfig {
    fn default() -> Self {
        Self {
            outlier_threshold_sigma: 2.5,
            min_cluster_size: 3,
            density_radius: 0.1,
            max_outlier_fraction: 0.1,
        }
    }
}

/// Cluster quality metrics computed over all points and clusters.
#[derive(Debug, Clone)]
pub struct ClusterQuality {
    /// Mean silhouette coefficient over all points (range −1 to 1; higher is better).
    pub silhouette_score: f64,
    /// Davies-Bouldin index (lower is better; 0 if only one cluster).
    pub davies_bouldin_index: f64,
    /// Calinski-Harabász score (higher is better; 0 if degenerate).
    pub calinski_harabasz_score: f64,
    /// Mean squared distance from each point to its centroid.
    pub intra_cluster_variance: f64,
}

/// Summary statistics for the analyzer.
#[derive(Debug, Clone)]
pub struct EcaAnalyzerStats {
    /// Total number of points held by the analyzer.
    pub point_count: usize,
    /// Number of clusters currently registered.
    pub cluster_count: usize,
    /// Cumulative number of quality analyses performed.
    pub total_analyses: u64,
    /// Average number of points per cluster (0.0 if no clusters).
    pub avg_cluster_size: f64,
    /// Number of outliers detected in the last `detect_outliers` call.
    pub outlier_count: usize,
}

// ─── Analyzer ─────────────────────────────────────────────────────────────────

/// Comprehensive cluster analysis system for embedding spaces.
///
/// Tracks points and their cluster assignments, computes quality metrics,
/// detects outliers, estimates local density, and identifies cluster drift
/// relative to a previous snapshot.
pub struct EmbeddingClusterAnalyzer {
    /// Analyzer configuration.
    pub config: EcaAnalyzerConfig,
    /// All points currently held by the analyzer.
    pub points: Vec<EcaClusterPoint>,
    /// Registered cluster descriptors.
    pub clusters: Vec<ClusterDescriptor>,
    /// Cumulative number of `compute_cluster_quality` calls.
    pub total_analyses: u64,
    /// Cached outlier count from the last `detect_outliers` call.
    last_outlier_count: usize,
}

impl EmbeddingClusterAnalyzer {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Create a new analyzer with the given configuration.
    pub fn new(config: EcaAnalyzerConfig) -> Self {
        Self {
            config,
            points: Vec::new(),
            clusters: Vec::new(),
            total_analyses: 0,
            last_outlier_count: 0,
        }
    }

    // ── Point management ──────────────────────────────────────────────────────

    /// Add an embedding point to the analyzer.
    ///
    /// `distance_to_centroid` starts at `0.0`; call `set_clusters` or
    /// `recompute_distances` to update it.
    pub fn add_point(&mut self, id: String, embedding: Vec<f64>, cluster: Option<ClusterId>) {
        self.points.push(EcaClusterPoint {
            id,
            embedding,
            cluster,
            distance_to_centroid: 0.0,
        });
    }

    /// Replace the cluster descriptors and re-assign / recompute distances.
    ///
    /// For each point whose `cluster` field is `None`, the nearest cluster
    /// (by cosine distance to centroid) is assigned. Then `distance_to_centroid`
    /// is recomputed for every point using L2 distance to its centroid.
    pub fn set_clusters(&mut self, descriptors: Vec<ClusterDescriptor>) {
        self.clusters = descriptors;
        self.assign_unassigned_points();
        self.recompute_distances();
    }

    /// Assign points with `cluster == None` to the nearest cluster centroid
    /// (cosine distance).
    fn assign_unassigned_points(&mut self) {
        if self.clusters.is_empty() {
            return;
        }
        for point in &mut self.points {
            if point.cluster.is_some() {
                continue;
            }
            let best = self
                .clusters
                .iter()
                .enumerate()
                .min_by(|(_, ca), (_, cb)| {
                    let da = Self::cosine_distance_static(&point.embedding, &ca.centroid);
                    let db = Self::cosine_distance_static(&point.embedding, &cb.centroid);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i);
            if let Some(idx) = best {
                point.cluster = Some(ClusterId(idx));
            }
        }
    }

    /// Recompute `distance_to_centroid` for every assigned point.
    fn recompute_distances(&mut self) {
        for point in &mut self.points {
            let dist = match point.cluster {
                None => 0.0,
                Some(cid) => self
                    .clusters
                    .get(cid.0)
                    .map(|c| Self::l2_distance_static(&point.embedding, &c.centroid))
                    .unwrap_or(0.0),
            };
            point.distance_to_centroid = dist;
        }
    }

    // ── Distance metrics ──────────────────────────────────────────────────────

    /// Euclidean (L2) distance between two vectors.
    ///
    /// Returns `0.0` if either slice is empty.
    pub fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
        Self::l2_distance_static(a, b)
    }

    fn l2_distance_static(a: &[f64], b: &[f64]) -> f64 {
        let len = a.len().min(b.len());
        a[..len]
            .iter()
            .zip(b[..len].iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    /// Cosine distance (1 − cosine similarity) between two vectors.
    ///
    /// Returns `1.0` if either vector has zero norm.
    pub fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
        Self::cosine_distance_static(a, b)
    }

    fn cosine_distance_static(a: &[f64], b: &[f64]) -> f64 {
        let len = a.len().min(b.len());
        let dot: f64 = a[..len]
            .iter()
            .zip(b[..len].iter())
            .map(|(x, y)| x * y)
            .sum();
        let norm_a: f64 = a[..len].iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b[..len].iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 1.0;
        }
        let similarity = dot / (norm_a * norm_b);
        // Clamp to [-1, 1] to guard against floating-point drift.
        1.0 - similarity.clamp(-1.0, 1.0)
    }

    // ── Cluster quality ───────────────────────────────────────────────────────

    /// Compute comprehensive cluster quality metrics.
    ///
    /// Increments `total_analyses` on each call.
    pub fn compute_cluster_quality(&mut self) -> ClusterQuality {
        self.total_analyses += 1;

        let n = self.points.len();
        let k = self.clusters.len();

        // ── Intra-cluster variance ─────────────────────────────────────────
        let intra_cluster_variance = if n == 0 {
            0.0
        } else {
            self.points
                .iter()
                .map(|p| p.distance_to_centroid * p.distance_to_centroid)
                .sum::<f64>()
                / n as f64
        };

        // ── Silhouette score ───────────────────────────────────────────────
        let silhouette_score = self.compute_silhouette();

        // ── Davies-Bouldin index ───────────────────────────────────────────
        let davies_bouldin_index = self.compute_davies_bouldin();

        // ── Calinski-Harabász score ────────────────────────────────────────
        let calinski_harabasz_score = self.compute_calinski_harabasz(n, k);

        ClusterQuality {
            silhouette_score,
            davies_bouldin_index,
            calinski_harabasz_score,
            intra_cluster_variance,
        }
    }

    /// Mean silhouette coefficient (simplified centroid-based variant).
    ///
    /// For each point:
    ///   - `a` = distance to its own cluster centroid
    ///   - `b` = minimum average distance to any other cluster centroid
    ///   - silhouette = (b − a) / max(a, b)
    fn compute_silhouette(&self) -> f64 {
        let n = self.points.len();
        let k = self.clusters.len();
        if n == 0 || k < 2 {
            return 0.0;
        }

        let scores: Vec<f64> = self
            .points
            .iter()
            .map(|point| {
                let own_cluster_idx = match point.cluster {
                    Some(cid) => cid.0,
                    None => return 0.0,
                };

                let a = point.distance_to_centroid;

                // b = minimum L2 distance to any other cluster centroid
                let b = self
                    .clusters
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != own_cluster_idx)
                    .map(|(_, c)| Self::l2_distance_static(&point.embedding, &c.centroid))
                    .fold(f64::MAX, f64::min);

                if b == f64::MAX {
                    return 0.0;
                }

                let denom = a.max(b);
                if denom == 0.0 {
                    0.0
                } else {
                    (b - a) / denom
                }
            })
            .collect();

        if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        }
    }

    /// Davies-Bouldin index.
    ///
    /// Returns `0.0` when there is only one cluster or no clusters.
    fn compute_davies_bouldin(&self) -> f64 {
        let k = self.clusters.len();
        if k < 2 {
            return 0.0;
        }

        // σ_i = mean distance to centroid for cluster i
        let sigma: Vec<f64> = self
            .clusters
            .iter()
            .map(|c| {
                let members: Vec<f64> = self
                    .points
                    .iter()
                    .filter(|p| p.cluster == Some(c.id))
                    .map(|p| p.distance_to_centroid)
                    .collect();
                if members.is_empty() {
                    0.0
                } else {
                    members.iter().sum::<f64>() / members.len() as f64
                }
            })
            .collect();

        let db: f64 = self
            .clusters
            .iter()
            .enumerate()
            .map(|(i, ci)| {
                let max_ratio = self
                    .clusters
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(j, cj)| {
                        let dist = Self::l2_distance_static(&ci.centroid, &cj.centroid);
                        if dist == 0.0 {
                            0.0
                        } else {
                            (sigma[i] + sigma[j]) / dist
                        }
                    })
                    .fold(f64::NEG_INFINITY, f64::max);
                if max_ratio == f64::NEG_INFINITY {
                    0.0
                } else {
                    max_ratio
                }
            })
            .sum::<f64>();

        db / k as f64
    }

    /// Calinski-Harabász score.
    ///
    /// Returns `0.0` for degenerate cases (< 2 clusters, < 2 points, etc.).
    fn compute_calinski_harabasz(&self, n: usize, k: usize) -> f64 {
        if n < 2 || k < 2 || n <= k {
            return 0.0;
        }

        // Global centroid
        let dim = self.points.first().map(|p| p.embedding.len()).unwrap_or(0);
        if dim == 0 {
            return 0.0;
        }

        let mut global_centroid = vec![0.0_f64; dim];
        for point in &self.points {
            for (g, v) in global_centroid.iter_mut().zip(point.embedding.iter()) {
                *g += v;
            }
        }
        let n_f = n as f64;
        for g in &mut global_centroid {
            *g /= n_f;
        }

        // BGSS = sum over clusters of n_k * ||centroid_k - global_centroid||^2
        let bgss: f64 = self
            .clusters
            .iter()
            .map(|c| {
                let n_k = self
                    .points
                    .iter()
                    .filter(|p| p.cluster == Some(c.id))
                    .count() as f64;
                let dist_sq = Self::l2_distance_static(&c.centroid, &global_centroid).powi(2);
                n_k * dist_sq
            })
            .sum();

        // WGSS = sum of all distance_to_centroid^2
        let wgss: f64 = self
            .points
            .iter()
            .map(|p| p.distance_to_centroid * p.distance_to_centroid)
            .sum();

        if wgss == 0.0 {
            return 0.0;
        }

        let numerator = bgss / (k as f64 - 1.0);
        let denominator = wgss / (n as f64 - k as f64);
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    // ── Outlier detection ─────────────────────────────────────────────────────

    /// Detect outlier points using three strategies:
    ///
    /// 1. **FarFromCentroid** — per-cluster z-score of `distance_to_centroid` > `threshold_sigma`
    /// 2. **IsolatedPoint** — member of a cluster with fewer than `min_cluster_size` points
    ///
    /// Results are capped at `max_outlier_fraction × total_points`, ordered by
    /// descending outlier score.
    pub fn detect_outliers(&mut self) -> Vec<OutlierScore> {
        let total = self.points.len();
        if total == 0 {
            self.last_outlier_count = 0;
            return Vec::new();
        }

        let mut scores: Vec<OutlierScore> = Vec::new();

        for cluster in &self.clusters {
            let members: Vec<(usize, f64)> = self
                .points
                .iter()
                .enumerate()
                .filter(|(_, p)| p.cluster == Some(cluster.id))
                .map(|(i, p)| (i, p.distance_to_centroid))
                .collect();

            let count = members.len();

            // IsolatedPoint check
            if count < self.config.min_cluster_size {
                for (idx, dist) in &members {
                    scores.push(OutlierScore {
                        point_id: self.points[*idx].id.clone(),
                        score: 1.0 + dist,
                        reason: OutlierReason::IsolatedPoint,
                    });
                }
                continue;
            }

            // FarFromCentroid check
            let mean = members.iter().map(|(_, d)| *d).sum::<f64>() / count as f64;
            let variance = members
                .iter()
                .map(|(_, d)| (d - mean) * (d - mean))
                .sum::<f64>()
                / count as f64;
            let std_dev = variance.sqrt();

            let threshold = mean + self.config.outlier_threshold_sigma * std_dev;

            for (idx, dist) in &members {
                if *dist > threshold {
                    let score = if std_dev > 0.0 {
                        (dist - mean) / std_dev
                    } else {
                        0.0
                    };
                    scores.push(OutlierScore {
                        point_id: self.points[*idx].id.clone(),
                        score,
                        reason: OutlierReason::FarFromCentroid {
                            distance: *dist,
                            threshold,
                        },
                    });
                }
            }
        }

        // Sort descending by score
        scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Cap at max_outlier_fraction
        let max_count = ((total as f64) * self.config.max_outlier_fraction).ceil() as usize;
        scores.truncate(max_count);

        self.last_outlier_count = scores.len();
        scores
    }

    // ── Local density ─────────────────────────────────────────────────────────

    /// Estimate the local density around a point (by index).
    ///
    /// Counts how many other points lie within `density_radius` (L2).
    /// Returns `0.0` for invalid indices.
    pub fn local_density(&self, point_idx: usize) -> f64 {
        let Some(target) = self.points.get(point_idx) else {
            return 0.0;
        };
        let radius = self.config.density_radius;
        let count = self
            .points
            .iter()
            .enumerate()
            .filter(|(i, other)| {
                *i != point_idx
                    && Self::l2_distance_static(&target.embedding, &other.embedding) <= radius
            })
            .count();
        count as f64
    }

    // ── Cluster evolution ─────────────────────────────────────────────────────

    /// Compare cluster centroids between `self` (current) and `prev` (snapshot).
    ///
    /// For each cluster in `self`, finds the closest cluster in `prev` by L2
    /// centroid distance. If the distance exceeds `0.1`, a message is appended:
    /// `"cluster {id} shifted by {dist:.3}"`.
    pub fn cluster_evolution(&self, prev: &EmbeddingClusterAnalyzer) -> Vec<String> {
        let mut events = Vec::new();

        for curr_cluster in &self.clusters {
            // Find closest cluster in prev by centroid distance
            let closest = prev.clusters.iter().min_by(|a, b| {
                let da = Self::l2_distance_static(&curr_cluster.centroid, &a.centroid);
                let db = Self::l2_distance_static(&curr_cluster.centroid, &b.centroid);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });

            if let Some(prev_cluster) = closest {
                let dist = Self::l2_distance_static(&curr_cluster.centroid, &prev_cluster.centroid);
                if dist > 0.1 {
                    events.push(format!(
                        "cluster {} shifted by {:.3}",
                        curr_cluster.id.0, dist
                    ));
                }
            }
        }

        events
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Return the `k` points in `cluster` that are closest to the centroid.
    ///
    /// Points are ordered by ascending `distance_to_centroid`.
    /// Returns an empty `Vec` if the cluster does not exist.
    pub fn top_k_by_cluster(&self, cluster: ClusterId, k: usize) -> Vec<&EcaClusterPoint> {
        let mut members: Vec<&EcaClusterPoint> = self
            .points
            .iter()
            .filter(|p| p.cluster == Some(cluster))
            .collect();

        members.sort_by(|a, b| {
            a.distance_to_centroid
                .partial_cmp(&b.distance_to_centroid)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        members.truncate(k);
        members
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    /// Return a summary of the current analyzer state.
    pub fn analyzer_stats(&self) -> EcaAnalyzerStats {
        let point_count = self.points.len();
        let cluster_count = self.clusters.len();
        let avg_cluster_size = if cluster_count == 0 {
            0.0
        } else {
            point_count as f64 / cluster_count as f64
        };

        EcaAnalyzerStats {
            point_count,
            cluster_count,
            total_analyses: self.total_analyses,
            avg_cluster_size,
            outlier_count: self.last_outlier_count,
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::embedding_cluster_analyzer::{
        ClusterDescriptor, ClusterId, EcaAnalyzerConfig, EcaClusterPoint, EmbeddingClusterAnalyzer,
        OutlierReason,
    };

    // ── Helper constructors ──────────────────────────────────────────────────

    fn default_config() -> EcaAnalyzerConfig {
        EcaAnalyzerConfig::default()
    }

    fn make_analyzer() -> EmbeddingClusterAnalyzer {
        EmbeddingClusterAnalyzer::new(default_config())
    }

    fn make_descriptor(id: usize, centroid: Vec<f64>) -> ClusterDescriptor {
        ClusterDescriptor {
            id: ClusterId(id),
            centroid,
            radius: 1.0,
            density: 1.0,
            point_count: 0,
            label: None,
        }
    }

    // ── 1: Default config values ─────────────────────────────────────────────

    #[test]
    fn test_default_config() {
        let cfg = EcaAnalyzerConfig::default();
        assert!((cfg.outlier_threshold_sigma - 2.5).abs() < 1e-10);
        assert_eq!(cfg.min_cluster_size, 3);
        assert!((cfg.density_radius - 0.1).abs() < 1e-10);
        assert!((cfg.max_outlier_fraction - 0.1).abs() < 1e-10);
    }

    // ── 2: New analyzer is empty ─────────────────────────────────────────────

    #[test]
    fn test_new_analyzer_empty() {
        let a = make_analyzer();
        assert_eq!(a.points.len(), 0);
        assert_eq!(a.clusters.len(), 0);
        assert_eq!(a.total_analyses, 0);
    }

    // ── 3: Add point increments point count ──────────────────────────────────

    #[test]
    fn test_add_point_count() {
        let mut a = make_analyzer();
        a.add_point("p1".into(), vec![1.0, 0.0], None);
        a.add_point("p2".into(), vec![0.0, 1.0], None);
        assert_eq!(a.points.len(), 2);
    }

    // ── 4: Add point stores correct id ───────────────────────────────────────

    #[test]
    fn test_add_point_id() {
        let mut a = make_analyzer();
        a.add_point("my-point".into(), vec![1.0], None);
        assert_eq!(a.points[0].id, "my-point");
    }

    // ── 5: Add point stores correct embedding ────────────────────────────────

    #[test]
    fn test_add_point_embedding() {
        let mut a = make_analyzer();
        a.add_point("p".into(), vec![3.0, 4.0], None);
        assert_eq!(a.points[0].embedding, vec![3.0, 4.0]);
    }

    // ── 6: Add point initial distance is zero ────────────────────────────────

    #[test]
    fn test_add_point_initial_distance_zero() {
        let mut a = make_analyzer();
        a.add_point("p".into(), vec![1.0], None);
        assert_eq!(a.points[0].distance_to_centroid, 0.0);
    }

    // ── 7: L2 distance — zero vector ─────────────────────────────────────────

    #[test]
    fn test_l2_distance_zero() {
        let d = EmbeddingClusterAnalyzer::l2_distance(&[0.0, 0.0], &[0.0, 0.0]);
        assert!(d.abs() < 1e-10);
    }

    // ── 8: L2 distance — 3-4-5 triangle ─────────────────────────────────────

    #[test]
    fn test_l2_distance_345() {
        let d = EmbeddingClusterAnalyzer::l2_distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 5.0).abs() < 1e-10);
    }

    // ── 9: L2 distance — symmetric ───────────────────────────────────────────

    #[test]
    fn test_l2_distance_symmetric() {
        let a = &[1.0, 2.0, 3.0];
        let b = &[4.0, 5.0, 6.0];
        let d1 = EmbeddingClusterAnalyzer::l2_distance(a, b);
        let d2 = EmbeddingClusterAnalyzer::l2_distance(b, a);
        assert!((d1 - d2).abs() < 1e-10);
    }

    // ── 10: Cosine distance — identical vectors ───────────────────────────────

    #[test]
    fn test_cosine_distance_identical() {
        let v = &[1.0, 2.0, 3.0];
        let d = EmbeddingClusterAnalyzer::cosine_distance(v, v);
        assert!(d.abs() < 1e-10);
    }

    // ── 11: Cosine distance — orthogonal vectors ──────────────────────────────

    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = &[1.0, 0.0];
        let b = &[0.0, 1.0];
        let d = EmbeddingClusterAnalyzer::cosine_distance(a, b);
        assert!((d - 1.0).abs() < 1e-10);
    }

    // ── 12: Cosine distance — zero vector returns 1.0 ─────────────────────────

    #[test]
    fn test_cosine_distance_zero_vector() {
        let d = EmbeddingClusterAnalyzer::cosine_distance(&[0.0, 0.0], &[1.0, 0.0]);
        assert!((d - 1.0).abs() < 1e-10);
    }

    // ── 13: set_clusters registers descriptors ────────────────────────────────

    #[test]
    fn test_set_clusters_registers() {
        let mut a = make_analyzer();
        a.set_clusters(vec![make_descriptor(0, vec![1.0, 0.0])]);
        assert_eq!(a.clusters.len(), 1);
    }

    // ── 14: set_clusters assigns unassigned points ────────────────────────────

    #[test]
    fn test_set_clusters_assigns_unassigned() {
        let mut a = make_analyzer();
        a.add_point("p".into(), vec![1.0, 0.0], None);
        a.set_clusters(vec![make_descriptor(0, vec![1.0, 0.0])]);
        assert_eq!(a.points[0].cluster, Some(ClusterId(0)));
    }

    // ── 15: set_clusters recomputes distance ──────────────────────────────────

    #[test]
    fn test_set_clusters_recomputes_distance() {
        let mut a = make_analyzer();
        a.add_point("p".into(), vec![4.0, 0.0], None);
        a.set_clusters(vec![make_descriptor(0, vec![0.0, 0.0])]);
        assert!((a.points[0].distance_to_centroid - 4.0).abs() < 1e-10);
    }

    // ── 16: set_clusters preserves explicit assignment ────────────────────────

    #[test]
    fn test_set_clusters_preserves_explicit() {
        let mut a = make_analyzer();
        a.add_point("p".into(), vec![0.0, 1.0], Some(ClusterId(1)));
        a.set_clusters(vec![
            make_descriptor(0, vec![0.0, 1.0]),
            make_descriptor(1, vec![1.0, 0.0]),
        ]);
        // Explicit cluster should NOT be overwritten
        assert_eq!(a.points[0].cluster, Some(ClusterId(1)));
    }

    // ── 17: intra_cluster_variance is mean of squared distances ──────────────

    #[test]
    fn test_intra_cluster_variance() {
        let mut a = make_analyzer();
        // Two points, both at distance 3 from centroid [0,0] along x-axis
        a.add_point("p1".into(), vec![3.0, 0.0], Some(ClusterId(0)));
        a.add_point("p2".into(), vec![-3.0, 0.0], Some(ClusterId(0)));
        a.set_clusters(vec![make_descriptor(0, vec![0.0, 0.0])]);
        let q = a.compute_cluster_quality();
        // Each distance = 3, so variance = (9 + 9) / 2 = 9
        assert!((q.intra_cluster_variance - 9.0).abs() < 1e-9);
    }

    // ── 18: silhouette is 0 with single cluster ───────────────────────────────

    #[test]
    fn test_silhouette_single_cluster() {
        let mut a = make_analyzer();
        a.add_point("p1".into(), vec![1.0, 0.0], Some(ClusterId(0)));
        a.set_clusters(vec![make_descriptor(0, vec![0.0, 0.0])]);
        let q = a.compute_cluster_quality();
        assert_eq!(q.silhouette_score, 0.0);
    }

    // ── 19: silhouette positive for well-separated clusters ───────────────────

    #[test]
    fn test_silhouette_well_separated() {
        let mut a = make_analyzer();
        // Cluster 0 near origin, cluster 1 far away
        for i in 0..5_u32 {
            a.add_point(
                format!("a{i}"),
                vec![i as f64 * 0.01, 0.0],
                Some(ClusterId(0)),
            );
        }
        for i in 0..5_u32 {
            a.add_point(
                format!("b{i}"),
                vec![100.0 + i as f64 * 0.01, 0.0],
                Some(ClusterId(1)),
            );
        }
        a.set_clusters(vec![
            make_descriptor(0, vec![0.02, 0.0]),
            make_descriptor(1, vec![100.02, 0.0]),
        ]);
        let q = a.compute_cluster_quality();
        assert!(
            q.silhouette_score > 0.5,
            "Expected high silhouette, got {}",
            q.silhouette_score
        );
    }

    // ── 20: davies_bouldin 0 with single cluster ─────────────────────────────

    #[test]
    fn test_davies_bouldin_single_cluster() {
        let mut a = make_analyzer();
        a.add_point("p".into(), vec![1.0], Some(ClusterId(0)));
        a.set_clusters(vec![make_descriptor(0, vec![0.0])]);
        let q = a.compute_cluster_quality();
        assert_eq!(q.davies_bouldin_index, 0.0);
    }

    // ── 21: calinski_harabasz 0 with single cluster ───────────────────────────

    #[test]
    fn test_calinski_harabasz_single_cluster() {
        let mut a = make_analyzer();
        a.add_point("p1".into(), vec![1.0], Some(ClusterId(0)));
        a.add_point("p2".into(), vec![2.0], Some(ClusterId(0)));
        a.set_clusters(vec![make_descriptor(0, vec![1.5])]);
        let q = a.compute_cluster_quality();
        assert_eq!(q.calinski_harabasz_score, 0.0);
    }

    // ── 22: total_analyses increments ────────────────────────────────────────

    #[test]
    fn test_total_analyses_increments() {
        let mut a = make_analyzer();
        a.add_point("p".into(), vec![1.0], Some(ClusterId(0)));
        a.set_clusters(vec![make_descriptor(0, vec![0.0])]);
        a.compute_cluster_quality();
        a.compute_cluster_quality();
        a.compute_cluster_quality();
        assert_eq!(a.total_analyses, 3);
    }

    // ── 23: detect_outliers returns empty for empty analyzer ─────────────────

    #[test]
    fn test_detect_outliers_empty() {
        let mut a = make_analyzer();
        let outliers = a.detect_outliers();
        assert!(outliers.is_empty());
    }

    // ── 24: detect_outliers flags FarFromCentroid ─────────────────────────────

    #[test]
    fn test_detect_outliers_far_from_centroid() {
        let mut a = EmbeddingClusterAnalyzer::new(EcaAnalyzerConfig {
            outlier_threshold_sigma: 1.0,
            min_cluster_size: 2,
            max_outlier_fraction: 1.0,
            ..Default::default()
        });
        // 5 tight points + 1 very far point
        for i in 0..5_u32 {
            a.add_point(format!("n{i}"), vec![i as f64 * 0.01], Some(ClusterId(0)));
        }
        a.add_point("far".into(), vec![1000.0], Some(ClusterId(0)));
        a.set_clusters(vec![make_descriptor(0, vec![0.02])]);

        let outliers = a.detect_outliers();
        assert!(!outliers.is_empty(), "Expected at least one outlier");
        let far = outliers.iter().find(|o| o.point_id == "far");
        assert!(far.is_some(), "Expected 'far' to be detected as outlier");
        assert!(
            matches!(
                far.expect("test: 'far' outlier should be present in results")
                    .reason,
                OutlierReason::FarFromCentroid { .. }
            ),
            "Expected FarFromCentroid reason"
        );
    }

    // ── 25: detect_outliers flags IsolatedPoint ────────────────────────────────

    #[test]
    fn test_detect_outliers_isolated_point() {
        let mut a = EmbeddingClusterAnalyzer::new(EcaAnalyzerConfig {
            min_cluster_size: 3,
            max_outlier_fraction: 1.0,
            ..Default::default()
        });
        // Only 2 points in cluster (< min_cluster_size=3)
        a.add_point("p1".into(), vec![0.0], Some(ClusterId(0)));
        a.add_point("p2".into(), vec![0.1], Some(ClusterId(0)));
        a.set_clusters(vec![make_descriptor(0, vec![0.05])]);

        let outliers = a.detect_outliers();
        assert_eq!(outliers.len(), 2);
        assert!(outliers
            .iter()
            .all(|o| matches!(o.reason, OutlierReason::IsolatedPoint)));
    }

    // ── 26: detect_outliers caps at max_outlier_fraction ──────────────────────

    #[test]
    fn test_detect_outliers_cap() {
        let mut a = EmbeddingClusterAnalyzer::new(EcaAnalyzerConfig {
            outlier_threshold_sigma: 0.001, // very sensitive — many outliers
            min_cluster_size: 3,
            max_outlier_fraction: 0.2,
            ..Default::default()
        });
        for i in 0..20_u32 {
            a.add_point(format!("p{i}"), vec![i as f64], Some(ClusterId(0)));
        }
        a.set_clusters(vec![make_descriptor(0, vec![10.0])]);

        let outliers = a.detect_outliers();
        let cap = ((20_f64) * 0.2).ceil() as usize;
        assert!(
            outliers.len() <= cap,
            "outliers {} > cap {}",
            outliers.len(),
            cap
        );
    }

    // ── 27: local_density counts neighbors within radius ─────────────────────

    #[test]
    fn test_local_density_basic() {
        let mut a = EmbeddingClusterAnalyzer::new(EcaAnalyzerConfig {
            density_radius: 1.5,
            ..Default::default()
        });
        a.add_point("origin".into(), vec![0.0], None);
        a.add_point("near1".into(), vec![1.0], None);
        a.add_point("near2".into(), vec![-1.0], None);
        a.add_point("far".into(), vec![10.0], None);

        // origin should have 2 neighbors (near1 and near2)
        let density = a.local_density(0);
        assert!((density - 2.0).abs() < 1e-10);
    }

    // ── 28: local_density returns 0.0 for invalid index ──────────────────────

    #[test]
    fn test_local_density_invalid_index() {
        let a = make_analyzer();
        assert_eq!(a.local_density(999), 0.0);
    }

    // ── 29: cluster_evolution detects centroid shift ──────────────────────────

    #[test]
    fn test_cluster_evolution_shift() {
        let mut prev = make_analyzer();
        prev.set_clusters(vec![make_descriptor(0, vec![0.0, 0.0])]);

        let mut curr = make_analyzer();
        // shift centroid by 5.0 >> 0.1 threshold
        curr.set_clusters(vec![make_descriptor(0, vec![5.0, 0.0])]);

        let events = curr.cluster_evolution(&prev);
        assert!(!events.is_empty(), "Expected shift event");
        assert!(events[0].contains("shifted"), "Event: {}", events[0]);
    }

    // ── 30: cluster_evolution no event for tiny shift ────────────────────────

    #[test]
    fn test_cluster_evolution_no_shift() {
        let mut prev = make_analyzer();
        prev.set_clusters(vec![make_descriptor(0, vec![0.0, 0.0])]);

        let mut curr = make_analyzer();
        curr.set_clusters(vec![make_descriptor(0, vec![0.05, 0.0])]);

        let events = curr.cluster_evolution(&prev);
        assert!(events.is_empty(), "Expected no shift events");
    }

    // ── 31: cluster_evolution empty when no prev clusters ────────────────────

    #[test]
    fn test_cluster_evolution_empty_prev() {
        let prev = make_analyzer();
        let mut curr = make_analyzer();
        curr.set_clusters(vec![make_descriptor(0, vec![1.0])]);

        let events = curr.cluster_evolution(&prev);
        // prev has no clusters, so no events can be generated
        assert!(events.is_empty());
    }

    // ── 32: top_k_by_cluster returns correct count ───────────────────────────

    #[test]
    fn test_top_k_by_cluster_count() {
        let mut a = make_analyzer();
        for i in 0..10_u32 {
            a.add_point(format!("p{i}"), vec![i as f64], Some(ClusterId(0)));
        }
        a.set_clusters(vec![make_descriptor(0, vec![0.0])]);
        let top = a.top_k_by_cluster(ClusterId(0), 3);
        assert_eq!(top.len(), 3);
    }

    // ── 33: top_k_by_cluster ordered by distance ─────────────────────────────

    #[test]
    fn test_top_k_by_cluster_order() {
        let mut a = make_analyzer();
        // distances from centroid [0.0] will be 5,3,1 — sorted → 1,3,5
        a.add_point("far".into(), vec![5.0], Some(ClusterId(0)));
        a.add_point("mid".into(), vec![3.0], Some(ClusterId(0)));
        a.add_point("close".into(), vec![1.0], Some(ClusterId(0)));
        a.set_clusters(vec![make_descriptor(0, vec![0.0])]);

        let top = a.top_k_by_cluster(ClusterId(0), 3);
        assert_eq!(top[0].id, "close");
        assert_eq!(top[1].id, "mid");
        assert_eq!(top[2].id, "far");
    }

    // ── 34: top_k_by_cluster returns empty for unknown cluster ────────────────

    #[test]
    fn test_top_k_by_cluster_unknown() {
        let mut a = make_analyzer();
        a.add_point("p".into(), vec![1.0], Some(ClusterId(0)));
        a.set_clusters(vec![make_descriptor(0, vec![0.0])]);
        let top = a.top_k_by_cluster(ClusterId(99), 5);
        assert!(top.is_empty());
    }

    // ── 35: analyzer_stats correct point count ────────────────────────────────

    #[test]
    fn test_analyzer_stats_point_count() {
        let mut a = make_analyzer();
        a.add_point("a".into(), vec![1.0], None);
        a.add_point("b".into(), vec![2.0], None);
        let stats = a.analyzer_stats();
        assert_eq!(stats.point_count, 2);
    }

    // ── 36: analyzer_stats correct cluster count ──────────────────────────────

    #[test]
    fn test_analyzer_stats_cluster_count() {
        let mut a = make_analyzer();
        a.set_clusters(vec![
            make_descriptor(0, vec![0.0]),
            make_descriptor(1, vec![1.0]),
        ]);
        let stats = a.analyzer_stats();
        assert_eq!(stats.cluster_count, 2);
    }

    // ── 37: analyzer_stats avg_cluster_size ──────────────────────────────────

    #[test]
    fn test_analyzer_stats_avg_cluster_size() {
        let mut a = make_analyzer();
        for _ in 0..6 {
            a.add_point("p".into(), vec![0.0], None);
        }
        a.set_clusters(vec![
            make_descriptor(0, vec![0.0]),
            make_descriptor(1, vec![1.0]),
        ]);
        let stats = a.analyzer_stats();
        assert!((stats.avg_cluster_size - 3.0).abs() < 1e-10);
    }

    // ── 38: analyzer_stats outlier_count after detect_outliers ───────────────

    #[test]
    fn test_analyzer_stats_outlier_count() {
        let mut a = EmbeddingClusterAnalyzer::new(EcaAnalyzerConfig {
            min_cluster_size: 3,
            max_outlier_fraction: 1.0,
            ..Default::default()
        });
        a.add_point("p1".into(), vec![0.0], Some(ClusterId(0)));
        a.add_point("p2".into(), vec![0.1], Some(ClusterId(0)));
        a.set_clusters(vec![make_descriptor(0, vec![0.05])]);

        let outliers = a.detect_outliers();
        let expected = outliers.len();
        let stats = a.analyzer_stats();
        assert_eq!(stats.outlier_count, expected);
    }

    // ── 39: multiple set_clusters replaces previous ───────────────────────────

    #[test]
    fn test_set_clusters_replaces_previous() {
        let mut a = make_analyzer();
        a.set_clusters(vec![make_descriptor(0, vec![0.0])]);
        a.set_clusters(vec![
            make_descriptor(0, vec![0.0]),
            make_descriptor(1, vec![1.0]),
        ]);
        assert_eq!(a.clusters.len(), 2);
    }

    // ── 40: nearest cluster assignment uses cosine metric ─────────────────────

    #[test]
    fn test_assignment_uses_cosine() {
        let mut a = make_analyzer();
        // [1.0, 0.0] is closer (cosine) to [2.0, 0.0] than to [0.0, 1.0]
        a.add_point("p".into(), vec![1.0, 0.0], None);
        a.set_clusters(vec![
            make_descriptor(0, vec![0.0, 1.0]),
            make_descriptor(1, vec![2.0, 0.0]),
        ]);
        assert_eq!(a.points[0].cluster, Some(ClusterId(1)));
    }

    // ── 41: calinski_harabász positive for well-separated clusters ────────────

    #[test]
    fn test_calinski_harabasz_positive() {
        let mut a = make_analyzer();
        for i in 0..5_u32 {
            a.add_point(
                format!("a{i}"),
                vec![i as f64 * 0.01, 0.0],
                Some(ClusterId(0)),
            );
        }
        for i in 0..5_u32 {
            a.add_point(
                format!("b{i}"),
                vec![100.0 + i as f64 * 0.01, 0.0],
                Some(ClusterId(1)),
            );
        }
        a.set_clusters(vec![
            make_descriptor(0, vec![0.02, 0.0]),
            make_descriptor(1, vec![100.02, 0.0]),
        ]);
        let q = a.compute_cluster_quality();
        assert!(q.calinski_harabasz_score > 0.0);
    }

    // ── 42: davies_bouldin low for well-separated clusters ────────────────────

    #[test]
    fn test_davies_bouldin_well_separated() {
        let mut a = make_analyzer();
        for i in 0..5_u32 {
            a.add_point(format!("a{i}"), vec![i as f64 * 0.01], Some(ClusterId(0)));
        }
        for i in 0..5_u32 {
            a.add_point(
                format!("b{i}"),
                vec![1000.0 + i as f64 * 0.01],
                Some(ClusterId(1)),
            );
        }
        a.set_clusters(vec![
            make_descriptor(0, vec![0.02]),
            make_descriptor(1, vec![1000.02]),
        ]);
        let q = a.compute_cluster_quality();
        assert!(
            q.davies_bouldin_index < 0.1,
            "DB index: {}",
            q.davies_bouldin_index
        );
    }

    // ── 43: outlier score ordering (highest first) ────────────────────────────

    #[test]
    fn test_outlier_score_ordering() {
        let mut a = EmbeddingClusterAnalyzer::new(EcaAnalyzerConfig {
            outlier_threshold_sigma: 0.5,
            min_cluster_size: 2,
            max_outlier_fraction: 1.0,
            ..Default::default()
        });
        // 5 tight cluster members + 2 outliers at different distances
        for i in 0..5_u32 {
            a.add_point(format!("n{i}"), vec![i as f64 * 0.001], Some(ClusterId(0)));
        }
        a.add_point("out1".into(), vec![100.0], Some(ClusterId(0)));
        a.add_point("out2".into(), vec![200.0], Some(ClusterId(0)));
        a.set_clusters(vec![make_descriptor(0, vec![0.002])]);

        let outliers = a.detect_outliers();
        for window in outliers.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "Not sorted: {} < {}",
                window[0].score,
                window[1].score
            );
        }
    }

    // ── 44: ClusterDescriptor label stored correctly ──────────────────────────

    #[test]
    fn test_cluster_descriptor_label() {
        let mut d = make_descriptor(0, vec![1.0]);
        d.label = Some("science".to_string());
        assert_eq!(d.label.as_deref(), Some("science"));
    }

    // ── 45: EcaClusterPoint fields accessible ────────────────────────────────

    #[test]
    fn test_cluster_point_fields() {
        let p = EcaClusterPoint {
            id: "x".into(),
            embedding: vec![1.0, 2.0],
            cluster: Some(ClusterId(3)),
            distance_to_centroid: 0.5,
        };
        assert_eq!(p.id, "x");
        assert_eq!(p.cluster, Some(ClusterId(3)));
        assert!((p.distance_to_centroid - 0.5).abs() < 1e-10);
    }
}
