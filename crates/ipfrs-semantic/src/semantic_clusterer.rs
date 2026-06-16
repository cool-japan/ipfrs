//! `SemanticClusterer` — Multi-algorithm semantic vector clustering engine.
//!
//! Supports KMeans (with KMeans++ seeding), MiniBatchKMeans, DBSCAN, and
//! Agglomerative clustering over f64 embedding vectors. Uses xorshift64 for
//! all pseudo-random operations (no `rand` crate).

use std::fmt;
use thiserror::Error;

// ---------------------------------------------------------------------------
// PRNG — xorshift64 (seed=42 default)
// ---------------------------------------------------------------------------

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

// ---------------------------------------------------------------------------
// Public error type
// ---------------------------------------------------------------------------

/// Errors that can occur during clustering operations.
#[derive(Debug, Error)]
pub enum ClusterError {
    /// Not enough points to form the requested number of clusters.
    #[error("insufficient points: need at least {min}, got {got}")]
    InsufficientPoints { min: usize, got: usize },

    /// A point's embedding dimensionality does not match the clusterer's `dims`.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// All clusters ended up empty after an iteration.
    #[error("all clusters became empty")]
    EmptyClusters,

    /// A configuration parameter is invalid.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}

// ---------------------------------------------------------------------------
// Linkage criterion for agglomerative clustering
// ---------------------------------------------------------------------------

/// Linkage criterion used by agglomerative clustering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    /// Minimise variance (simplified: behaves like `Average` in this implementation).
    Ward,
    /// Distance between the two farthest members of each cluster.
    Complete,
    /// Mean of all pairwise distances between cluster members.
    Average,
    /// Distance between the two nearest members of each cluster.
    Single,
}

impl fmt::Display for Linkage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Linkage::Ward => write!(f, "ward"),
            Linkage::Complete => write!(f, "complete"),
            Linkage::Average => write!(f, "average"),
            Linkage::Single => write!(f, "single"),
        }
    }
}

// ---------------------------------------------------------------------------
// Algorithm selector
// ---------------------------------------------------------------------------

/// Which clustering algorithm the `SemanticClusterer` should use.
#[derive(Debug, Clone)]
pub enum ClusterAlgorithm {
    /// Standard KMeans with KMeans++ seeding.
    KMeans {
        /// Number of clusters to form.
        k: usize,
        /// Maximum number of EM iterations.
        max_iter: u32,
        /// Centroid-shift tolerance for early stopping.
        tolerance: f64,
    },
    /// Mini-batch variant of KMeans for large datasets.
    MiniBatchKMeans {
        /// Number of clusters.
        k: usize,
        /// Points sampled per iteration.
        batch_size: usize,
        /// Maximum iterations.
        max_iter: u32,
    },
    /// Density-Based Spatial Clustering of Applications with Noise.
    DBSCAN {
        /// Neighbourhood radius.
        eps: f64,
        /// Minimum neighbours (including self) to be a core point.
        min_samples: usize,
    },
    /// Hierarchical agglomerative clustering.
    Agglomerative {
        /// Target number of clusters.
        k: usize,
        /// Linkage criterion.
        linkage: Linkage,
    },
}

impl fmt::Display for ClusterAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClusterAlgorithm::KMeans { k, .. } => write!(f, "kmeans(k={k})"),
            ClusterAlgorithm::MiniBatchKMeans { k, .. } => {
                write!(f, "mini_batch_kmeans(k={k})")
            }
            ClusterAlgorithm::DBSCAN { eps, min_samples } => {
                write!(f, "dbscan(eps={eps},min_samples={min_samples})")
            }
            ClusterAlgorithm::Agglomerative { k, linkage } => {
                write!(f, "agglomerative(k={k},linkage={linkage})")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Core data structures
// ---------------------------------------------------------------------------

/// A single embedding vector with an associated string identifier.
#[derive(Debug, Clone)]
pub struct ScClusterPoint {
    /// Unique identifier (e.g. CID or file path).
    pub id: String,
    /// The embedding vector.
    pub embedding: Vec<f64>,
    /// Assigned cluster index, or `None` for noise / unclustered points.
    pub cluster_id: Option<usize>,
}

impl ScClusterPoint {
    /// Construct a new unassigned cluster point.
    pub fn new(id: impl Into<String>, embedding: Vec<f64>) -> Self {
        Self {
            id: id.into(),
            embedding,
            cluster_id: None,
        }
    }
}

/// A cluster produced by the clusterer.
#[derive(Debug, Clone)]
pub struct ScCluster {
    /// Zero-based cluster index.
    pub id: usize,
    /// Centroid vector (mean of all member embeddings).
    pub centroid: Vec<f64>,
    /// IDs of all member points.
    pub member_ids: Vec<String>,
    /// Sum of squared Euclidean distances from each member to the centroid.
    pub inertia: f64,
}

impl ScCluster {
    /// Number of members in this cluster.
    pub fn size(&self) -> usize {
        self.member_ids.len()
    }

    /// `true` when the cluster contains no members.
    pub fn is_empty(&self) -> bool {
        self.member_ids.is_empty()
    }
}

/// The full result of a clustering run.
#[derive(Debug, Clone)]
pub struct ScClusteringResult {
    /// Non-noise clusters produced.
    pub clusters: Vec<ScCluster>,
    /// Point IDs that were classified as noise (DBSCAN) or left unclustered.
    pub noise_ids: Vec<String>,
    /// Human-readable label for the algorithm used.
    pub algorithm: String,
    /// Mean silhouette coefficient in `[-1, 1]`.
    pub silhouette_score: f64,
    /// Total inertia (sum of all per-cluster inertias).
    pub inertia: f64,
    /// Number of EM / optimisation iterations performed.
    pub iterations: u32,
}

/// Summary statistics for a clustering result.
#[derive(Debug, Clone)]
pub struct ScClustererStats {
    /// Total points assigned to a cluster (excluding noise).
    pub total_clustered: usize,
    /// Number of noise points.
    pub noise_count: usize,
    /// Mean cluster size (excluding noise).
    pub avg_cluster_size: f64,
    /// Size of the largest cluster.
    pub largest_cluster: usize,
    /// Size of the smallest cluster (0 when there are no clusters).
    pub smallest_cluster: usize,
}

// ---------------------------------------------------------------------------
// Main engine
// ---------------------------------------------------------------------------

/// Multi-algorithm semantic vector clustering engine.
///
/// ```rust
/// use ipfrs_semantic::semantic_clusterer::{
///     SemanticClusterer, ClusterAlgorithm, ScClusterPoint,
/// };
///
/// let points: Vec<ScClusterPoint> = (0..20)
///     .map(|i| ScClusterPoint::new(format!("p{i}"), vec![(i % 4) as f64, (i / 4) as f64]))
///     .collect();
///
/// let clusterer = SemanticClusterer::new(
///     ClusterAlgorithm::KMeans { k: 4, max_iter: 100, tolerance: 1e-6 },
///     2,
/// );
/// let result = clusterer.fit(&points).expect("clustering failed");
/// assert_eq!(result.clusters.len(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct SemanticClusterer {
    /// Algorithm configuration.
    pub algorithm: ClusterAlgorithm,
    /// Expected embedding dimensionality.
    pub dims: usize,
}

impl SemanticClusterer {
    /// Create a new clusterer with the given algorithm and embedding dimension.
    pub fn new(algorithm: ClusterAlgorithm, dims: usize) -> Self {
        Self { algorithm, dims }
    }

    // -----------------------------------------------------------------------
    // Public interface
    // -----------------------------------------------------------------------

    /// Run clustering on the provided points.
    ///
    /// Returns a `ScClusteringResult` on success.
    pub fn fit(&self, points: &[ScClusterPoint]) -> Result<ScClusteringResult, ClusterError> {
        self.validate_points(points)?;
        let algorithm_label = self.algorithm.to_string();
        let result = match &self.algorithm {
            ClusterAlgorithm::KMeans {
                k,
                max_iter,
                tolerance,
            } => self.fit_kmeans(points, *k, *max_iter, *tolerance),
            ClusterAlgorithm::MiniBatchKMeans {
                k,
                batch_size,
                max_iter,
            } => self.fit_mini_batch_kmeans(points, *k, *batch_size, *max_iter),
            ClusterAlgorithm::DBSCAN { eps, min_samples } => {
                self.fit_dbscan(points, *eps, *min_samples)
            }
            ClusterAlgorithm::Agglomerative { k, linkage } => {
                self.fit_agglomerative(points, *k, *linkage)
            }
        }?;

        // Build final result with silhouette score
        let mut final_result = result;
        final_result.algorithm = algorithm_label;
        let tagged = tag_points(points, &final_result);
        final_result.silhouette_score = Self::silhouette_score(&tagged, &final_result);
        Ok(final_result)
    }

    /// Assign a new point to the nearest cluster centroid in `result`.
    ///
    /// Returns `None` when `result` has no clusters.
    pub fn predict(&self, point: &ScClusterPoint, result: &ScClusteringResult) -> Option<usize> {
        if result.clusters.is_empty() {
            return None;
        }
        let mut best_id = 0usize;
        let mut best_dist = f64::MAX;
        for cluster in &result.clusters {
            let d = Self::euclidean_distance(&point.embedding, &cluster.centroid);
            if d < best_dist {
                best_dist = d;
                best_id = cluster.id;
            }
        }
        Some(best_id)
    }

    /// Cosine distance: `1 - cosine_similarity(a, b)`.
    ///
    /// Returns `1.0` for zero vectors.
    pub fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if na == 0.0 || nb == 0.0 {
            return 1.0;
        }
        1.0 - (dot / (na * nb)).clamp(-1.0, 1.0)
    }

    /// Euclidean distance between two vectors of equal length.
    pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Mean silhouette coefficient for the clustering.
    ///
    /// Uses a random subset of up to 100 points when the dataset is large.
    pub fn silhouette_score(points: &[ScClusterPoint], result: &ScClusteringResult) -> f64 {
        if result.clusters.len() < 2 {
            return 0.0;
        }
        // Build (point, cluster_id) pairs — noise excluded
        let assigned: Vec<(&ScClusterPoint, usize)> = points
            .iter()
            .filter_map(|p| p.cluster_id.map(|c| (p, c)))
            .collect();

        if assigned.len() < 2 {
            return 0.0;
        }

        // Subsample deterministically when large
        let sample: Vec<&(&ScClusterPoint, usize)> = if assigned.len() > 100 {
            let mut state: u64 = 42;
            let mut indices: Vec<usize> = (0..assigned.len()).collect();
            // Fisher-Yates partial shuffle
            for i in 0..100 {
                let j = i + (xorshift64(&mut state) as usize % (assigned.len() - i));
                indices.swap(i, j);
            }
            indices[..100].iter().map(|&i| &assigned[i]).collect()
        } else {
            assigned.iter().collect()
        };

        let scores: Vec<f64> = sample
            .iter()
            .filter_map(|(p, cid)| silhouette_one(p, *cid, &assigned))
            .collect();

        if scores.is_empty() {
            return 0.0;
        }
        scores.iter().sum::<f64>() / scores.len() as f64
    }

    /// Compute the centroid (element-wise mean) of a set of embeddings.
    ///
    /// Returns a zero vector when `embeddings` is empty or their slices are empty.
    pub fn compute_centroid(embeddings: &[&[f64]]) -> Vec<f64> {
        if embeddings.is_empty() {
            return Vec::new();
        }
        let dims = embeddings[0].len();
        if dims == 0 {
            return Vec::new();
        }
        let mut centroid = vec![0.0f64; dims];
        for emb in embeddings {
            for (c, v) in centroid.iter_mut().zip(emb.iter()) {
                *c += v;
            }
        }
        let n = embeddings.len() as f64;
        centroid.iter_mut().for_each(|c| *c /= n);
        centroid
    }

    /// Summary statistics for a completed clustering result.
    pub fn stats(result: &ScClusteringResult) -> ScClustererStats {
        let total_clustered: usize = result.clusters.iter().map(|c| c.member_ids.len()).sum();
        let noise_count = result.noise_ids.len();
        let avg_cluster_size = if result.clusters.is_empty() {
            0.0
        } else {
            total_clustered as f64 / result.clusters.len() as f64
        };
        let largest_cluster = result
            .clusters
            .iter()
            .map(|c| c.member_ids.len())
            .max()
            .unwrap_or(0);
        let smallest_cluster = result
            .clusters
            .iter()
            .map(|c| c.member_ids.len())
            .min()
            .unwrap_or(0);
        ScClustererStats {
            total_clustered,
            noise_count,
            avg_cluster_size,
            largest_cluster,
            smallest_cluster,
        }
    }

    // -----------------------------------------------------------------------
    // Validation helpers
    // -----------------------------------------------------------------------

    fn validate_points(&self, points: &[ScClusterPoint]) -> Result<(), ClusterError> {
        for p in points {
            if p.embedding.len() != self.dims {
                return Err(ClusterError::DimensionMismatch {
                    expected: self.dims,
                    got: p.embedding.len(),
                });
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // KMeans
    // -----------------------------------------------------------------------

    fn fit_kmeans(
        &self,
        points: &[ScClusterPoint],
        k: usize,
        max_iter: u32,
        tolerance: f64,
    ) -> Result<ScClusteringResult, ClusterError> {
        if k == 0 {
            return Err(ClusterError::InvalidParameter("k must be > 0".into()));
        }
        if points.len() < k {
            return Err(ClusterError::InsufficientPoints {
                min: k,
                got: points.len(),
            });
        }

        let mut centroids = kmeans_plus_plus_init(points, k, 42);
        let mut assignments = vec![0usize; points.len()];
        let mut iterations = 0u32;

        for _ in 0..max_iter {
            iterations += 1;

            // Assignment step
            for (i, p) in points.iter().enumerate() {
                assignments[i] = nearest_centroid(&p.embedding, &centroids);
            }

            // Update step
            let new_centroids = recompute_centroids(points, &assignments, k, self.dims);

            // Convergence check
            let max_shift = centroids
                .iter()
                .zip(new_centroids.iter())
                .map(|(old, new)| Self::euclidean_distance(old, new))
                .fold(0.0f64, f64::max);

            centroids = new_centroids;
            if max_shift < tolerance {
                break;
            }
        }

        build_result_from_centroids(points, &assignments, centroids, iterations)
    }

    // -----------------------------------------------------------------------
    // MiniBatchKMeans
    // -----------------------------------------------------------------------

    fn fit_mini_batch_kmeans(
        &self,
        points: &[ScClusterPoint],
        k: usize,
        batch_size: usize,
        max_iter: u32,
    ) -> Result<ScClusteringResult, ClusterError> {
        if k == 0 {
            return Err(ClusterError::InvalidParameter("k must be > 0".into()));
        }
        if points.len() < k {
            return Err(ClusterError::InsufficientPoints {
                min: k,
                got: points.len(),
            });
        }
        let effective_batch = batch_size.min(points.len());
        if effective_batch == 0 {
            return Err(ClusterError::InvalidParameter(
                "batch_size must be > 0".into(),
            ));
        }

        let mut centroids = kmeans_plus_plus_init(points, k, 42);
        // Per-centroid update counts for incremental averaging
        let mut counts = vec![1u64; k];
        let mut state: u64 = 0x_dead_beef_cafe_babe_u64;
        let mut iterations = 0u32;

        for _ in 0..max_iter {
            iterations += 1;

            // Sample batch (with replacement via xorshift64)
            let batch_indices: Vec<usize> = (0..effective_batch)
                .map(|_| xorshift64(&mut state) as usize % points.len())
                .collect();

            // Assign batch points to nearest centroid
            let batch_assignments: Vec<usize> = batch_indices
                .iter()
                .map(|&i| nearest_centroid(&points[i].embedding, &centroids))
                .collect();

            // Incremental centroid update
            for (&point_idx, &cid) in batch_indices.iter().zip(batch_assignments.iter()) {
                counts[cid] += 1;
                let lr = 1.0 / counts[cid] as f64;
                let emb = &points[point_idx].embedding;
                for (c, v) in centroids[cid].iter_mut().zip(emb.iter()) {
                    *c += lr * (v - *c);
                }
            }
        }

        // Final assignment of all points
        let assignments: Vec<usize> = points
            .iter()
            .map(|p| nearest_centroid(&p.embedding, &centroids))
            .collect();

        build_result_from_centroids(points, &assignments, centroids, iterations)
    }

    // -----------------------------------------------------------------------
    // DBSCAN
    // -----------------------------------------------------------------------

    fn fit_dbscan(
        &self,
        points: &[ScClusterPoint],
        eps: f64,
        min_samples: usize,
    ) -> Result<ScClusteringResult, ClusterError> {
        if eps <= 0.0 {
            return Err(ClusterError::InvalidParameter(
                "eps must be positive".into(),
            ));
        }
        if min_samples == 0 {
            return Err(ClusterError::InvalidParameter(
                "min_samples must be > 0".into(),
            ));
        }

        let n = points.len();
        // -1 = unvisited, usize::MAX = noise, otherwise cluster index
        let mut label: Vec<Option<usize>> = vec![None; n];
        let mut cluster_id = 0usize;

        // Pre-compute eps-neighbourhoods
        let neighbours: Vec<Vec<usize>> = (0..n)
            .map(|i| {
                (0..n)
                    .filter(|&j| {
                        Self::euclidean_distance(&points[i].embedding, &points[j].embedding) <= eps
                    })
                    .collect()
            })
            .collect();

        for i in 0..n {
            if label[i].is_some() {
                continue;
            }
            if neighbours[i].len() < min_samples {
                // Mark as noise temporarily (will be revised if reachable)
                label[i] = Some(usize::MAX);
                continue;
            }
            // Expand cluster
            label[i] = Some(cluster_id);
            let mut queue: Vec<usize> = neighbours[i].clone();
            let mut head = 0;
            while head < queue.len() {
                let q = queue[head];
                head += 1;
                if label[q] == Some(usize::MAX) {
                    // Border point — assign to cluster
                    label[q] = Some(cluster_id);
                }
                if label[q].is_some() && label[q] != Some(usize::MAX) {
                    continue;
                }
                label[q] = Some(cluster_id);
                if neighbours[q].len() >= min_samples {
                    for &nb in &neighbours[q] {
                        if label[nb].is_none() || label[nb] == Some(usize::MAX) {
                            queue.push(nb);
                        }
                    }
                }
            }
            cluster_id += 1;
        }

        // Collect results
        let k = cluster_id;
        let mut cluster_members: Vec<Vec<String>> = vec![Vec::new(); k];
        let mut noise_ids: Vec<String> = Vec::new();

        for (i, lbl) in label.iter().enumerate() {
            match lbl {
                None | Some(usize::MAX) => noise_ids.push(points[i].id.clone()),
                &Some(cid) => cluster_members[cid].push(points[i].id.clone()),
            }
        }

        // Build ScCluster for each group
        let clusters: Vec<ScCluster> = cluster_members
            .into_iter()
            .enumerate()
            .map(|(cid, members)| {
                let embeddings: Vec<&[f64]> = members
                    .iter()
                    .filter_map(|id| points.iter().find(|p| &p.id == id))
                    .map(|p| p.embedding.as_slice())
                    .collect();
                let centroid = Self::compute_centroid(&embeddings);
                let inertia = embeddings
                    .iter()
                    .map(|e| Self::euclidean_distance(e, &centroid).powi(2))
                    .sum();
                ScCluster {
                    id: cid,
                    centroid,
                    member_ids: members,
                    inertia,
                }
            })
            .collect();

        let total_inertia: f64 = clusters.iter().map(|c| c.inertia).sum();

        Ok(ScClusteringResult {
            clusters,
            noise_ids,
            algorithm: String::new(), // filled by caller
            silhouette_score: 0.0,    // filled by caller
            inertia: total_inertia,
            iterations: 1,
        })
    }

    // -----------------------------------------------------------------------
    // Agglomerative
    // -----------------------------------------------------------------------

    fn fit_agglomerative(
        &self,
        points: &[ScClusterPoint],
        k: usize,
        linkage: Linkage,
    ) -> Result<ScClusteringResult, ClusterError> {
        if k == 0 {
            return Err(ClusterError::InvalidParameter("k must be > 0".into()));
        }
        if points.len() < k {
            return Err(ClusterError::InsufficientPoints {
                min: k,
                got: points.len(),
            });
        }

        let n = points.len();
        // Each point starts in its own cluster (represented as a set of point indices)
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

        let mut iterations = 0u32;

        while clusters.len() > k {
            iterations += 1;
            // Find the pair with minimum linkage distance
            let nc = clusters.len();
            let mut min_dist = f64::MAX;
            let mut merge_a = 0usize;
            let mut merge_b = 1usize;

            for a in 0..nc {
                for b in (a + 1)..nc {
                    let d = linkage_distance(&clusters[a], &clusters[b], points, linkage);
                    if d < min_dist {
                        min_dist = d;
                        merge_a = a;
                        merge_b = b;
                    }
                }
            }

            // Merge b into a
            let b_members = clusters.remove(merge_b);
            clusters[merge_a].extend(b_members);
        }

        // Build ScCluster objects
        let result_clusters: Vec<ScCluster> = clusters
            .into_iter()
            .enumerate()
            .map(|(cid, member_indices)| {
                let embeddings: Vec<&[f64]> = member_indices
                    .iter()
                    .map(|&i| points[i].embedding.as_slice())
                    .collect();
                let centroid = Self::compute_centroid(&embeddings);
                let inertia = embeddings
                    .iter()
                    .map(|e| Self::euclidean_distance(e, &centroid).powi(2))
                    .sum();
                ScCluster {
                    id: cid,
                    centroid,
                    member_ids: member_indices
                        .iter()
                        .map(|&i| points[i].id.clone())
                        .collect(),
                    inertia,
                }
            })
            .collect();

        let total_inertia: f64 = result_clusters.iter().map(|c| c.inertia).sum();

        Ok(ScClusteringResult {
            clusters: result_clusters,
            noise_ids: Vec::new(),
            algorithm: String::new(),
            silhouette_score: 0.0,
            inertia: total_inertia,
            iterations,
        })
    }
}

// ---------------------------------------------------------------------------
// Module-level helpers (not part of public API surface but accessible)
// ---------------------------------------------------------------------------

/// KMeans++ centroid initialisation using xorshift64.
fn kmeans_plus_plus_init(points: &[ScClusterPoint], k: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = if seed == 0 { 1 } else { seed };
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);

    // Pick first centroid uniformly at random
    let first = xorshift64(&mut state) as usize % points.len();
    centroids.push(points[first].embedding.clone());

    for _ in 1..k {
        // Compute squared distances to the nearest existing centroid
        let dists: Vec<f64> = points
            .iter()
            .map(|p| {
                centroids
                    .iter()
                    .map(|c| SemanticClusterer::euclidean_distance(&p.embedding, c).powi(2))
                    .fold(f64::MAX, f64::min)
            })
            .collect();

        let total: f64 = dists.iter().sum();
        if total == 0.0 {
            // All points are coincident; pick randomly
            let idx = xorshift64(&mut state) as usize % points.len();
            centroids.push(points[idx].embedding.clone());
            continue;
        }

        // Weighted random draw via xorshift64
        let threshold = (xorshift64(&mut state) as f64 / u64::MAX as f64) * total;
        let mut cumulative = 0.0;
        let mut chosen = points.len() - 1;
        for (i, &d) in dists.iter().enumerate() {
            cumulative += d;
            if cumulative >= threshold {
                chosen = i;
                break;
            }
        }
        centroids.push(points[chosen].embedding.clone());
    }

    centroids
}

/// Return the index of the nearest centroid to `embedding`.
fn nearest_centroid(embedding: &[f64], centroids: &[Vec<f64>]) -> usize {
    let mut best = 0usize;
    let mut best_dist = f64::MAX;
    for (i, c) in centroids.iter().enumerate() {
        let d = SemanticClusterer::euclidean_distance(embedding, c);
        if d < best_dist {
            best_dist = d;
            best = i;
        }
    }
    best
}

/// Recompute centroids as the mean of all assigned points.
/// Empty clusters retain their previous position.
fn recompute_centroids(
    points: &[ScClusterPoint],
    assignments: &[usize],
    k: usize,
    dims: usize,
) -> Vec<Vec<f64>> {
    let mut sums = vec![vec![0.0f64; dims]; k];
    let mut counts = vec![0usize; k];
    for (p, &cid) in points.iter().zip(assignments.iter()) {
        for (s, v) in sums[cid].iter_mut().zip(p.embedding.iter()) {
            *s += v;
        }
        counts[cid] += 1;
    }
    sums.iter_mut()
        .zip(counts.iter())
        .map(|(sum, &cnt)| {
            if cnt > 0 {
                sum.iter().map(|&s| s / cnt as f64).collect()
            } else {
                sum.clone()
            }
        })
        .collect()
}

/// Build a `ScClusteringResult` from a flat assignment array and centroid list.
fn build_result_from_centroids(
    points: &[ScClusterPoint],
    assignments: &[usize],
    centroids: Vec<Vec<f64>>,
    iterations: u32,
) -> Result<ScClusteringResult, ClusterError> {
    let k = centroids.len();
    let mut member_sets: Vec<Vec<String>> = vec![Vec::new(); k];
    for (p, &cid) in points.iter().zip(assignments.iter()) {
        member_sets[cid].push(p.id.clone());
    }

    let clusters: Vec<ScCluster> = centroids
        .into_iter()
        .enumerate()
        .map(|(cid, centroid)| {
            let members = &member_sets[cid];
            let inertia: f64 = members
                .iter()
                .filter_map(|id| points.iter().find(|p| &p.id == id))
                .map(|p| SemanticClusterer::euclidean_distance(&p.embedding, &centroid).powi(2))
                .sum();
            ScCluster {
                id: cid,
                centroid,
                member_ids: members.clone(),
                inertia,
            }
        })
        .collect();

    if clusters.iter().all(|c| c.is_empty()) {
        return Err(ClusterError::EmptyClusters);
    }

    let total_inertia: f64 = clusters.iter().map(|c| c.inertia).sum();

    Ok(ScClusteringResult {
        clusters,
        noise_ids: Vec::new(),
        algorithm: String::new(),
        silhouette_score: 0.0,
        inertia: total_inertia,
        iterations,
    })
}

/// Copy cluster assignments back into a fresh `ScClusterPoint` slice.
fn tag_points(points: &[ScClusterPoint], result: &ScClusteringResult) -> Vec<ScClusterPoint> {
    let mut tagged: Vec<ScClusterPoint> = points.to_vec();
    for p in &mut tagged {
        p.cluster_id = None;
    }
    for cluster in &result.clusters {
        for id in &cluster.member_ids {
            if let Some(tp) = tagged.iter_mut().find(|p| &p.id == id) {
                tp.cluster_id = Some(cluster.id);
            }
        }
    }
    tagged
}

/// Silhouette coefficient for a single point.
fn silhouette_one(
    point: &ScClusterPoint,
    cid: usize,
    all: &[(&ScClusterPoint, usize)],
) -> Option<f64> {
    // Mean distance to points in the same cluster (a)
    let same: Vec<f64> = all
        .iter()
        .filter(|(p, c)| *c == cid && p.id != point.id)
        .map(|(p, _)| SemanticClusterer::euclidean_distance(&point.embedding, &p.embedding))
        .collect();

    let a = if same.is_empty() {
        return None;
    } else {
        same.iter().sum::<f64>() / same.len() as f64
    };

    // Nearest-other-cluster mean distance (b)
    let other_clusters: std::collections::HashSet<usize> = all
        .iter()
        .filter(|(_, c)| *c != cid)
        .map(|(_, c)| *c)
        .collect();

    let b = other_clusters
        .iter()
        .map(|&oc| {
            let dists: Vec<f64> = all
                .iter()
                .filter(|(_, c)| *c == oc)
                .map(|(p, _)| SemanticClusterer::euclidean_distance(&point.embedding, &p.embedding))
                .collect();
            if dists.is_empty() {
                f64::MAX
            } else {
                dists.iter().sum::<f64>() / dists.len() as f64
            }
        })
        .fold(f64::MAX, f64::min);

    if b == f64::MAX {
        return None;
    }

    let denom = a.max(b);
    if denom == 0.0 {
        Some(0.0)
    } else {
        Some((b - a) / denom)
    }
}

/// Linkage distance between two clusters identified by their point-index sets.
fn linkage_distance(a: &[usize], b: &[usize], points: &[ScClusterPoint], linkage: Linkage) -> f64 {
    let mut dists: Vec<f64> = Vec::with_capacity(a.len() * b.len());
    for &ai in a {
        for &bi in b {
            dists.push(SemanticClusterer::euclidean_distance(
                &points[ai].embedding,
                &points[bi].embedding,
            ));
        }
    }
    if dists.is_empty() {
        return 0.0;
    }
    match linkage {
        Linkage::Single => dists.iter().cloned().fold(f64::MAX, f64::min),
        Linkage::Complete => dists.iter().cloned().fold(f64::MIN, f64::max),
        Linkage::Average | Linkage::Ward => dists.iter().sum::<f64>() / dists.len() as f64,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{
        kmeans_plus_plus_init, tag_points, xorshift64, ClusterAlgorithm, ClusterError, Linkage,
        ScClusterPoint, SemanticClusterer,
    };

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn pts(coords: &[(f64, f64)]) -> Vec<ScClusterPoint> {
        coords
            .iter()
            .enumerate()
            .map(|(i, &(x, y))| ScClusterPoint::new(format!("p{i}"), vec![x, y]))
            .collect()
    }

    fn well_separated_2d() -> Vec<ScClusterPoint> {
        // Three well-separated groups
        let mut v = Vec::new();
        for i in 0..10 {
            v.push(ScClusterPoint::new(
                format!("a{i}"),
                vec![i as f64 * 0.01, i as f64 * 0.01],
            ));
        }
        for i in 0..10 {
            v.push(ScClusterPoint::new(
                format!("b{i}"),
                vec![10.0 + i as f64 * 0.01, 10.0 + i as f64 * 0.01],
            ));
        }
        for i in 0..10 {
            v.push(ScClusterPoint::new(
                format!("c{i}"),
                vec![20.0 + i as f64 * 0.01, 20.0 + i as f64 * 0.01],
            ));
        }
        v
    }

    // -----------------------------------------------------------------------
    // 1. xorshift64 produces non-zero output from seed 42
    // -----------------------------------------------------------------------
    #[test]
    fn test_xorshift64_nonzero() {
        let mut s: u64 = 42;
        let v = xorshift64(&mut s);
        assert_ne!(v, 0);
    }

    // -----------------------------------------------------------------------
    // 2. xorshift64 produces distinct successive values
    // -----------------------------------------------------------------------
    #[test]
    fn test_xorshift64_distinct() {
        let mut s: u64 = 42;
        let a = xorshift64(&mut s);
        let b = xorshift64(&mut s);
        assert_ne!(a, b);
    }

    // -----------------------------------------------------------------------
    // 3. euclidean_distance: identical vectors → 0
    // -----------------------------------------------------------------------
    #[test]
    fn test_euclidean_self_distance_zero() {
        let v = vec![1.0, 2.0, 3.0];
        assert_eq!(SemanticClusterer::euclidean_distance(&v, &v), 0.0);
    }

    // -----------------------------------------------------------------------
    // 4. euclidean_distance: known value
    // -----------------------------------------------------------------------
    #[test]
    fn test_euclidean_known() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let d = SemanticClusterer::euclidean_distance(&a, &b);
        assert!((d - 5.0).abs() < 1e-10, "expected 5, got {d}");
    }

    // -----------------------------------------------------------------------
    // 5. cosine_distance: identical non-zero vectors → 0
    // -----------------------------------------------------------------------
    #[test]
    fn test_cosine_distance_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let d = SemanticClusterer::cosine_distance(&v, &v);
        assert!(d.abs() < 1e-10, "expected 0, got {d}");
    }

    // -----------------------------------------------------------------------
    // 6. cosine_distance: orthogonal vectors → 1
    // -----------------------------------------------------------------------
    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let d = SemanticClusterer::cosine_distance(&a, &b);
        assert!((d - 1.0).abs() < 1e-10, "expected 1, got {d}");
    }

    // -----------------------------------------------------------------------
    // 7. cosine_distance: zero vector → 1
    // -----------------------------------------------------------------------
    #[test]
    fn test_cosine_distance_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        assert_eq!(SemanticClusterer::cosine_distance(&a, &b), 1.0);
    }

    // -----------------------------------------------------------------------
    // 8. compute_centroid: empty slice → empty vec
    // -----------------------------------------------------------------------
    #[test]
    fn test_compute_centroid_empty() {
        let c = SemanticClusterer::compute_centroid(&[]);
        assert!(c.is_empty());
    }

    // -----------------------------------------------------------------------
    // 9. compute_centroid: known value
    // -----------------------------------------------------------------------
    #[test]
    fn test_compute_centroid_known() {
        let a = [0.0f64, 2.0];
        let b = [2.0f64, 0.0];
        let c = SemanticClusterer::compute_centroid(&[&a, &b]);
        assert!((c[0] - 1.0).abs() < 1e-10);
        assert!((c[1] - 1.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // 10. KMeans: produces exactly k clusters
    // -----------------------------------------------------------------------
    #[test]
    fn test_kmeans_cluster_count() {
        let points = well_separated_2d();
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::KMeans {
                k: 3,
                max_iter: 100,
                tolerance: 1e-6,
            },
            2,
        );
        let result = clusterer.fit(&points).expect("fit failed");
        assert_eq!(result.clusters.len(), 3);
    }

    // -----------------------------------------------------------------------
    // 11. KMeans: all points assigned (no noise)
    // -----------------------------------------------------------------------
    #[test]
    fn test_kmeans_no_noise() {
        let points = well_separated_2d();
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::KMeans {
                k: 3,
                max_iter: 100,
                tolerance: 1e-6,
            },
            2,
        );
        let result = clusterer.fit(&points).expect("fit failed");
        assert!(result.noise_ids.is_empty());
    }

    // -----------------------------------------------------------------------
    // 12. KMeans: total members == total points
    // -----------------------------------------------------------------------
    #[test]
    fn test_kmeans_all_points_assigned() {
        let points = well_separated_2d();
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::KMeans {
                k: 3,
                max_iter: 100,
                tolerance: 1e-6,
            },
            2,
        );
        let result = clusterer.fit(&points).expect("fit failed");
        let total: usize = result.clusters.iter().map(|c| c.member_ids.len()).sum();
        assert_eq!(total, points.len());
    }

    // -----------------------------------------------------------------------
    // 13. KMeans: insufficient points error
    // -----------------------------------------------------------------------
    #[test]
    fn test_kmeans_insufficient_points() {
        let points = pts(&[(0.0, 0.0), (1.0, 1.0)]);
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::KMeans {
                k: 5,
                max_iter: 10,
                tolerance: 1e-4,
            },
            2,
        );
        let err = clusterer.fit(&points).expect_err("should fail");
        assert!(matches!(err, ClusterError::InsufficientPoints { .. }));
    }

    // -----------------------------------------------------------------------
    // 14. KMeans: dimension mismatch error
    // -----------------------------------------------------------------------
    #[test]
    fn test_kmeans_dimension_mismatch() {
        let points = vec![ScClusterPoint::new("x", vec![1.0, 2.0, 3.0])];
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::KMeans {
                k: 1,
                max_iter: 10,
                tolerance: 1e-4,
            },
            2,
        );
        let err = clusterer.fit(&points).expect_err("should fail");
        assert!(matches!(err, ClusterError::DimensionMismatch { .. }));
    }

    // -----------------------------------------------------------------------
    // 15. KMeans k=0 → InvalidParameter
    // -----------------------------------------------------------------------
    #[test]
    fn test_kmeans_k_zero() {
        let points = pts(&[(0.0, 0.0)]);
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::KMeans {
                k: 0,
                max_iter: 10,
                tolerance: 1e-4,
            },
            2,
        );
        let err = clusterer.fit(&points).expect_err("should fail");
        assert!(matches!(err, ClusterError::InvalidParameter(_)));
    }

    // -----------------------------------------------------------------------
    // 16. KMeans: inertia is non-negative
    // -----------------------------------------------------------------------
    #[test]
    fn test_kmeans_inertia_nonneg() {
        let points = well_separated_2d();
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::KMeans {
                k: 3,
                max_iter: 100,
                tolerance: 1e-6,
            },
            2,
        );
        let result = clusterer.fit(&points).expect("fit failed");
        assert!(result.inertia >= 0.0);
    }

    // -----------------------------------------------------------------------
    // 17. KMeans: silhouette_score in [-1, 1]
    // -----------------------------------------------------------------------
    #[test]
    fn test_kmeans_silhouette_range() {
        let points = well_separated_2d();
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::KMeans {
                k: 3,
                max_iter: 100,
                tolerance: 1e-6,
            },
            2,
        );
        let result = clusterer.fit(&points).expect("fit failed");
        assert!(
            (-1.0..=1.0).contains(&result.silhouette_score),
            "score={}",
            result.silhouette_score
        );
    }

    // -----------------------------------------------------------------------
    // 18. KMeans: k=1 special case — all points in one cluster
    // -----------------------------------------------------------------------
    #[test]
    fn test_kmeans_k1() {
        let points = pts(&[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]);
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::KMeans {
                k: 1,
                max_iter: 10,
                tolerance: 1e-4,
            },
            2,
        );
        let result = clusterer.fit(&points).expect("fit failed");
        assert_eq!(result.clusters.len(), 1);
        assert_eq!(result.clusters[0].member_ids.len(), 3);
    }

    // -----------------------------------------------------------------------
    // 19. MiniBatchKMeans: produces k clusters
    // -----------------------------------------------------------------------
    #[test]
    fn test_mini_batch_kmeans_cluster_count() {
        let points = well_separated_2d();
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::MiniBatchKMeans {
                k: 3,
                batch_size: 10,
                max_iter: 200,
            },
            2,
        );
        let result = clusterer.fit(&points).expect("fit failed");
        assert_eq!(result.clusters.len(), 3);
    }

    // -----------------------------------------------------------------------
    // 20. MiniBatchKMeans: all points assigned
    // -----------------------------------------------------------------------
    #[test]
    fn test_mini_batch_all_assigned() {
        let points = well_separated_2d();
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::MiniBatchKMeans {
                k: 3,
                batch_size: 10,
                max_iter: 200,
            },
            2,
        );
        let result = clusterer.fit(&points).expect("fit failed");
        let total: usize = result.clusters.iter().map(|c| c.member_ids.len()).sum();
        assert_eq!(total + result.noise_ids.len(), points.len());
    }

    // -----------------------------------------------------------------------
    // 21. DBSCAN: noise detection
    // -----------------------------------------------------------------------
    #[test]
    fn test_dbscan_noise_detection() {
        let mut points = well_separated_2d();
        // Add an isolated outlier
        points.push(ScClusterPoint::new("outlier", vec![100.0, 100.0]));
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::DBSCAN {
                eps: 1.0,
                min_samples: 2,
            },
            2,
        );
        let result = clusterer.fit(&points).expect("fit failed");
        assert!(result.noise_ids.contains(&"outlier".to_string()));
    }

    // -----------------------------------------------------------------------
    // 22. DBSCAN: well-separated clusters found
    // -----------------------------------------------------------------------
    #[test]
    fn test_dbscan_finds_clusters() {
        let points = well_separated_2d();
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::DBSCAN {
                eps: 1.0,
                min_samples: 2,
            },
            2,
        );
        let result = clusterer.fit(&points).expect("fit failed");
        assert!(
            result.clusters.len() >= 3,
            "found {} clusters",
            result.clusters.len()
        );
    }

    // -----------------------------------------------------------------------
    // 23. DBSCAN: eps <= 0 → InvalidParameter
    // -----------------------------------------------------------------------
    #[test]
    fn test_dbscan_invalid_eps() {
        let points = pts(&[(0.0, 0.0)]);
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::DBSCAN {
                eps: -0.1,
                min_samples: 2,
            },
            2,
        );
        assert!(matches!(
            clusterer.fit(&points).expect_err("should fail"),
            ClusterError::InvalidParameter(_)
        ));
    }

    // -----------------------------------------------------------------------
    // 24. DBSCAN: min_samples=0 → InvalidParameter
    // -----------------------------------------------------------------------
    #[test]
    fn test_dbscan_invalid_min_samples() {
        let points = pts(&[(0.0, 0.0)]);
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::DBSCAN {
                eps: 1.0,
                min_samples: 0,
            },
            2,
        );
        assert!(matches!(
            clusterer.fit(&points).expect_err("should fail"),
            ClusterError::InvalidParameter(_)
        ));
    }

    // -----------------------------------------------------------------------
    // 25. Agglomerative (Ward): produces k clusters
    // -----------------------------------------------------------------------
    #[test]
    fn test_agglomerative_ward_k_clusters() {
        let points = well_separated_2d();
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::Agglomerative {
                k: 3,
                linkage: Linkage::Ward,
            },
            2,
        );
        let result = clusterer.fit(&points).expect("fit failed");
        assert_eq!(result.clusters.len(), 3);
    }

    // -----------------------------------------------------------------------
    // 26. Agglomerative (Single): all points assigned
    // -----------------------------------------------------------------------
    #[test]
    fn test_agglomerative_single_all_assigned() {
        let points = well_separated_2d();
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::Agglomerative {
                k: 3,
                linkage: Linkage::Single,
            },
            2,
        );
        let result = clusterer.fit(&points).expect("fit failed");
        let total: usize = result.clusters.iter().map(|c| c.member_ids.len()).sum();
        assert_eq!(total, points.len());
    }

    // -----------------------------------------------------------------------
    // 27. Agglomerative (Complete): correct cluster count
    // -----------------------------------------------------------------------
    #[test]
    fn test_agglomerative_complete_count() {
        let points = well_separated_2d();
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::Agglomerative {
                k: 2,
                linkage: Linkage::Complete,
            },
            2,
        );
        let result = clusterer.fit(&points).expect("fit failed");
        assert_eq!(result.clusters.len(), 2);
    }

    // -----------------------------------------------------------------------
    // 28. Agglomerative (Average): correct cluster count
    // -----------------------------------------------------------------------
    #[test]
    fn test_agglomerative_average_count() {
        let points = pts(&[
            (0.0, 0.0),
            (0.1, 0.0),
            (0.0, 0.1),
            (10.0, 0.0),
            (10.1, 0.0),
            (10.0, 0.1),
        ]);
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::Agglomerative {
                k: 2,
                linkage: Linkage::Average,
            },
            2,
        );
        let result = clusterer.fit(&points).expect("fit failed");
        assert_eq!(result.clusters.len(), 2);
    }

    // -----------------------------------------------------------------------
    // 29. predict: assigns to the cluster with the nearest centroid
    // -----------------------------------------------------------------------
    #[test]
    fn test_predict_nearest_cluster() {
        let points = well_separated_2d();
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::KMeans {
                k: 3,
                max_iter: 100,
                tolerance: 1e-6,
            },
            2,
        );
        let result = clusterer.fit(&points).expect("fit failed");

        // Point very close to first cluster's centroid (near [0.05, 0.05])
        let new_point = ScClusterPoint::new("new", vec![0.05, 0.05]);
        let predicted = clusterer.predict(&new_point, &result);
        assert!(predicted.is_some());
        let pid = predicted.expect("predict returned None");
        // The predicted cluster should have members from the 'a' group
        let cluster = result
            .clusters
            .iter()
            .find(|c| c.id == pid)
            .expect("cluster not found");
        assert!(
            cluster.member_ids.iter().any(|id| id.starts_with('a')),
            "predicted cluster should contain 'a' points, got {:?}",
            cluster.member_ids
        );
    }

    // -----------------------------------------------------------------------
    // 30. predict: empty result → None
    // -----------------------------------------------------------------------
    #[test]
    fn test_predict_empty_result() {
        use super::ScClusteringResult;
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::KMeans {
                k: 1,
                max_iter: 1,
                tolerance: 1e-4,
            },
            2,
        );
        let empty_result = ScClusteringResult {
            clusters: vec![],
            noise_ids: vec![],
            algorithm: "test".into(),
            silhouette_score: 0.0,
            inertia: 0.0,
            iterations: 0,
        };
        let point = ScClusterPoint::new("x", vec![0.0, 0.0]);
        assert!(clusterer.predict(&point, &empty_result).is_none());
    }

    // -----------------------------------------------------------------------
    // 31. stats: totals are consistent
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_consistency() {
        let points = well_separated_2d();
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::KMeans {
                k: 3,
                max_iter: 100,
                tolerance: 1e-6,
            },
            2,
        );
        let result = clusterer.fit(&points).expect("fit failed");
        let stats = SemanticClusterer::stats(&result);
        assert_eq!(stats.total_clustered + stats.noise_count, points.len());
    }

    // -----------------------------------------------------------------------
    // 32. stats: avg_cluster_size is correct
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_avg_cluster_size() {
        let points = well_separated_2d(); // 30 points, k=3 → 10 each
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::KMeans {
                k: 3,
                max_iter: 100,
                tolerance: 1e-6,
            },
            2,
        );
        let result = clusterer.fit(&points).expect("fit failed");
        let stats = SemanticClusterer::stats(&result);
        assert!(
            (stats.avg_cluster_size - 10.0).abs() < 1.0,
            "avg={}",
            stats.avg_cluster_size
        );
    }

    // -----------------------------------------------------------------------
    // 33. stats: largest >= smallest
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_largest_ge_smallest() {
        let points = well_separated_2d();
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::KMeans {
                k: 3,
                max_iter: 100,
                tolerance: 1e-6,
            },
            2,
        );
        let result = clusterer.fit(&points).expect("fit failed");
        let stats = SemanticClusterer::stats(&result);
        assert!(stats.largest_cluster >= stats.smallest_cluster);
    }

    // -----------------------------------------------------------------------
    // 34. kmeans_plus_plus_init: produces exactly k centroids
    // -----------------------------------------------------------------------
    #[test]
    fn test_kmeans_pp_init_count() {
        let points = well_separated_2d();
        let centroids = kmeans_plus_plus_init(&points, 3, 42);
        assert_eq!(centroids.len(), 3);
    }

    // -----------------------------------------------------------------------
    // 35. kmeans_plus_plus_init: centroids have correct dimensionality
    // -----------------------------------------------------------------------
    #[test]
    fn test_kmeans_pp_init_dims() {
        let points = well_separated_2d();
        let centroids = kmeans_plus_plus_init(&points, 3, 42);
        for c in &centroids {
            assert_eq!(c.len(), 2);
        }
    }

    // -----------------------------------------------------------------------
    // 36. ScClusterPoint::new: sets cluster_id to None
    // -----------------------------------------------------------------------
    #[test]
    fn test_cluster_point_new_unclustered() {
        let p = ScClusterPoint::new("id", vec![1.0, 2.0]);
        assert!(p.cluster_id.is_none());
    }

    // -----------------------------------------------------------------------
    // 37. ScCluster::is_empty / size
    // -----------------------------------------------------------------------
    #[test]
    fn test_sc_cluster_size_and_empty() {
        use super::ScCluster;
        let empty = ScCluster {
            id: 0,
            centroid: vec![0.0],
            member_ids: vec![],
            inertia: 0.0,
        };
        assert!(empty.is_empty());
        assert_eq!(empty.size(), 0);

        let non_empty = ScCluster {
            id: 1,
            centroid: vec![1.0],
            member_ids: vec!["a".into(), "b".into()],
            inertia: 0.5,
        };
        assert!(!non_empty.is_empty());
        assert_eq!(non_empty.size(), 2);
    }

    // -----------------------------------------------------------------------
    // 38. DBSCAN: single-point cluster (min_samples=1)
    // -----------------------------------------------------------------------
    #[test]
    fn test_dbscan_single_point_cluster() {
        let points = vec![ScClusterPoint::new("only", vec![0.0, 0.0])];
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::DBSCAN {
                eps: 1.0,
                min_samples: 1,
            },
            2,
        );
        let result = clusterer.fit(&points).expect("fit failed");
        assert_eq!(result.clusters.len(), 1);
        assert!(result.noise_ids.is_empty());
    }

    // -----------------------------------------------------------------------
    // 39. tag_points assigns cluster_id correctly
    // -----------------------------------------------------------------------
    #[test]
    fn test_tag_points() {
        let points = pts(&[(0.0, 0.0), (1.0, 1.0)]);
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::KMeans {
                k: 2,
                max_iter: 10,
                tolerance: 1e-4,
            },
            2,
        );
        let result = clusterer.fit(&points).expect("fit failed");
        let tagged = tag_points(&points, &result);
        for tp in &tagged {
            assert!(
                tp.cluster_id.is_some(),
                "point {} should be assigned",
                tp.id
            );
        }
    }

    // -----------------------------------------------------------------------
    // 40. KMeans: algorithm label contains "kmeans"
    // -----------------------------------------------------------------------
    #[test]
    fn test_kmeans_algorithm_label() {
        let points = well_separated_2d();
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::KMeans {
                k: 3,
                max_iter: 50,
                tolerance: 1e-6,
            },
            2,
        );
        let result = clusterer.fit(&points).expect("fit failed");
        assert!(
            result.algorithm.contains("kmeans"),
            "label={}",
            result.algorithm
        );
    }

    // -----------------------------------------------------------------------
    // 41. DBSCAN: algorithm label contains "dbscan"
    // -----------------------------------------------------------------------
    #[test]
    fn test_dbscan_algorithm_label() {
        let points = well_separated_2d();
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::DBSCAN {
                eps: 1.0,
                min_samples: 2,
            },
            2,
        );
        let result = clusterer.fit(&points).expect("fit failed");
        assert!(
            result.algorithm.contains("dbscan"),
            "label={}",
            result.algorithm
        );
    }

    // -----------------------------------------------------------------------
    // 42. Agglomerative k=n: each point in its own cluster
    // -----------------------------------------------------------------------
    #[test]
    fn test_agglomerative_k_equals_n() {
        let points = pts(&[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]);
        let clusterer = SemanticClusterer::new(
            ClusterAlgorithm::Agglomerative {
                k: 3,
                linkage: Linkage::Single,
            },
            2,
        );
        let result = clusterer.fit(&points).expect("fit failed");
        assert_eq!(result.clusters.len(), 3);
        for c in &result.clusters {
            assert_eq!(c.member_ids.len(), 1);
        }
    }
}
