//! Merkle batch inclusion proof generation and verification.
//!
//! Provides [`MerkleBatchProver`] which generates and verifies batch inclusion
//! proofs for multiple leaves in a single Merkle tree traversal, achieving
//! O(n + log n) cost instead of O(n * log n) for individual proofs.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur during Merkle batch operations.
#[derive(Debug, Error, PartialEq)]
pub enum MerkleError {
    /// The tree has no leaves.
    #[error("Merkle tree is empty")]
    EmptyTree,

    /// A requested leaf index exceeds the number of leaves.
    #[error("leaf index {index} is out of bounds for tree of size {tree_size}")]
    LeafIndexOutOfBounds { index: usize, tree_size: usize },

    /// Proof verification failed.
    #[error("invalid proof: {reason}")]
    InvalidProof { reason: String },

    /// A leaf index was supplied more than once in a batch request.
    #[error("duplicate leaf index {index}")]
    DuplicateLeaf { index: usize },
}

// ---------------------------------------------------------------------------
// MerkleNode
// ---------------------------------------------------------------------------

/// A node in a Merkle tree (either a leaf or an internal node).
#[derive(Clone, Debug)]
pub enum MerkleNode {
    /// A leaf node storing the FNV-1a hash of the original data.
    Leaf {
        /// Position of the leaf in the original leaf array.
        index: usize,
        /// FNV-1a hash of the leaf's raw data.
        hash: u64,
    },
    /// An internal node storing the FNV-1a hash of its two children combined.
    Internal {
        /// Hash of the left child.
        left: u64,
        /// Hash of the right child.
        right: u64,
        /// FNV-1a hash over `left.to_le_bytes() || right.to_le_bytes()`.
        hash: u64,
    },
}

impl MerkleNode {
    /// Returns the hash stored in this node.
    pub fn hash(&self) -> u64 {
        match self {
            MerkleNode::Leaf { hash, .. } => *hash,
            MerkleNode::Internal { hash, .. } => *hash,
        }
    }
}

// ---------------------------------------------------------------------------
// BatchProof
// ---------------------------------------------------------------------------

/// A batch inclusion proof for multiple leaves of a Merkle tree.
#[derive(Debug, Clone)]
pub struct BatchProof {
    /// Sorted list of leaf indices covered by this proof.
    pub leaf_indices: Vec<usize>,
    /// Sibling hashes needed for verification, keyed by `(level, hash)`.
    /// `level` 0 is the leaf level; the root is at `level = tree_height`.
    pub sibling_hashes: Vec<(usize, u64)>,
    /// The expected Merkle root.
    pub root: u64,
}

impl BatchProof {
    /// Returns the number of sibling hashes in this proof.
    pub fn proof_size(&self) -> usize {
        self.sibling_hashes.len()
    }
}

// ---------------------------------------------------------------------------
// FNV-1a helpers
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit offset basis and prime.
const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME: u64 = 1_099_511_628_211;

/// Compute the FNV-1a 64-bit hash of a byte slice.
#[inline]
fn fnv1a(data: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// MerkleBatchProver
// ---------------------------------------------------------------------------

/// Generates and verifies batch Merkle inclusion proofs.
///
/// The tree is built over a padded-to-next-power-of-2 leaf array (padding with
/// zero hashes). Leaf hashes use FNV-1a; internal nodes combine children with
/// FNV-1a over their concatenated little-endian representations.
///
/// # Example
///
/// ```rust
/// use ipfrs_core::merkle_batch::MerkleBatchProver;
///
/// let data: &[&[u8]] = &[b"hello", b"world", b"foo", b"bar"];
/// let prover = MerkleBatchProver::new(data).unwrap();
/// let proof = prover.prove_batch(&[0, 2]).unwrap();
/// assert!(prover.verify_batch(&proof, &[b"hello", b"foo"]).unwrap());
/// ```
#[derive(Debug)]
pub struct MerkleBatchProver {
    /// FNV-1a hashes of the original leaf data.
    pub leaves: Vec<u64>,
}

impl MerkleBatchProver {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new prover from raw leaf data.
    ///
    /// # Errors
    ///
    /// Returns [`MerkleError::EmptyTree`] if `data` is empty.
    pub fn new(data: &[&[u8]]) -> Result<Self, MerkleError> {
        if data.is_empty() {
            return Err(MerkleError::EmptyTree);
        }
        let leaves = data.iter().map(|d| Self::leaf_hash(d)).collect();
        Ok(Self { leaves })
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Returns the number of leaves padded to the next power of two.
    pub fn tree_size(&self) -> usize {
        next_power_of_two(self.leaves.len())
    }

    /// Compute the Merkle root.
    ///
    /// The leaf layer is padded to `tree_size()` with zero hashes.
    pub fn root(&self) -> u64 {
        let size = self.tree_size();
        let mut level: Vec<u64> = (0..size)
            .map(|i| {
                if i < self.leaves.len() {
                    self.leaves[i]
                } else {
                    0u64
                }
            })
            .collect();

        while level.len() > 1 {
            level = level
                .chunks(2)
                .map(|pair| Self::combine(pair[0], pair[1]))
                .collect();
        }

        level[0]
    }

    /// Generate a batch inclusion proof for the given leaf indices.
    ///
    /// # Errors
    ///
    /// - [`MerkleError::LeafIndexOutOfBounds`] if any index >= `leaves.len()`.
    /// - [`MerkleError::DuplicateLeaf`] if the same index appears more than once.
    pub fn prove_batch(&self, indices: &[usize]) -> Result<BatchProof, MerkleError> {
        // Validate: bounds and duplicates.
        let n = self.leaves.len();
        let mut sorted = indices.to_vec();
        sorted.sort_unstable();

        for &idx in &sorted {
            if idx >= n {
                return Err(MerkleError::LeafIndexOutOfBounds {
                    index: idx,
                    tree_size: n,
                });
            }
        }
        for window in sorted.windows(2) {
            if window[0] == window[1] {
                return Err(MerkleError::DuplicateLeaf { index: window[0] });
            }
        }

        // Build full padded tree.
        let size = self.tree_size();
        let tree = build_tree(&self.leaves, size);
        let height = tree.len() - 1; // number of levels above leaves

        // Collect sibling hashes using a set to dedup.
        // We track which positions at each level are "already covered" by the
        // batch (i.e., their hash will be recomputed from children), so we only
        // emit siblings for uncovered positions.
        let mut sibling_hashes: Vec<(usize, u64)> = Vec::new();

        // At each level we track the set of node positions that will be
        // recomputed (either they are a target leaf or the parent of two
        // recomputed children).  Siblings of recomputed nodes are needed.
        let mut covered: std::collections::BTreeSet<usize> = sorted.iter().copied().collect();

        for (level, level_nodes) in tree.iter().enumerate().take(height) {
            let mut next_covered: std::collections::BTreeSet<usize> =
                std::collections::BTreeSet::new();
            // For every covered node, its sibling may be needed unless also covered.
            for &pos in &covered {
                let sibling = if pos % 2 == 0 { pos + 1 } else { pos - 1 };
                let parent = pos / 2;
                next_covered.insert(parent);

                // The sibling is needed iff it is NOT itself in `covered`.
                if !covered.contains(&sibling) {
                    // Fetch sibling hash from the tree at the current level.
                    let sib_hash = level_nodes.get(sibling).copied().unwrap_or(0);
                    // Avoid emitting the same (level, hash) pair twice.
                    let entry = (level, sib_hash);
                    if !sibling_hashes.contains(&entry) {
                        sibling_hashes.push(entry);
                    }
                }
            }
            covered = next_covered;
        }

        Ok(BatchProof {
            leaf_indices: sorted,
            sibling_hashes,
            root: self.root(),
        })
    }

    /// Verify a [`BatchProof`] against the provided data items.
    ///
    /// `data_items` must correspond 1-to-1 with `proof.leaf_indices` (same
    /// order and count).
    ///
    /// # Errors
    ///
    /// Returns [`MerkleError::InvalidProof`] if the lengths do not match or if
    /// internal reconstruction fails.
    pub fn verify_batch(
        &self,
        proof: &BatchProof,
        data_items: &[&[u8]],
    ) -> Result<bool, MerkleError> {
        if data_items.len() != proof.leaf_indices.len() {
            return Err(MerkleError::InvalidProof {
                reason: format!(
                    "data_items length {} does not match leaf_indices length {}",
                    data_items.len(),
                    proof.leaf_indices.len()
                ),
            });
        }

        let size = self.tree_size();
        let height = size.trailing_zeros() as usize; // log2(size)

        // Rebuild the partial tree bottom-up using the proof's sibling hashes.
        // We store known hashes as a map: (level, position) -> hash.
        let mut known: std::collections::HashMap<(usize, usize), u64> =
            std::collections::HashMap::new();

        // Insert leaf hashes.
        for (i, &leaf_idx) in proof.leaf_indices.iter().enumerate() {
            let hash = Self::leaf_hash(data_items[i]);
            known.insert((0, leaf_idx), hash);
        }

        // Insert sibling hashes level by level.
        // We need the positions of siblings to insert them correctly.
        // Re-derive sibling positions from the proof's leaf_indices.
        let mut covered: std::collections::BTreeSet<usize> =
            proof.leaf_indices.iter().copied().collect();
        let mut sibling_iter = proof.sibling_hashes.iter();

        for level in 0..height {
            let mut next_covered: std::collections::BTreeSet<usize> =
                std::collections::BTreeSet::new();
            for &pos in &covered {
                let sibling = if pos % 2 == 0 { pos + 1 } else { pos - 1 };
                let parent = pos / 2;
                next_covered.insert(parent);

                if !covered.contains(&sibling) {
                    // Pull next sibling hash from the iterator.
                    match sibling_iter.next() {
                        Some(&(_lv, sib_hash)) => {
                            known.insert((level, sibling), sib_hash);
                        }
                        None => {
                            return Err(MerkleError::InvalidProof {
                                reason: "not enough sibling hashes in proof".to_string(),
                            });
                        }
                    }
                }
            }
            covered = next_covered;
        }

        // Propagate upwards to reconstruct the root.
        // Collect all positions at level 0 that are known.
        let mut current_level_known: std::collections::HashMap<usize, u64> = known
            .iter()
            .filter(|((lvl, _), _)| *lvl == 0)
            .map(|((_, pos), &hash)| (*pos, hash))
            .collect();

        for level in 0..height {
            // Add sibling-provided hashes at this level.
            for ((lvl, pos), &hash) in &known {
                if *lvl == level {
                    current_level_known.insert(*pos, hash);
                }
            }

            let mut next_level: std::collections::HashMap<usize, u64> =
                std::collections::HashMap::new();

            // Find pairs where both or at least one is known (with sibling).
            let mut positions: Vec<usize> = current_level_known.keys().copied().collect();
            positions.sort_unstable();
            positions.dedup();

            // Attempt to compute parents for all known positions.
            let mut processed_parents: std::collections::HashSet<usize> =
                std::collections::HashSet::new();
            for pos in positions {
                let parent = pos / 2;
                if processed_parents.contains(&parent) {
                    continue;
                }
                let left_pos = parent * 2;
                let right_pos = parent * 2 + 1;
                if let (Some(&left_h), Some(&right_h)) = (
                    current_level_known.get(&left_pos),
                    current_level_known.get(&right_pos),
                ) {
                    let parent_hash = Self::combine(left_h, right_h);
                    next_level.insert(parent, parent_hash);
                    processed_parents.insert(parent);
                }
            }

            current_level_known = next_level;
        }

        // The root should be at position 0 of the topmost level.
        match current_level_known.get(&0) {
            Some(&reconstructed_root) => Ok(reconstructed_root == proof.root),
            None => Err(MerkleError::InvalidProof {
                reason: "could not reconstruct root from proof".to_string(),
            }),
        }
    }

    // -----------------------------------------------------------------------
    // Hash primitives
    // -----------------------------------------------------------------------

    /// Compute FNV-1a hash of leaf data.
    pub fn leaf_hash(data: &[u8]) -> u64 {
        fnv1a(data)
    }

    /// Combine two child hashes into a parent hash.
    ///
    /// Uses FNV-1a over the concatenation of the left and right hashes
    /// in little-endian byte order, making it non-commutative.
    pub fn combine(left: u64, right: u64) -> u64 {
        let mut buf = [0u8; 16];
        buf[..8].copy_from_slice(&left.to_le_bytes());
        buf[8..].copy_from_slice(&right.to_le_bytes());
        fnv1a(&buf)
    }
}

// ---------------------------------------------------------------------------
// Internal tree-building helper
// ---------------------------------------------------------------------------

/// Returns the smallest power of two that is >= `n`.  Panics only for `n == 0`
/// (which the public API guards against).
fn next_power_of_two(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

/// Build the complete padded Merkle tree as a vector of levels.
///
/// `tree[0]` is the leaf level (padded to `size`), `tree[height]` is a
/// single-element vec containing the root.
fn build_tree(leaves: &[u64], size: usize) -> Vec<Vec<u64>> {
    let mut level: Vec<u64> = (0..size)
        .map(|i| if i < leaves.len() { leaves[i] } else { 0u64 })
        .collect();

    let mut tree = vec![level.clone()];
    while level.len() > 1 {
        level = level
            .chunks(2)
            .map(|pair| MerkleBatchProver::combine(pair[0], pair[1]))
            .collect();
        tree.push(level.clone());
    }
    tree
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    #[test]
    fn test_new_single_leaf() {
        let prover = MerkleBatchProver::new(&[b"hello"]).unwrap();
        assert_eq!(prover.leaves.len(), 1);
        assert_eq!(prover.leaves[0], MerkleBatchProver::leaf_hash(b"hello"));
    }

    #[test]
    fn test_new_empty_returns_empty_tree() {
        let result = MerkleBatchProver::new(&[]);
        assert_eq!(result.unwrap_err(), MerkleError::EmptyTree);
    }

    // ------------------------------------------------------------------
    // root()
    // ------------------------------------------------------------------

    #[test]
    fn test_root_single_leaf_equals_leaf_hash() {
        let prover = MerkleBatchProver::new(&[b"abc"]).unwrap();
        assert_eq!(prover.root(), MerkleBatchProver::leaf_hash(b"abc"));
    }

    #[test]
    fn test_root_two_leaves_equals_combine() {
        let prover = MerkleBatchProver::new(&[b"left", b"right"]).unwrap();
        let h0 = MerkleBatchProver::leaf_hash(b"left");
        let h1 = MerkleBatchProver::leaf_hash(b"right");
        assert_eq!(prover.root(), MerkleBatchProver::combine(h0, h1));
    }

    #[test]
    fn test_root_power_of_two_tree_correct() {
        // 4-leaf tree: root = combine(combine(h0,h1), combine(h2,h3))
        let data: &[&[u8]] = &[b"a", b"b", b"c", b"d"];
        let prover = MerkleBatchProver::new(data).unwrap();
        let h0 = MerkleBatchProver::leaf_hash(b"a");
        let h1 = MerkleBatchProver::leaf_hash(b"b");
        let h2 = MerkleBatchProver::leaf_hash(b"c");
        let h3 = MerkleBatchProver::leaf_hash(b"d");
        let expected = MerkleBatchProver::combine(
            MerkleBatchProver::combine(h0, h1),
            MerkleBatchProver::combine(h2, h3),
        );
        assert_eq!(prover.root(), expected);
    }

    #[test]
    fn test_root_non_power_of_two_pads_with_zeros() {
        // 3-leaf tree pads to 4; the 4th leaf hash is 0.
        let data: &[&[u8]] = &[b"x", b"y", b"z"];
        let prover = MerkleBatchProver::new(data).unwrap();
        let h0 = MerkleBatchProver::leaf_hash(b"x");
        let h1 = MerkleBatchProver::leaf_hash(b"y");
        let h2 = MerkleBatchProver::leaf_hash(b"z");
        let h3 = 0u64; // padding
        let expected = MerkleBatchProver::combine(
            MerkleBatchProver::combine(h0, h1),
            MerkleBatchProver::combine(h2, h3),
        );
        assert_eq!(prover.root(), expected);
    }

    // ------------------------------------------------------------------
    // prove_batch() error cases
    // ------------------------------------------------------------------

    #[test]
    fn test_prove_batch_out_of_bounds_returns_error() {
        let prover = MerkleBatchProver::new(&[b"a", b"b"]).unwrap();
        let result = prover.prove_batch(&[5]);
        assert!(matches!(
            result.unwrap_err(),
            MerkleError::LeafIndexOutOfBounds {
                index: 5,
                tree_size: 2
            }
        ));
    }

    #[test]
    fn test_prove_batch_duplicate_index_returns_error() {
        let prover = MerkleBatchProver::new(&[b"a", b"b", b"c"]).unwrap();
        let result = prover.prove_batch(&[1, 1]);
        assert!(matches!(
            result.unwrap_err(),
            MerkleError::DuplicateLeaf { index: 1 }
        ));
    }

    // ------------------------------------------------------------------
    // prove_batch() success cases
    // ------------------------------------------------------------------

    #[test]
    fn test_prove_batch_single_leaf_produces_proof() {
        let prover = MerkleBatchProver::new(&[b"a", b"b", b"c", b"d"]).unwrap();
        let proof = prover.prove_batch(&[0]).unwrap();
        assert_eq!(proof.leaf_indices, vec![0]);
        // For a single leaf in a 4-leaf tree we need log2(4) = 2 siblings.
        assert_eq!(proof.proof_size(), 2);
    }

    #[test]
    fn test_prove_batch_two_leaves_smaller_than_two_individual() {
        let data: &[&[u8]] = &[b"a", b"b", b"c", b"d"];
        let prover = MerkleBatchProver::new(data).unwrap();
        // Adjacent leaves 0 and 1 share a parent; only 1 sibling needed at level 1.
        let batch_proof = prover.prove_batch(&[0, 1]).unwrap();
        let proof_0 = prover.prove_batch(&[0]).unwrap();
        let proof_1 = prover.prove_batch(&[1]).unwrap();
        assert!(batch_proof.proof_size() < proof_0.proof_size() + proof_1.proof_size());
    }

    // ------------------------------------------------------------------
    // verify_batch()
    // ------------------------------------------------------------------

    #[test]
    fn test_verify_batch_valid_proof_returns_true() {
        let data: &[&[u8]] = &[b"hello", b"world", b"foo", b"bar"];
        let prover = MerkleBatchProver::new(data).unwrap();
        let proof = prover.prove_batch(&[1, 3]).unwrap();
        let result = prover.verify_batch(&proof, &[b"world", b"bar"]).unwrap();
        assert!(result);
    }

    #[test]
    fn test_verify_batch_tampered_data_returns_false() {
        let data: &[&[u8]] = &[b"hello", b"world", b"foo", b"bar"];
        let prover = MerkleBatchProver::new(data).unwrap();
        let proof = prover.prove_batch(&[0, 2]).unwrap();
        // Pass tampered data for index 0.
        let result = prover.verify_batch(&proof, &[b"tampered", b"foo"]).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_verify_batch_indices_0_and_1_in_4_leaf_tree() {
        let data: &[&[u8]] = &[b"a", b"b", b"c", b"d"];
        let prover = MerkleBatchProver::new(data).unwrap();
        let proof = prover.prove_batch(&[0, 1]).unwrap();
        let result = prover.verify_batch(&proof, &[b"a", b"b"]).unwrap();
        assert!(result);
    }

    // ------------------------------------------------------------------
    // Primitives
    // ------------------------------------------------------------------

    #[test]
    fn test_leaf_hash_deterministic() {
        let h1 = MerkleBatchProver::leaf_hash(b"deterministic");
        let h2 = MerkleBatchProver::leaf_hash(b"deterministic");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_combine_is_non_commutative() {
        let a = MerkleBatchProver::leaf_hash(b"left");
        let b = MerkleBatchProver::leaf_hash(b"right");
        assert_ne!(a, b); // sanity: different hashes
        assert_ne!(
            MerkleBatchProver::combine(a, b),
            MerkleBatchProver::combine(b, a)
        );
    }

    #[test]
    fn test_tree_size_is_next_power_of_two() {
        let cases: &[(&[&[u8]], usize)] = &[
            (&[b"a"], 1),
            (&[b"a", b"b"], 2),
            (&[b"a", b"b", b"c"], 4),
            (&[b"a", b"b", b"c", b"d"], 4),
            (&[b"a", b"b", b"c", b"d", b"e"], 8),
        ];
        for (data, expected) in cases {
            let prover = MerkleBatchProver::new(data).unwrap();
            assert_eq!(prover.tree_size(), *expected, "data.len()={}", data.len());
        }
    }

    // ------------------------------------------------------------------
    // Additional edge-case / regression tests
    // ------------------------------------------------------------------

    #[test]
    fn test_verify_all_leaves_single_leaf_tree() {
        let prover = MerkleBatchProver::new(&[b"solo"]).unwrap();
        let proof = prover.prove_batch(&[0]).unwrap();
        assert!(prover.verify_batch(&proof, &[b"solo"]).unwrap());
    }

    #[test]
    fn test_root_eight_leaf_tree() {
        let data: &[&[u8]] = &[b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"8"];
        let prover = MerkleBatchProver::new(data).unwrap();
        // Verify round-trip: prove all leaves, verify.
        let proof = prover.prove_batch(&[0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
        assert!(prover
            .verify_batch(&proof, &[b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"8"])
            .unwrap());
    }
}
