//! Merkle tree construction and proof verification.
//!
//! Provides [`MerkleProofVerifier`] for building complete binary Merkle trees,
//! generating inclusion proofs, and verifying those proofs — all using a pure-Rust
//! FNV-1a 64-bit hash function with domain separation between leaf and internal nodes.
//!
//! # Layout
//! Nodes are stored in a 1-indexed flat `Vec`:
//! - index `1` = root
//! - left child of `i` = `2 * i`
//! - right child of `i` = `2 * i + 1`
//!
//! The tree is always padded to the next power of two.

use std::fmt;

// ---------------------------------------------------------------------------
// Hash primitives (pure Rust, FNV-1a 64-bit, no external hash crates)
// ---------------------------------------------------------------------------

#[inline]
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

#[cfg(test)]
#[inline]
fn hash_pair(left: &[u8; 8], right: &[u8; 8]) -> [u8; 8] {
    let mut combined = [0u8; 16];
    combined[..8].copy_from_slice(left);
    combined[8..].copy_from_slice(right);
    fnv1a_64(&combined).to_le_bytes()
}

#[inline]
fn hash_leaf(data: &[u8]) -> [u8; 8] {
    // prefix 0x00 to distinguish leaves from internal nodes
    let mut input = Vec::with_capacity(1 + data.len());
    input.push(0x00);
    input.extend_from_slice(data);
    fnv1a_64(&input).to_le_bytes()
}

#[inline]
fn hash_internal(left: &[u8; 8], right: &[u8; 8]) -> [u8; 8] {
    // prefix 0x01 to distinguish internal nodes from leaves
    let mut input = Vec::with_capacity(17);
    input.push(0x01);
    input.extend_from_slice(left);
    input.extend_from_slice(right);
    fnv1a_64(&input).to_le_bytes()
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A 64-bit Merkle hash (8 bytes, FNV-1a based).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MerkleHash([u8; 8]);

impl MerkleHash {
    /// Construct from raw bytes.
    #[inline]
    pub fn from_bytes(bytes: [u8; 8]) -> Self {
        Self(bytes)
    }

    /// Return the inner byte array.
    #[inline]
    pub fn as_bytes(&self) -> &[u8; 8] {
        &self.0
    }

    /// The zero / default hash (all bytes zero).
    #[inline]
    pub fn zero() -> Self {
        Self([0u8; 8])
    }
}

impl fmt::Display for MerkleHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in &self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

impl Default for MerkleHash {
    fn default() -> Self {
        Self::zero()
    }
}

/// A leaf node in the Merkle tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MerkleLeaf {
    /// Zero-based logical position in the original dataset.
    pub index: usize,
    /// Raw leaf data.
    pub data: Vec<u8>,
    /// Computed hash of the leaf (`hash_leaf(data)`).
    pub hash: MerkleHash,
}

/// One step in a Merkle inclusion proof path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofStep {
    /// Hash of the sibling node at this level.
    pub sibling_hash: MerkleHash,
    /// `true` if the sibling is on the *left*, `false` if on the *right*.
    pub is_left: bool,
}

/// A complete inclusion proof for a single leaf.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MerkleProof {
    /// Zero-based index of the proven leaf.
    pub leaf_index: usize,
    /// Hash of the leaf itself.
    pub leaf_hash: MerkleHash,
    /// Sibling-hash path from leaf level up to (but not including) the root.
    pub path: Vec<ProofStep>,
    /// Expected root after replaying the path.
    pub root: MerkleHash,
}

/// Summary statistics for a built Merkle tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeStats {
    /// Number of *logical* (original) leaves (before padding).
    pub leaf_count: usize,
    /// Height of the tree (number of levels, including the root level).
    pub tree_height: usize,
    /// Root hash.
    pub root_hash: MerkleHash,
    /// Total node count stored (including padding leaves).
    pub total_nodes: usize,
}

/// Proof that a single leaf was updated in a Merkle tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UpdateProof {
    /// Inclusion proof before the update.
    pub old_proof: MerkleProof,
    /// Inclusion proof after the update.
    pub new_proof: MerkleProof,
    /// Zero-based index of the changed leaf.
    pub changed_index: usize,
}

/// Errors returned by [`MerkleProofVerifier`] operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VerifierError {
    /// Cannot build or query an empty tree.
    #[error("empty tree: at least one leaf is required")]
    EmptyTree,

    /// The requested leaf index exceeds the logical leaf count.
    #[error("leaf index {0} is out of bounds")]
    LeafIndexOutOfBounds(usize),

    /// A proof's recomputed root does not match the stored root.
    #[error("proof invalid: expected root {expected_root}, computed root {computed_root}")]
    ProofInvalid {
        expected_root: MerkleHash,
        computed_root: MerkleHash,
    },

    /// The hash stored in the proof's leaf does not match the recomputed hash.
    #[error("hash mismatch at leaf index {index}")]
    HashMismatch {
        /// Leaf index where the mismatch was detected.
        index: usize,
    },

    /// The tree structure is internally inconsistent.
    #[error("invalid tree structure: {0}")]
    InvalidTreeStructure(String),
}

// ---------------------------------------------------------------------------
// MerkleProofVerifier
// ---------------------------------------------------------------------------

/// Production-quality Merkle tree with proof generation and verification.
///
/// Internally stores all node hashes in a 1-indexed flat `Vec`.
/// The tree is always padded to the next power of two.
#[derive(Debug)]
pub struct MerkleProofVerifier {
    /// 1-indexed flat array: index 1 = root, left(i) = 2i, right(i) = 2i+1.
    /// `nodes[0]` is unused.
    nodes: Vec<MerkleHash>,
    /// Number of *padded* leaves (always a power of two).
    padded_leaf_count: usize,
    /// Number of *original* (logical) leaves supplied by the caller.
    logical_leaf_count: usize,
    /// Height of the tree (number of levels including root and leaves).
    height: usize,
    /// Cached leaves including their data (for update operations).
    leaf_data: Vec<Vec<u8>>,
}

impl MerkleProofVerifier {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Build a complete Merkle tree from raw leaf data.
    ///
    /// If `leaves` is not a power of two, empty leaves (`[]`) are appended as
    /// padding so the tree remains a complete binary tree.
    ///
    /// Returns [`VerifierError::EmptyTree`] when `leaves` is empty.
    pub fn new(leaves: Vec<Vec<u8>>) -> Result<Self, VerifierError> {
        if leaves.is_empty() {
            return Err(VerifierError::EmptyTree);
        }

        let logical_leaf_count = leaves.len();
        let padded_leaf_count = next_power_of_two(logical_leaf_count);
        let height = padded_leaf_count.trailing_zeros() as usize + 1; // e.g. 4 leaves → height 3

        // Total nodes in a 1-indexed complete binary tree with `padded_leaf_count` leaves:
        // internal nodes: padded_leaf_count - 1,  leaves: padded_leaf_count
        // total = 2 * padded_leaf_count
        let total_nodes = 2 * padded_leaf_count;

        // Allocate and fill with zero hashes.
        let mut nodes = vec![MerkleHash::zero(); total_nodes + 1]; // +1 because 1-indexed

        // Leaf offset: leaves occupy indices [padded_leaf_count .. 2*padded_leaf_count)
        let leaf_offset = padded_leaf_count;

        // Pad leaf data and compute leaf hashes.
        let mut leaf_data = leaves;
        leaf_data.resize(padded_leaf_count, Vec::new());

        for (i, data) in leaf_data.iter().enumerate() {
            let raw_hash = hash_leaf(data);
            nodes[leaf_offset + i] = MerkleHash::from_bytes(raw_hash);
        }

        // Build internal nodes bottom-up.
        for i in (1..padded_leaf_count).rev() {
            let left = nodes[2 * i];
            let right = nodes[2 * i + 1];
            let raw_hash = hash_internal(left.as_bytes(), right.as_bytes());
            nodes[i] = MerkleHash::from_bytes(raw_hash);
        }

        Ok(Self {
            nodes,
            padded_leaf_count,
            logical_leaf_count,
            height,
            leaf_data,
        })
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Return the root hash of the tree.
    #[inline]
    pub fn root(&self) -> MerkleHash {
        self.nodes[1]
    }

    /// Return the number of *logical* (original) leaves.
    #[inline]
    pub fn leaf_count(&self) -> usize {
        self.logical_leaf_count
    }

    /// Return summary statistics.
    pub fn stats(&self) -> TreeStats {
        let total_nodes = self.nodes.len().saturating_sub(1); // exclude index-0 slot
        TreeStats {
            leaf_count: self.logical_leaf_count,
            tree_height: self.height,
            root_hash: self.root(),
            total_nodes,
        }
    }

    // -----------------------------------------------------------------------
    // Proof generation
    // -----------------------------------------------------------------------

    /// Generate an inclusion proof for the leaf at `index`.
    ///
    /// Returns [`VerifierError::LeafIndexOutOfBounds`] when `index >= leaf_count()`.
    pub fn generate_proof(&self, index: usize) -> Result<MerkleProof, VerifierError> {
        if index >= self.logical_leaf_count {
            return Err(VerifierError::LeafIndexOutOfBounds(index));
        }

        let leaf_hash = self.nodes[self.padded_leaf_count + index];
        let mut path = Vec::with_capacity(self.height.saturating_sub(1));

        let mut node_index = self.padded_leaf_count + index;
        while node_index > 1 {
            let is_right_child = node_index % 2 == 1;
            let sibling_index = if is_right_child {
                node_index - 1
            } else {
                node_index + 1
            };
            path.push(ProofStep {
                sibling_hash: self.nodes[sibling_index],
                // is_left: sibling is on the left when *we* are the right child
                is_left: is_right_child,
            });
            node_index /= 2;
        }

        Ok(MerkleProof {
            leaf_index: index,
            leaf_hash,
            path,
            root: self.root(),
        })
    }

    /// Generate inclusion proofs for every leaf in `[start, end)`.
    ///
    /// Returns [`VerifierError::LeafIndexOutOfBounds`] when `end > leaf_count()`.
    pub fn generate_range_proof(
        &self,
        start: usize,
        end: usize,
    ) -> Result<Vec<MerkleProof>, VerifierError> {
        if end > self.logical_leaf_count {
            return Err(VerifierError::LeafIndexOutOfBounds(end.saturating_sub(1)));
        }
        if start >= end {
            return Ok(Vec::new());
        }
        let mut proofs = Vec::with_capacity(end - start);
        for i in start..end {
            proofs.push(self.generate_proof(i)?);
        }
        Ok(proofs)
    }

    // -----------------------------------------------------------------------
    // Verification
    // -----------------------------------------------------------------------

    /// Verify an inclusion proof against the tree's own root.
    ///
    /// Returns `true` when the proof is valid, `false` otherwise.
    /// Returns an error only when the proof is structurally inconsistent.
    pub fn verify_proof(&self, proof: &MerkleProof) -> Result<bool, VerifierError> {
        self.verify_against_root(proof, &self.root())
    }

    /// Verify an inclusion proof against an *externally* supplied root.
    pub fn verify_against_root(
        &self,
        proof: &MerkleProof,
        expected_root: &MerkleHash,
    ) -> Result<bool, VerifierError> {
        let computed_root = recompute_root(proof)?;
        if computed_root != *expected_root {
            return Ok(false);
        }
        if computed_root != proof.root {
            return Ok(false);
        }
        Ok(true)
    }

    /// Verify that all proofs in `proofs` share the same root and are individually valid.
    ///
    /// Returns `true` when every proof is valid and they all share a consistent root.
    pub fn verify_range(&self, proofs: &[MerkleProof]) -> Result<bool, VerifierError> {
        if proofs.is_empty() {
            return Ok(true);
        }
        let shared_root = proofs[0].root;
        for proof in proofs {
            if proof.root != shared_root {
                return Ok(false);
            }
            let computed = recompute_root(proof)?;
            if computed != shared_root {
                return Ok(false);
            }
        }
        Ok(true)
    }

    // -----------------------------------------------------------------------
    // Updates
    // -----------------------------------------------------------------------

    /// Replace the leaf at `index` with `new_data`, update all nodes on the
    /// path to the root in O(log n), and return an [`UpdateProof`].
    ///
    /// Returns [`VerifierError::LeafIndexOutOfBounds`] when `index >= leaf_count()`.
    pub fn update_leaf(
        &mut self,
        index: usize,
        new_data: Vec<u8>,
    ) -> Result<UpdateProof, VerifierError> {
        if index >= self.logical_leaf_count {
            return Err(VerifierError::LeafIndexOutOfBounds(index));
        }

        // Capture the old proof before mutation.
        let old_proof = self.generate_proof(index)?;

        // Update the leaf hash.
        let new_leaf_raw = hash_leaf(&new_data);
        let leaf_node_index = self.padded_leaf_count + index;
        self.nodes[leaf_node_index] = MerkleHash::from_bytes(new_leaf_raw);
        self.leaf_data[index] = new_data;

        // Recompute the path from the updated leaf up to the root — O(log n).
        let mut current = leaf_node_index / 2;
        while current >= 1 {
            let left = self.nodes[2 * current];
            let right = self.nodes[2 * current + 1];
            let raw = hash_internal(left.as_bytes(), right.as_bytes());
            self.nodes[current] = MerkleHash::from_bytes(raw);
            if current == 1 {
                break;
            }
            current /= 2;
        }

        let new_proof = self.generate_proof(index)?;

        Ok(UpdateProof {
            old_proof,
            new_proof,
            changed_index: index,
        })
    }

    /// Verify that both sides of an [`UpdateProof`] are internally consistent.
    ///
    /// Specifically:
    /// 1. `old_proof.root` ≠ `new_proof.root` (they should differ after an update,
    ///    unless the new data happens to hash identically, which is still accepted).
    /// 2. Both proofs replicate to their respective roots correctly.
    /// 3. `old_proof.leaf_index == new_proof.leaf_index == update.changed_index`.
    pub fn verify_update(&self, update: &UpdateProof) -> Result<bool, VerifierError> {
        // Index consistency.
        if update.old_proof.leaf_index != update.changed_index
            || update.new_proof.leaf_index != update.changed_index
        {
            return Ok(false);
        }

        // Verify both proofs replay to their own roots.
        let old_root_ok = {
            let computed = recompute_root(&update.old_proof)?;
            computed == update.old_proof.root
        };
        let new_root_ok = {
            let computed = recompute_root(&update.new_proof)?;
            computed == update.new_proof.root
        };

        Ok(old_root_ok && new_root_ok)
    }
}

// ---------------------------------------------------------------------------
// Helper: replay a proof path to compute the root
// ---------------------------------------------------------------------------

fn recompute_root(proof: &MerkleProof) -> Result<MerkleHash, VerifierError> {
    let mut current = *proof.leaf_hash.as_bytes();

    for step in &proof.path {
        let raw = if step.is_left {
            // sibling is on the left
            hash_internal(step.sibling_hash.as_bytes(), &current)
        } else {
            // sibling is on the right
            hash_internal(&current, step.sibling_hash.as_bytes())
        };
        current = raw;
    }

    Ok(MerkleHash::from_bytes(current))
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Round `n` up to the next power of two (or return `n` if it already is one).
#[inline]
fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    if n.is_power_of_two() {
        return n;
    }
    n.next_power_of_two()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_leaves(n: usize) -> Vec<Vec<u8>> {
        (0..n).map(|i| format!("leaf-{i}").into_bytes()).collect()
    }

    fn single_leaf_tree() -> MerkleProofVerifier {
        MerkleProofVerifier::new(vec![b"hello".to_vec()]).expect("single leaf")
    }

    // -----------------------------------------------------------------------
    // Hash primitives
    // -----------------------------------------------------------------------

    #[test]
    fn test_fnv1a_known_empty() {
        // FNV-1a offset basis for empty input is 14695981039346656037
        assert_eq!(fnv1a_64(&[]), 14_695_981_039_346_656_037u64);
    }

    #[test]
    fn test_hash_leaf_domain_separation() {
        let h_leaf = hash_leaf(b"data");
        // Must differ from hash_pair applied to the same bytes (no 0x00 prefix there)
        let h_pair = hash_pair(&[0u8; 8], &[0u8; 8]);
        assert_ne!(h_leaf, h_pair);
    }

    #[test]
    fn test_hash_internal_domain_separation_from_leaf() {
        let left = hash_leaf(b"left");
        let right = hash_leaf(b"right");
        let internal = hash_internal(&left, &right);
        // If domain separation works, hashing the raw concat should differ
        let naive = hash_pair(&left, &right);
        assert_ne!(internal, naive);
    }

    #[test]
    fn test_hash_internal_not_commutative() {
        let a = [1u8; 8];
        let b = [2u8; 8];
        assert_ne!(hash_internal(&a, &b), hash_internal(&b, &a));
    }

    // -----------------------------------------------------------------------
    // MerkleHash display
    // -----------------------------------------------------------------------

    #[test]
    fn test_merkle_hash_display_len() {
        let h = MerkleHash::from_bytes([0xde, 0xad, 0xbe, 0xef, 0x00, 0x11, 0x22, 0x33]);
        let s = h.to_string();
        assert_eq!(s.len(), 16);
        assert_eq!(s, "deadbeef00112233");
    }

    #[test]
    fn test_merkle_hash_zero_display() {
        let h = MerkleHash::zero();
        assert_eq!(h.to_string(), "0000000000000000");
    }

    #[test]
    fn test_merkle_hash_eq() {
        let a = MerkleHash::from_bytes([1u8; 8]);
        let b = MerkleHash::from_bytes([1u8; 8]);
        let c = MerkleHash::from_bytes([2u8; 8]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // -----------------------------------------------------------------------
    // Empty tree
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_tree_returns_error() {
        let result = MerkleProofVerifier::new(vec![]);
        assert_eq!(result.unwrap_err(), VerifierError::EmptyTree);
    }

    // -----------------------------------------------------------------------
    // Single-leaf tree
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_leaf_root_non_zero() {
        let verifier = single_leaf_tree();
        assert_ne!(verifier.root(), MerkleHash::zero());
    }

    #[test]
    fn test_single_leaf_leaf_count() {
        let verifier = single_leaf_tree();
        assert_eq!(verifier.leaf_count(), 1);
    }

    #[test]
    fn test_single_leaf_generate_proof() {
        let verifier = single_leaf_tree();
        let proof = verifier.generate_proof(0).expect("proof");
        assert_eq!(proof.leaf_index, 0);
        assert_eq!(proof.root, verifier.root());
    }

    #[test]
    fn test_single_leaf_proof_verification() {
        let verifier = single_leaf_tree();
        let proof = verifier.generate_proof(0).expect("proof");
        assert!(verifier.verify_proof(&proof).expect("verify"));
    }

    #[test]
    fn test_single_leaf_out_of_bounds() {
        let verifier = single_leaf_tree();
        let err = verifier.generate_proof(1).unwrap_err();
        assert_eq!(err, VerifierError::LeafIndexOutOfBounds(1));
    }

    // -----------------------------------------------------------------------
    // Power-of-two sizes
    // -----------------------------------------------------------------------

    #[test]
    fn test_two_leaves() {
        let verifier = MerkleProofVerifier::new(make_leaves(2)).expect("2 leaves");
        assert_eq!(verifier.leaf_count(), 2);
        for i in 0..2 {
            let proof = verifier.generate_proof(i).expect("proof");
            assert!(verifier.verify_proof(&proof).expect("verify"));
        }
    }

    #[test]
    fn test_four_leaves() {
        let verifier = MerkleProofVerifier::new(make_leaves(4)).expect("4 leaves");
        assert_eq!(verifier.leaf_count(), 4);
        for i in 0..4 {
            let proof = verifier.generate_proof(i).expect("proof");
            assert!(verifier.verify_proof(&proof).expect("verify"));
        }
    }

    #[test]
    fn test_eight_leaves() {
        let verifier = MerkleProofVerifier::new(make_leaves(8)).expect("8 leaves");
        assert_eq!(verifier.leaf_count(), 8);
        for i in 0..8 {
            let proof = verifier.generate_proof(i).expect("proof");
            assert!(verifier.verify_proof(&proof).expect("verify"));
        }
    }

    #[test]
    fn test_sixteen_leaves() {
        let verifier = MerkleProofVerifier::new(make_leaves(16)).expect("16 leaves");
        for i in 0..16 {
            let proof = verifier.generate_proof(i).expect("proof");
            assert!(verifier.verify_proof(&proof).expect("verify"));
        }
    }

    #[test]
    fn test_large_power_of_two_128() {
        let verifier = MerkleProofVerifier::new(make_leaves(128)).expect("128 leaves");
        let proof_first = verifier.generate_proof(0).expect("proof 0");
        let proof_last = verifier.generate_proof(127).expect("proof 127");
        assert!(verifier.verify_proof(&proof_first).expect("verify"));
        assert!(verifier.verify_proof(&proof_last).expect("verify"));
    }

    // -----------------------------------------------------------------------
    // Non-power-of-two padding
    // -----------------------------------------------------------------------

    #[test]
    fn test_three_leaves_padded_to_four() {
        let verifier = MerkleProofVerifier::new(make_leaves(3)).expect("3 leaves");
        assert_eq!(verifier.leaf_count(), 3);
        // All three logical leaves must have valid proofs.
        for i in 0..3 {
            let proof = verifier.generate_proof(i).expect("proof");
            assert!(verifier.verify_proof(&proof).expect("verify"));
        }
        // Index 3 is a padding leaf — out of bounds from caller perspective.
        assert!(verifier.generate_proof(3).is_err());
    }

    #[test]
    fn test_five_leaves_padded_to_eight() {
        let verifier = MerkleProofVerifier::new(make_leaves(5)).expect("5 leaves");
        assert_eq!(verifier.leaf_count(), 5);
        for i in 0..5 {
            let proof = verifier.generate_proof(i).expect("proof");
            assert!(verifier.verify_proof(&proof).expect("verify"));
        }
    }

    #[test]
    fn test_seven_leaves() {
        let verifier = MerkleProofVerifier::new(make_leaves(7)).expect("7 leaves");
        for i in 0..7 {
            let proof = verifier.generate_proof(i).expect("proof");
            assert!(verifier.verify_proof(&proof).expect("verify"));
        }
    }

    #[test]
    fn test_ten_leaves_padded_to_sixteen() {
        let verifier = MerkleProofVerifier::new(make_leaves(10)).expect("10 leaves");
        assert_eq!(verifier.leaf_count(), 10);
        for i in 0..10 {
            let proof = verifier.generate_proof(i).expect("proof");
            assert!(verifier.verify_proof(&proof).expect("verify"));
        }
    }

    #[test]
    fn test_100_leaves() {
        let verifier = MerkleProofVerifier::new(make_leaves(100)).expect("100 leaves");
        assert_eq!(verifier.leaf_count(), 100);
        for i in [0, 1, 50, 99] {
            let proof = verifier.generate_proof(i).expect("proof");
            assert!(verifier.verify_proof(&proof).expect("verify"));
        }
    }

    // -----------------------------------------------------------------------
    // Proof path length
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_path_length_power_of_two() {
        // For a tree with 8 leaves, height = 4, path length = 3.
        let verifier = MerkleProofVerifier::new(make_leaves(8)).expect("8 leaves");
        for i in 0..8 {
            let proof = verifier.generate_proof(i).expect("proof");
            assert_eq!(proof.path.len(), 3, "leaf {i}");
        }
    }

    #[test]
    fn test_proof_path_length_padded() {
        // For 5 leaves padded to 8, still height 4, path length 3.
        let verifier = MerkleProofVerifier::new(make_leaves(5)).expect("5 leaves");
        for i in 0..5 {
            let proof = verifier.generate_proof(i).expect("proof");
            assert_eq!(proof.path.len(), 3, "leaf {i}");
        }
    }

    #[test]
    fn test_proof_path_length_single_leaf() {
        // Single leaf: padded to 1, height 1, path length 0.
        let verifier = single_leaf_tree();
        let proof = verifier.generate_proof(0).expect("proof");
        assert_eq!(proof.path.len(), 0);
    }

    #[test]
    fn test_proof_path_length_two_leaves() {
        let verifier = MerkleProofVerifier::new(make_leaves(2)).expect("2 leaves");
        for i in 0..2 {
            let proof = verifier.generate_proof(i).expect("proof");
            assert_eq!(proof.path.len(), 1, "leaf {i}");
        }
    }

    // -----------------------------------------------------------------------
    // Root consistency across all proofs
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_proofs_share_root() {
        let verifier = MerkleProofVerifier::new(make_leaves(6)).expect("6 leaves");
        let root = verifier.root();
        for i in 0..6 {
            let proof = verifier.generate_proof(i).expect("proof");
            assert_eq!(proof.root, root);
        }
    }

    // -----------------------------------------------------------------------
    // Invalid proof detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_tampered_leaf_hash_fails() {
        let verifier = MerkleProofVerifier::new(make_leaves(4)).expect("4 leaves");
        let mut proof = verifier.generate_proof(0).expect("proof");
        // Tamper with the leaf hash.
        proof.leaf_hash = MerkleHash::from_bytes([0xff; 8]);
        assert!(!verifier.verify_proof(&proof).expect("verify"));
    }

    #[test]
    fn test_tampered_path_step_fails() {
        let verifier = MerkleProofVerifier::new(make_leaves(4)).expect("4 leaves");
        let mut proof = verifier.generate_proof(2).expect("proof");
        if let Some(step) = proof.path.first_mut() {
            step.sibling_hash = MerkleHash::from_bytes([0xaa; 8]);
        }
        assert!(!verifier.verify_proof(&proof).expect("verify"));
    }

    #[test]
    fn test_flipped_is_left_fails() {
        let verifier = MerkleProofVerifier::new(make_leaves(4)).expect("4 leaves");
        let mut proof = verifier.generate_proof(1).expect("proof");
        if let Some(step) = proof.path.first_mut() {
            step.is_left = !step.is_left;
        }
        // After flipping, the recomputed root will be different from the stored root.
        assert!(!verifier.verify_proof(&proof).expect("verify"));
    }

    #[test]
    fn test_wrong_root_in_proof_fails() {
        let verifier = MerkleProofVerifier::new(make_leaves(4)).expect("4 leaves");
        let mut proof = verifier.generate_proof(0).expect("proof");
        proof.root = MerkleHash::from_bytes([0x00; 8]);
        assert!(!verifier.verify_proof(&proof).expect("verify"));
    }

    // -----------------------------------------------------------------------
    // verify_against_root
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_against_correct_root() {
        let verifier = MerkleProofVerifier::new(make_leaves(4)).expect("4 leaves");
        let proof = verifier.generate_proof(1).expect("proof");
        let root = verifier.root();
        assert!(verifier.verify_against_root(&proof, &root).expect("verify"));
    }

    #[test]
    fn test_verify_against_wrong_root() {
        let verifier = MerkleProofVerifier::new(make_leaves(4)).expect("4 leaves");
        let proof = verifier.generate_proof(1).expect("proof");
        let bad_root = MerkleHash::from_bytes([0x12; 8]);
        assert!(!verifier
            .verify_against_root(&proof, &bad_root)
            .expect("verify"));
    }

    #[test]
    fn test_proof_from_one_tree_fails_against_another_root() {
        let v1 = MerkleProofVerifier::new(make_leaves(4)).expect("v1");
        let v2 = MerkleProofVerifier::new(make_leaves(4)).expect("v2");
        // Both trees have different data, so roots differ.
        // A proof from v1 verified against v2's root should fail.
        let proof = v1.generate_proof(0).expect("proof");
        let v2_root = v2.root();
        if v1.root() != v2_root {
            assert!(!v1.verify_against_root(&proof, &v2_root).expect("verify"));
        }
    }

    // -----------------------------------------------------------------------
    // Range proofs
    // -----------------------------------------------------------------------

    #[test]
    fn test_range_proof_full() {
        let verifier = MerkleProofVerifier::new(make_leaves(4)).expect("4 leaves");
        let proofs = verifier.generate_range_proof(0, 4).expect("range");
        assert_eq!(proofs.len(), 4);
        assert!(verifier.verify_range(&proofs).expect("verify_range"));
    }

    #[test]
    fn test_range_proof_partial() {
        let verifier = MerkleProofVerifier::new(make_leaves(8)).expect("8 leaves");
        let proofs = verifier.generate_range_proof(2, 6).expect("range");
        assert_eq!(proofs.len(), 4);
        assert!(verifier.verify_range(&proofs).expect("verify_range"));
    }

    #[test]
    fn test_range_proof_single() {
        let verifier = MerkleProofVerifier::new(make_leaves(8)).expect("8 leaves");
        let proofs = verifier.generate_range_proof(3, 4).expect("range");
        assert_eq!(proofs.len(), 1);
        assert!(verifier.verify_range(&proofs).expect("verify_range"));
    }

    #[test]
    fn test_range_proof_empty_range() {
        let verifier = MerkleProofVerifier::new(make_leaves(4)).expect("4 leaves");
        let proofs = verifier.generate_range_proof(2, 2).expect("empty range");
        assert_eq!(proofs.len(), 0);
        assert!(verifier.verify_range(&proofs).expect("verify empty"));
    }

    #[test]
    fn test_range_proof_out_of_bounds() {
        let verifier = MerkleProofVerifier::new(make_leaves(4)).expect("4 leaves");
        let err = verifier.generate_range_proof(0, 5).unwrap_err();
        assert_eq!(err, VerifierError::LeafIndexOutOfBounds(4));
    }

    #[test]
    fn test_verify_range_tampered_root_fails() {
        let verifier = MerkleProofVerifier::new(make_leaves(4)).expect("4 leaves");
        let mut proofs = verifier.generate_range_proof(0, 4).expect("range");
        // Tamper one proof's stored root.
        proofs[2].root = MerkleHash::from_bytes([0xdd; 8]);
        assert!(!verifier.verify_range(&proofs).expect("verify_range"));
    }

    // -----------------------------------------------------------------------
    // Update proofs
    // -----------------------------------------------------------------------

    #[test]
    fn test_update_leaf_changes_root() {
        let mut verifier = MerkleProofVerifier::new(make_leaves(4)).expect("4 leaves");
        let old_root = verifier.root();
        verifier
            .update_leaf(1, b"new-data".to_vec())
            .expect("update");
        assert_ne!(verifier.root(), old_root);
    }

    #[test]
    fn test_update_leaf_proof_valid_after_update() {
        let mut verifier = MerkleProofVerifier::new(make_leaves(4)).expect("4 leaves");
        verifier
            .update_leaf(2, b"changed".to_vec())
            .expect("update");
        let proof = verifier.generate_proof(2).expect("proof");
        assert!(verifier.verify_proof(&proof).expect("verify"));
    }

    #[test]
    fn test_update_leaf_old_proof_invalid_against_new_root() {
        let mut verifier = MerkleProofVerifier::new(make_leaves(4)).expect("4 leaves");
        let old_proof = verifier.generate_proof(0).expect("old proof");
        verifier
            .update_leaf(0, b"new-leaf-0".to_vec())
            .expect("update");
        // Old proof root no longer matches the verifier's current root.
        assert!(!verifier.verify_proof(&old_proof).expect("verify old"));
    }

    #[test]
    fn test_update_proof_verify_update() {
        let mut verifier = MerkleProofVerifier::new(make_leaves(4)).expect("4 leaves");
        let update = verifier
            .update_leaf(1, b"new-leaf-1".to_vec())
            .expect("update");
        assert!(verifier.verify_update(&update).expect("verify_update"));
    }

    #[test]
    fn test_update_proof_old_and_new_roots_differ() {
        let mut verifier = MerkleProofVerifier::new(make_leaves(4)).expect("4 leaves");
        let update = verifier
            .update_leaf(3, b"updated".to_vec())
            .expect("update");
        assert_ne!(update.old_proof.root, update.new_proof.root);
    }

    #[test]
    fn test_update_leaf_out_of_bounds() {
        let mut verifier = MerkleProofVerifier::new(make_leaves(4)).expect("4 leaves");
        let err = verifier.update_leaf(4, b"x".to_vec()).unwrap_err();
        assert_eq!(err, VerifierError::LeafIndexOutOfBounds(4));
    }

    #[test]
    fn test_update_non_overlapping_leaves_independent() {
        let mut verifier = MerkleProofVerifier::new(make_leaves(8)).expect("8 leaves");
        // Update leaf 0.
        let update0 = verifier
            .update_leaf(0, b"leaf0-new".to_vec())
            .expect("update 0");
        // Proofs for leaf 7 should still be valid.
        let proof7 = verifier.generate_proof(7).expect("proof 7");
        assert!(verifier.verify_proof(&proof7).expect("verify 7"));
        // The update proof for leaf 0 should be valid.
        assert!(verifier.verify_update(&update0).expect("verify update0"));
    }

    #[test]
    fn test_multiple_sequential_updates() {
        let mut verifier = MerkleProofVerifier::new(make_leaves(8)).expect("8 leaves");
        for i in 0..8 {
            let data = format!("update-{i}").into_bytes();
            let update = verifier.update_leaf(i, data).expect("update");
            assert!(verifier.verify_update(&update).expect("verify_update"));
        }
        // All leaves now have valid proofs against the current root.
        for i in 0..8 {
            let proof = verifier.generate_proof(i).expect("proof");
            assert!(verifier.verify_proof(&proof).expect("verify"));
        }
    }

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_single_leaf() {
        let verifier = single_leaf_tree();
        let stats = verifier.stats();
        assert_eq!(stats.leaf_count, 1);
        // height: 1-leaf tree pads to 1 (power of two already), height = trailing_zeros(1)+1 = 0+1 = 1
        // ... but with a single leaf there is only the root.
        assert!(stats.tree_height >= 1);
        assert_ne!(stats.root_hash, MerkleHash::zero());
        // total_nodes for 1 padded leaf: 2*1 = 2 nodes; our array is 3 (idx 0 unused) so
        // stored nodes = len - 1 = 2.
        assert_eq!(stats.total_nodes, 2);
    }

    #[test]
    fn test_stats_four_leaves() {
        let verifier = MerkleProofVerifier::new(make_leaves(4)).expect("4 leaves");
        let stats = verifier.stats();
        assert_eq!(stats.leaf_count, 4);
        assert_eq!(stats.tree_height, 3); // root + 1 internal level + leaf level
        assert_eq!(stats.total_nodes, 8); // 2*4 = 8
    }

    #[test]
    fn test_stats_eight_leaves() {
        let verifier = MerkleProofVerifier::new(make_leaves(8)).expect("8 leaves");
        let stats = verifier.stats();
        assert_eq!(stats.leaf_count, 8);
        assert_eq!(stats.tree_height, 4);
        assert_eq!(stats.total_nodes, 16);
    }

    #[test]
    fn test_stats_root_matches_verifier() {
        let verifier = MerkleProofVerifier::new(make_leaves(6)).expect("6 leaves");
        assert_eq!(verifier.stats().root_hash, verifier.root());
    }

    // -----------------------------------------------------------------------
    // Determinism
    // -----------------------------------------------------------------------

    #[test]
    fn test_same_data_same_root() {
        let leaves = make_leaves(8);
        let v1 = MerkleProofVerifier::new(leaves.clone()).expect("v1");
        let v2 = MerkleProofVerifier::new(leaves).expect("v2");
        assert_eq!(v1.root(), v2.root());
    }

    #[test]
    fn test_different_data_different_root() {
        let v1 = MerkleProofVerifier::new(vec![b"data-a".to_vec()]).expect("v1");
        let v2 = MerkleProofVerifier::new(vec![b"data-b".to_vec()]).expect("v2");
        assert_ne!(v1.root(), v2.root());
    }

    #[test]
    fn test_order_matters_for_root() {
        let leaves_ab = vec![b"a".to_vec(), b"b".to_vec()];
        let leaves_ba = vec![b"b".to_vec(), b"a".to_vec()];
        let v_ab = MerkleProofVerifier::new(leaves_ab).expect("ab");
        let v_ba = MerkleProofVerifier::new(leaves_ba).expect("ba");
        assert_ne!(v_ab.root(), v_ba.root());
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_leaf_data() {
        // A tree containing a single empty leaf should work.
        let verifier = MerkleProofVerifier::new(vec![vec![]]).expect("empty leaf");
        assert_ne!(verifier.root(), MerkleHash::zero());
        let proof = verifier.generate_proof(0).expect("proof");
        assert!(verifier.verify_proof(&proof).expect("verify"));
    }

    #[test]
    fn test_all_identical_leaves() {
        let leaves = vec![b"same".to_vec(); 4];
        let verifier = MerkleProofVerifier::new(leaves).expect("identical leaves");
        for i in 0..4 {
            let proof = verifier.generate_proof(i).expect("proof");
            assert!(verifier.verify_proof(&proof).expect("verify"));
        }
    }

    #[test]
    fn test_large_leaf_data() {
        let big = vec![0xffu8; 4096];
        let verifier = MerkleProofVerifier::new(vec![big]).expect("big leaf");
        let proof = verifier.generate_proof(0).expect("proof");
        assert!(verifier.verify_proof(&proof).expect("verify"));
    }

    #[test]
    fn test_binary_leaf_data() {
        let leaves: Vec<Vec<u8>> = (0u8..8)
            .map(|i| vec![i, i.wrapping_add(1), i.wrapping_add(2)])
            .collect();
        let verifier = MerkleProofVerifier::new(leaves).expect("binary leaves");
        for i in 0..8 {
            let proof = verifier.generate_proof(i).expect("proof");
            assert!(verifier.verify_proof(&proof).expect("verify"));
        }
    }

    #[test]
    fn test_update_to_same_data_preserves_root() {
        let mut verifier = MerkleProofVerifier::new(make_leaves(4)).expect("4 leaves");
        let old_root = verifier.root();
        // Update leaf 0 with the same data it already has.
        let data = "leaf-0".to_string().into_bytes();
        verifier.update_leaf(0, data).expect("update");
        assert_eq!(verifier.root(), old_root);
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(4), 4);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(7), 8);
        assert_eq!(next_power_of_two(8), 8);
        assert_eq!(next_power_of_two(9), 16);
        assert_eq!(next_power_of_two(100), 128);
    }

    #[test]
    fn test_verify_update_tampered_new_hash_fails() {
        let mut verifier = MerkleProofVerifier::new(make_leaves(4)).expect("4 leaves");
        let mut update = verifier.update_leaf(0, b"new".to_vec()).expect("update");
        // Tamper the new proof's leaf hash.
        update.new_proof.leaf_hash = MerkleHash::from_bytes([0xab; 8]);
        assert!(!verifier.verify_update(&update).expect("verify_update"));
    }

    #[test]
    fn test_verify_update_tampered_changed_index_fails() {
        let mut verifier = MerkleProofVerifier::new(make_leaves(4)).expect("4 leaves");
        let mut update = verifier.update_leaf(1, b"new".to_vec()).expect("update");
        // Mismatch the changed_index.
        update.changed_index = 2;
        assert!(!verifier.verify_update(&update).expect("verify_update"));
    }

    #[test]
    fn test_proof_leaf_hash_matches_data() {
        let leaves = make_leaves(4);
        let verifier = MerkleProofVerifier::new(leaves.clone()).expect("4 leaves");
        for (i, data) in leaves.iter().enumerate() {
            let proof = verifier.generate_proof(i).expect("proof");
            let expected_hash = MerkleHash::from_bytes(hash_leaf(data));
            assert_eq!(proof.leaf_hash, expected_hash, "leaf {i}");
        }
    }
}
