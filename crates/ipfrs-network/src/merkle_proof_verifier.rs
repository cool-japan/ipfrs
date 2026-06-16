//! Merkle inclusion proof verifier for content-addressed data.
//!
//! Supports multiple hash algorithms and proof formats, providing production-grade
//! verification for IPFS/IPFRS content addressing.

// ─── Pure-Rust SHA-256 ────────────────────────────────────────────────────────

/// First 64 fractional bits of the cube roots of the first 64 primes.
#[rustfmt::skip]
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-256 initial hash values (first 32 bits of fractional parts of sqrt of first 8 primes).
const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Compute SHA-256 of `data`, returning a 32-byte digest.
/// This is a self-contained pure-Rust implementation — no external crate used.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    // ── Pre-processing: padding ───────────────────────────────────────────────
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    // Pad to 56 mod 64 bytes (leaving 8 bytes for length).
    while msg.len() % 64 != 56 {
        msg.push(0x00);
    }
    // Append big-endian 64-bit bit length.
    msg.extend_from_slice(&bit_len.to_be_bytes());

    // ── Processing: 512-bit (64-byte) chunks ─────────────────────────────────
    let mut h = H0;
    for chunk in msg.chunks(64) {
        // Build message schedule W[0..64].
        let mut w = [0u32; 64];
        for (i, word) in w.iter_mut().enumerate().take(16) {
            let b = &chunk[i * 4..i * 4 + 4];
            *word = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        // ── 64-round compression ──────────────────────────────────────────────
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    // ── Produce final digest ──────────────────────────────────────────────────
    let mut out = [0u8; 32];
    for (i, &word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// FNV-1a 64-bit hash of `data`.
fn fnv1a_64(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;
    let mut hash = FNV_OFFSET;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ─── Hash algorithm enum ─────────────────────────────────────────────────────

/// Supported hashing algorithms for Merkle tree construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MerkleHashAlgo {
    /// Standard SHA-256 (pure Rust, no external crate).
    Sha256,
    /// Approximated Blake3: SHA-256 XOR'd with `[0xb3; 32]`.
    Blake3,
    /// FNV-1a 64-bit XOR approximation: first 8 bytes are the FNV-1a hash, rest zero,
    /// then XOR'd with the reversed FNV-1a hash of the input.
    FnvXor,
}

// ─── Core types ─────────────────────────────────────────────────────────────

/// A node in a Merkle tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MerkleNode {
    /// The hash stored at this node.
    pub hash: [u8; 32],
    /// Left child, if any.
    pub left: Option<Box<MerkleNode>>,
    /// Right child, if any.
    pub right: Option<Box<MerkleNode>>,
}

impl MerkleNode {
    /// Create a leaf node (no children).
    pub fn leaf(hash: [u8; 32]) -> Self {
        Self {
            hash,
            left: None,
            right: None,
        }
    }

    /// Create an internal node from two children.
    pub fn internal(hash: [u8; 32], left: MerkleNode, right: MerkleNode) -> Self {
        Self {
            hash,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
        }
    }
}

/// A single step in a Merkle inclusion proof.
///
/// Each step supplies the sibling hash and indicates on which side the sibling sits:
/// - `Left(h)` means the sibling is the *left* child → `hash_pair(h, current)`.
/// - `Right(h)` means the sibling is the *right* child → `hash_pair(current, h)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofStep {
    /// The sibling is on the left.
    Left([u8; 32]),
    /// The sibling is on the right.
    Right([u8; 32]),
}

/// A complete Merkle inclusion proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MerkleProof {
    /// Hash of the leaf being proved.
    pub leaf_hash: [u8; 32],
    /// Ordered list of proof steps from leaf to root.
    pub steps: Vec<ProofStep>,
    /// Expected root hash.
    pub root_hash: [u8; 32],
    /// Hash algorithm used to build the tree.
    pub algo: MerkleHashAlgo,
}

/// Result of verifying a single `MerkleProof`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationResult {
    /// Whether the proof is valid.
    pub valid: bool,
    /// The root hash computed from the proof steps.
    pub computed_root: [u8; 32],
    /// The expected root from the proof.
    pub expected_root: [u8; 32],
    /// Number of proof steps successfully processed.
    pub steps_verified: usize,
}

impl VerificationResult {
    /// Returns `true` if the proof verified successfully.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.valid
    }
}

/// An in-memory Merkle tree.
#[derive(Debug, Clone)]
pub struct MerkleTree {
    /// Original leaf hashes (after padding to next power-of-2).
    pub leaves: Vec<[u8; 32]>,
    /// All nodes stored level-order: leaves first, then their parents, … up to the root.
    /// Index layout (0-based, leaves are at indices `[leaves.len()-1 .. 2*leaves.len()-2]`
    /// in the canonical 1-indexed scheme; here we store them in a flat vec produced by
    /// level-order traversal where index 0 is the root).
    pub nodes: Vec<[u8; 32]>,
    /// Hash algorithm used.
    pub algo: MerkleHashAlgo,
    /// Tree depth (0 = single leaf/root).
    pub depth: usize,
}

// ─── Verifier ────────────────────────────────────────────────────────────────

/// Production-grade Merkle inclusion proof verifier.
///
/// Supports multiple hash algorithms and maintains cumulative statistics.
#[derive(Debug, Clone)]
pub struct MerkleProofVerifier {
    /// Hash algorithm to use for leaf and pair hashing.
    pub algo: MerkleHashAlgo,
    /// Total verifications performed (both valid and invalid).
    pub verifications_done: u64,
    /// Total failed (invalid) verifications.
    pub failures: u64,
}

impl MerkleProofVerifier {
    /// Create a new verifier using the specified algorithm.
    pub fn new(algo: MerkleHashAlgo) -> Self {
        Self {
            algo,
            verifications_done: 0,
            failures: 0,
        }
    }

    // ── Hashing primitives ────────────────────────────────────────────────────

    /// Hash a single leaf's raw data bytes.
    ///
    /// - `Sha256`: standard SHA-256.
    /// - `Blake3`: SHA-256 XOR `[0xb3; 32]`.
    /// - `FnvXor`: FNV-1a 64-bit stored in first 8 bytes (big-endian), rest zero,
    ///   then XOR with the reversed FNV-1a of the input (same 8-byte layout).
    pub fn hash_leaf(&self, data: &[u8]) -> [u8; 32] {
        Self::hash_leaf_with_algo(data, self.algo)
    }

    fn hash_leaf_with_algo(data: &[u8], algo: MerkleHashAlgo) -> [u8; 32] {
        match algo {
            MerkleHashAlgo::Sha256 => sha256(data),
            MerkleHashAlgo::Blake3 => {
                let mut digest = sha256(data);
                for byte in digest.iter_mut() {
                    *byte ^= 0xb3;
                }
                digest
            }
            MerkleHashAlgo::FnvXor => {
                let fwd = fnv1a_64(data);
                let rev = fnv1a_64(&data.iter().copied().rev().collect::<Vec<u8>>());
                let mut out = [0u8; 32];
                out[..8].copy_from_slice(&fwd.to_be_bytes());
                // XOR the first 8 bytes with the reversed hash.
                let rev_bytes = rev.to_be_bytes();
                for i in 0..8 {
                    out[i] ^= rev_bytes[i];
                }
                out
            }
        }
    }

    /// Hash a concatenated pair of child hashes (left ++ right = 64 bytes).
    pub fn hash_pair(&self, left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        Self::hash_pair_with_algo(left, right, self.algo)
    }

    fn hash_pair_with_algo(left: &[u8; 32], right: &[u8; 32], algo: MerkleHashAlgo) -> [u8; 32] {
        let mut combined = [0u8; 64];
        combined[..32].copy_from_slice(left);
        combined[32..].copy_from_slice(right);
        Self::hash_leaf_with_algo(&combined, algo)
    }

    // ── Tree construction ─────────────────────────────────────────────────────

    /// Build a Merkle tree from the given raw leaf data.
    ///
    /// Leaves are padded to the next power-of-two by duplicating the last leaf.
    /// Nodes are stored in a flat vec (root at index 0, level-order / breadth-first).
    pub fn build_tree(&self, leaves: &[Vec<u8>]) -> MerkleTree {
        if leaves.is_empty() {
            return MerkleTree {
                leaves: vec![],
                nodes: vec![],
                algo: self.algo,
                depth: 0,
            };
        }

        // Hash each leaf.
        let mut leaf_hashes: Vec<[u8; 32]> = leaves.iter().map(|l| self.hash_leaf(l)).collect();

        // Pad to next power-of-two.
        let n = leaf_hashes.len();
        let padded_len = n.next_power_of_two();
        if let Some(last) = leaf_hashes.last().copied() {
            while leaf_hashes.len() < padded_len {
                leaf_hashes.push(last);
            }
        }

        let depth = if padded_len == 1 {
            0
        } else {
            (padded_len as f64).log2() as usize
        };

        // Build nodes bottom-up. Total nodes = 2*padded_len - 1.
        // We store them in a flat array of size 2*padded_len - 1.
        // Index mapping (1-indexed): root = 1, children of i = 2i and 2i+1.
        // We convert to 0-indexed by subtracting 1.
        let total = 2 * padded_len - 1;
        let mut nodes = vec![[0u8; 32]; total];

        // Fill leaves at positions [padded_len-1 .. total-1] (0-indexed).
        for (i, &hash) in leaf_hashes.iter().enumerate() {
            nodes[padded_len - 1 + i] = hash;
        }

        // Build internal nodes bottom-up.
        if padded_len > 1 {
            for i in (0..padded_len - 1).rev() {
                let left = &nodes[2 * i + 1];
                let right = &nodes[2 * i + 2];
                nodes[i] = Self::hash_pair_with_algo(left, right, self.algo);
            }
        }

        MerkleTree {
            leaves: leaf_hashes,
            nodes,
            algo: self.algo,
            depth,
        }
    }

    // ── Proof generation ─────────────────────────────────────────────────────

    /// Generate an inclusion proof for the leaf at `leaf_index`.
    ///
    /// Returns `None` if `leaf_index` is out of range.
    pub fn generate_proof(&self, tree: &MerkleTree, leaf_index: usize) -> Option<MerkleProof> {
        let padded_len = tree.leaves.len();
        if padded_len == 0 || leaf_index >= padded_len {
            return None;
        }

        let total = tree.nodes.len();
        if total == 0 {
            return None;
        }

        let leaf_hash = tree.leaves[leaf_index];
        let root_hash = tree.nodes[0];
        let mut steps = Vec::new();

        // Start at the leaf's node index (0-indexed array: leaf i → node padded_len-1+i).
        let mut idx = padded_len - 1 + leaf_index;

        while idx > 0 {
            // Determine if this node is a left (even offset from parent's left child) or right child.
            // Parent of node i (0-indexed): (i-1)/2
            // Left child of parent p: 2*p+1
            // Right child: 2*p+2
            let is_left = (idx % 2) == 1; // odd index → left child, even → right child
            if is_left {
                // Sibling is to the right.
                let sibling_idx = idx + 1;
                if sibling_idx < total {
                    steps.push(ProofStep::Right(tree.nodes[sibling_idx]));
                }
            } else {
                // Sibling is to the left.
                let sibling_idx = idx - 1;
                steps.push(ProofStep::Left(tree.nodes[sibling_idx]));
            }
            idx = (idx - 1) / 2;
        }

        Some(MerkleProof {
            leaf_hash,
            steps,
            root_hash,
            algo: self.algo,
        })
    }

    // ── Proof verification ────────────────────────────────────────────────────

    /// Verify a single inclusion proof and update internal statistics.
    pub fn verify_proof(&mut self, proof: &MerkleProof) -> VerificationResult {
        let mut current = proof.leaf_hash;
        let mut steps_verified = 0;

        for step in &proof.steps {
            current = match step {
                ProofStep::Left(sibling) => {
                    Self::hash_pair_with_algo(sibling, &current, proof.algo)
                }
                ProofStep::Right(sibling) => {
                    Self::hash_pair_with_algo(&current, sibling, proof.algo)
                }
            };
            steps_verified += 1;
        }

        let valid = current == proof.root_hash;
        self.verifications_done += 1;
        if !valid {
            self.failures += 1;
        }

        VerificationResult {
            valid,
            computed_root: current,
            expected_root: proof.root_hash,
            steps_verified,
        }
    }

    /// Verify a batch of proofs, returning one result per proof.
    pub fn verify_batch(&mut self, proofs: &[MerkleProof]) -> Vec<VerificationResult> {
        proofs.iter().map(|p| self.verify_proof(p)).collect()
    }

    // ── Convenience helpers ───────────────────────────────────────────────────

    /// Build a tree from raw leaf data and return the root hash.
    pub fn root_of(&self, leaves: &[Vec<u8>]) -> [u8; 32] {
        let tree = self.build_tree(leaves);
        tree.nodes.first().copied().unwrap_or([0u8; 32])
    }

    /// Return `(verifications_done, failures)`.
    pub fn verifier_stats(&self) -> (u64, u64) {
        (self.verifications_done, self.failures)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::merkle_proof_verifier::{
        fnv1a_64, sha256, MerkleHashAlgo, MerkleNode, MerkleProof, MerkleProofVerifier, MerkleTree,
        ProofStep,
    };

    // ── SHA-256 primitive tests ────────────────────────────────────────────────

    /// 1. SHA-256 of empty string matches the well-known value.
    #[test]
    fn test_sha256_empty() {
        let digest = sha256(b"");
        let expected =
            hex_to_bytes("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(digest, expected);
    }

    /// 2. SHA-256("abc") matches the known reference output.
    ///
    /// The reference value `ba7816bf...` is verified against Python `hashlib.sha256(b"abc")`.
    #[test]
    fn test_sha256_abc() {
        let digest = sha256(b"abc");
        let expected =
            hex_to_bytes("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(digest, expected);
    }

    /// 3. SHA-256 is deterministic.
    #[test]
    fn test_sha256_deterministic() {
        let a = sha256(b"hello world");
        let b = sha256(b"hello world");
        assert_eq!(a, b);
    }

    /// 4. SHA-256 of different inputs produces different digests (collision resistance smoke test).
    #[test]
    fn test_sha256_distinct() {
        let a = sha256(b"foo");
        let b = sha256(b"bar");
        assert_ne!(a, b);
    }

    /// 5. SHA-256 output is always 32 bytes.
    #[test]
    fn test_sha256_output_len() {
        assert_eq!(sha256(b"test").len(), 32);
        assert_eq!(sha256(b"").len(), 32);
        assert_eq!(sha256(&[0u8; 1000]).len(), 32);
    }

    // ── FNV-1a tests ─────────────────────────────────────────────────────────

    /// 6. FNV-1a of known value.
    #[test]
    fn test_fnv1a_known() {
        // fnv1a_64("") = offset basis = 14695981039346656037
        assert_eq!(fnv1a_64(b""), 14_695_981_039_346_656_037_u64);
    }

    /// 7. FNV-1a is deterministic.
    #[test]
    fn test_fnv1a_deterministic() {
        assert_eq!(fnv1a_64(b"hello"), fnv1a_64(b"hello"));
    }

    /// 8. FNV-1a differs on different inputs.
    #[test]
    fn test_fnv1a_distinct() {
        assert_ne!(fnv1a_64(b"hello"), fnv1a_64(b"world"));
    }

    // ── Hash-leaf tests ───────────────────────────────────────────────────────

    /// 9. hash_leaf with Sha256 produces 32-byte output.
    #[test]
    fn test_hash_leaf_sha256_len() {
        let v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        assert_eq!(v.hash_leaf(b"data").len(), 32);
    }

    /// 10. hash_leaf with Blake3 differs from Sha256 for same input.
    #[test]
    fn test_hash_leaf_blake3_differs_from_sha256() {
        let vs = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let vb = MerkleProofVerifier::new(MerkleHashAlgo::Blake3);
        assert_ne!(vs.hash_leaf(b"data"), vb.hash_leaf(b"data"));
    }

    /// 11. hash_leaf with Blake3 equals SHA-256 XOR 0xb3 at every byte.
    #[test]
    fn test_hash_leaf_blake3_xor_correctness() {
        let vb = MerkleProofVerifier::new(MerkleHashAlgo::Blake3);
        let sha = sha256(b"test");
        let blake = vb.hash_leaf(b"test");
        for (s, b) in sha.iter().zip(blake.iter()) {
            assert_eq!(s ^ 0xb3, *b);
        }
    }

    /// 12. hash_leaf with FnvXor is deterministic.
    #[test]
    fn test_hash_leaf_fnvxor_deterministic() {
        let v = MerkleProofVerifier::new(MerkleHashAlgo::FnvXor);
        assert_eq!(v.hash_leaf(b"hello"), v.hash_leaf(b"hello"));
    }

    /// 13. hash_leaf with FnvXor differs from Sha256.
    #[test]
    fn test_hash_leaf_fnvxor_differs_from_sha256() {
        let vs = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let vf = MerkleProofVerifier::new(MerkleHashAlgo::FnvXor);
        assert_ne!(vs.hash_leaf(b"data"), vf.hash_leaf(b"data"));
    }

    // ── hash_pair tests ───────────────────────────────────────────────────────

    /// 14. hash_pair is not commutative (order matters).
    #[test]
    fn test_hash_pair_not_commutative() {
        let v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let a = [1u8; 32];
        let b = [2u8; 32];
        assert_ne!(v.hash_pair(&a, &b), v.hash_pair(&b, &a));
    }

    /// 15. hash_pair(h, h) == hash_pair(h, h) (idempotent for equal inputs).
    #[test]
    fn test_hash_pair_equal_inputs_deterministic() {
        let v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let h = [42u8; 32];
        assert_eq!(v.hash_pair(&h, &h), v.hash_pair(&h, &h));
    }

    // ── MerkleNode tests ─────────────────────────────────────────────────────

    /// 16. MerkleNode::leaf has no children.
    #[test]
    fn test_merkle_node_leaf() {
        let n = MerkleNode::leaf([0u8; 32]);
        assert!(n.left.is_none());
        assert!(n.right.is_none());
    }

    /// 17. MerkleNode::internal stores children.
    #[test]
    fn test_merkle_node_internal() {
        let l = MerkleNode::leaf([1u8; 32]);
        let r = MerkleNode::leaf([2u8; 32]);
        let parent = MerkleNode::internal([3u8; 32], l, r);
        assert!(parent.left.is_some());
        assert!(parent.right.is_some());
    }

    // ── Tree construction tests ───────────────────────────────────────────────

    /// 18. build_tree on an empty slice returns an empty tree.
    #[test]
    fn test_build_tree_empty() {
        let v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let tree = v.build_tree(&[]);
        assert!(tree.leaves.is_empty());
        assert!(tree.nodes.is_empty());
    }

    /// 19. build_tree on a single leaf: root equals hash_leaf.
    #[test]
    fn test_build_tree_single_leaf() {
        let v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let data = vec![b"hello".to_vec()];
        let tree = v.build_tree(&data);
        assert_eq!(tree.depth, 0);
        // Root should be hash_pair(leaf, leaf) for the padded single-leaf tree
        // (padded_len = 1, so total = 1 node, which is the leaf itself).
        assert_eq!(tree.nodes[0], v.hash_leaf(b"hello"));
    }

    /// 20. build_tree pads to next power-of-two.
    #[test]
    fn test_build_tree_padding() {
        let v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let data: Vec<Vec<u8>> = (0..3u8).map(|i| vec![i]).collect();
        let tree = v.build_tree(&data);
        assert_eq!(tree.leaves.len(), 4); // padded to 4
    }

    /// 21. build_tree with exactly 4 leaves has depth 2.
    #[test]
    fn test_build_tree_four_leaves_depth() {
        let v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let data: Vec<Vec<u8>> = (0..4u8).map(|i| vec![i]).collect();
        let tree = v.build_tree(&data);
        assert_eq!(tree.depth, 2);
        assert_eq!(tree.nodes.len(), 7); // 2*4-1
    }

    /// 22. build_tree root equals root_of.
    #[test]
    fn test_build_tree_root_matches_root_of() {
        let v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let data: Vec<Vec<u8>> = (0..4u8).map(|i| vec![i]).collect();
        let tree = v.build_tree(&data);
        let root = v.root_of(&data);
        assert_eq!(tree.nodes[0], root);
    }

    // ── Proof generation and verification (round-trip) ────────────────────────

    fn make_leaves(n: usize) -> Vec<Vec<u8>> {
        (0..n).map(|i| format!("leaf-{i}").into_bytes()).collect()
    }

    fn round_trip(algo: MerkleHashAlgo, n_leaves: usize, leaf_idx: usize) -> bool {
        let mut v = MerkleProofVerifier::new(algo);
        let leaves = make_leaves(n_leaves);
        let tree = v.build_tree(&leaves);
        let proof = v.generate_proof(&tree, leaf_idx).expect("proof");
        v.verify_proof(&proof).valid
    }

    /// 23. Round-trip: generate + verify, Sha256, 4 leaves, leaf 0.
    #[test]
    fn test_round_trip_sha256_4_leaf0() {
        assert!(round_trip(MerkleHashAlgo::Sha256, 4, 0));
    }

    /// 24. Round-trip: generate + verify, Sha256, 4 leaves, leaf 3.
    #[test]
    fn test_round_trip_sha256_4_leaf3() {
        assert!(round_trip(MerkleHashAlgo::Sha256, 4, 3));
    }

    /// 25. Round-trip: generate + verify, Blake3, 8 leaves, leaf 5.
    #[test]
    fn test_round_trip_blake3_8_leaf5() {
        assert!(round_trip(MerkleHashAlgo::Blake3, 8, 5));
    }

    /// 26. Round-trip: generate + verify, FnvXor, 4 leaves, leaf 2.
    #[test]
    fn test_round_trip_fnvxor_4_leaf2() {
        assert!(round_trip(MerkleHashAlgo::FnvXor, 4, 2));
    }

    /// 27. Round-trip with non-power-of-2 leaves (5 leaves).
    #[test]
    fn test_round_trip_non_power_of_two() {
        assert!(round_trip(MerkleHashAlgo::Sha256, 5, 2));
        assert!(round_trip(MerkleHashAlgo::Sha256, 5, 4));
    }

    /// 28. Tampered leaf hash → invalid proof.
    #[test]
    fn test_tampered_leaf_fails() {
        let mut v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let leaves = make_leaves(4);
        let tree = v.build_tree(&leaves);
        let mut proof = v.generate_proof(&tree, 0).expect("proof");
        proof.leaf_hash[0] ^= 0xff; // corrupt leaf
        let result = v.verify_proof(&proof);
        assert!(!result.valid);
    }

    /// 29. Tampered step hash → invalid proof.
    #[test]
    fn test_tampered_step_fails() {
        let mut v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let leaves = make_leaves(4);
        let tree = v.build_tree(&leaves);
        let mut proof = v.generate_proof(&tree, 1).expect("proof");
        if let Some(step) = proof.steps.first_mut() {
            match step {
                ProofStep::Left(h) | ProofStep::Right(h) => h[0] ^= 0xff,
            }
        }
        let result = v.verify_proof(&proof);
        assert!(!result.valid);
    }

    /// 30. Tampered root hash → invalid proof.
    #[test]
    fn test_tampered_root_fails() {
        let mut v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let leaves = make_leaves(4);
        let tree = v.build_tree(&leaves);
        let mut proof = v.generate_proof(&tree, 0).expect("proof");
        proof.root_hash[0] ^= 0x01;
        let result = v.verify_proof(&proof);
        assert!(!result.valid);
    }

    // ── Statistics tests ──────────────────────────────────────────────────────

    /// 31. verifier_stats initially zero.
    #[test]
    fn test_stats_initial() {
        let v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        assert_eq!(v.verifier_stats(), (0, 0));
    }

    /// 32. verifier_stats increments on each verify_proof call.
    #[test]
    fn test_stats_increments() {
        let mut v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let leaves = make_leaves(4);
        let tree = v.build_tree(&leaves);
        let proof = v.generate_proof(&tree, 0).expect("proof");
        v.verify_proof(&proof);
        v.verify_proof(&proof);
        assert_eq!(v.verifier_stats(), (2, 0));
    }

    /// 33. failures counter increments on invalid proof.
    #[test]
    fn test_stats_failures() {
        let mut v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let leaves = make_leaves(4);
        let tree = v.build_tree(&leaves);
        let mut proof = v.generate_proof(&tree, 0).expect("proof");
        proof.root_hash[0] ^= 0x01;
        v.verify_proof(&proof);
        assert_eq!(v.verifier_stats(), (1, 1));
    }

    // ── Batch verification ────────────────────────────────────────────────────

    /// 34. verify_batch returns a result for each proof.
    #[test]
    fn test_verify_batch_count() {
        let mut v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let leaves = make_leaves(4);
        let tree = v.build_tree(&leaves);
        let proofs: Vec<MerkleProof> = (0..4).filter_map(|i| v.generate_proof(&tree, i)).collect();
        let results = v.verify_batch(&proofs);
        assert_eq!(results.len(), 4);
    }

    /// 35. verify_batch: all valid proofs from a correct tree.
    #[test]
    fn test_verify_batch_all_valid() {
        let mut v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let leaves = make_leaves(8);
        let tree = v.build_tree(&leaves);
        let proofs: Vec<MerkleProof> = (0..8).filter_map(|i| v.generate_proof(&tree, i)).collect();
        let results = v.verify_batch(&proofs);
        assert!(results.iter().all(|r| r.valid));
    }

    /// 36. verify_batch updates verifications_done.
    #[test]
    fn test_verify_batch_updates_stats() {
        let mut v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let leaves = make_leaves(4);
        let tree = v.build_tree(&leaves);
        let proofs: Vec<MerkleProof> = (0..4).filter_map(|i| v.generate_proof(&tree, i)).collect();
        v.verify_batch(&proofs);
        assert_eq!(v.verifications_done, 4);
    }

    // ── generate_proof edge cases ─────────────────────────────────────────────

    /// 37. generate_proof returns None for out-of-range leaf_index.
    #[test]
    fn test_generate_proof_out_of_range() {
        let v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let leaves = make_leaves(4);
        let tree = v.build_tree(&leaves);
        assert!(v.generate_proof(&tree, 10).is_none());
    }

    /// 38. generate_proof returns None for an empty tree.
    #[test]
    fn test_generate_proof_empty_tree() {
        let v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let empty_tree = MerkleTree {
            leaves: vec![],
            nodes: vec![],
            algo: MerkleHashAlgo::Sha256,
            depth: 0,
        };
        assert!(v.generate_proof(&empty_tree, 0).is_none());
    }

    /// 39. Proof for a single-leaf tree has zero steps.
    #[test]
    fn test_proof_single_leaf_no_steps() {
        let v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let leaves = vec![b"only".to_vec()];
        let tree = v.build_tree(&leaves);
        let proof = v.generate_proof(&tree, 0).expect("proof");
        assert_eq!(proof.steps.len(), 0);
    }

    /// 40. Proof steps count for depth-2 tree (4 leaves) is 2.
    #[test]
    fn test_proof_steps_count_depth2() {
        let v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let leaves = make_leaves(4);
        let tree = v.build_tree(&leaves);
        let proof = v.generate_proof(&tree, 0).expect("proof");
        assert_eq!(proof.steps.len(), 2);
    }

    // ── VerificationResult fields ─────────────────────────────────────────────

    /// 41. VerificationResult.computed_root matches expected_root on valid proof.
    #[test]
    fn test_verification_result_roots_match_on_valid() {
        let mut v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let leaves = make_leaves(4);
        let tree = v.build_tree(&leaves);
        let proof = v.generate_proof(&tree, 1).expect("proof");
        let res = v.verify_proof(&proof);
        assert_eq!(res.computed_root, res.expected_root);
        assert!(res.is_valid());
    }

    /// 42. VerificationResult.steps_verified equals proof.steps.len() on valid proof.
    #[test]
    fn test_verification_result_steps_verified() {
        let mut v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let leaves = make_leaves(4);
        let tree = v.build_tree(&leaves);
        let proof = v.generate_proof(&tree, 2).expect("proof");
        let n_steps = proof.steps.len();
        let res = v.verify_proof(&proof);
        assert_eq!(res.steps_verified, n_steps);
    }

    // ── Cross-algorithm correctness ────────────────────────────────────────────

    /// 43. Different algo → different roots.
    #[test]
    fn test_different_algo_different_roots() {
        let vs = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let vb = MerkleProofVerifier::new(MerkleHashAlgo::Blake3);
        let leaves = make_leaves(4);
        assert_ne!(vs.root_of(&leaves), vb.root_of(&leaves));
    }

    /// 44. Proof built with Sha256 fails against Blake3 verifier.
    #[test]
    fn test_cross_algo_proof_fails() {
        let vs = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let leaves = make_leaves(4);
        let sha_tree = vs.build_tree(&leaves);
        let sha_proof = vs.generate_proof(&sha_tree, 0).expect("proof");

        // Verify using a Blake3 verifier (by manually overriding proof.algo).
        let mut blake_proof = sha_proof.clone();
        blake_proof.algo = MerkleHashAlgo::Blake3;

        let mut vb = MerkleProofVerifier::new(MerkleHashAlgo::Blake3);
        let result = vb.verify_proof(&blake_proof);
        // The computed root will differ from the sha256 root, so invalid.
        assert!(!result.valid);
    }

    // ── Large tree tests ──────────────────────────────────────────────────────

    /// 45. Round-trip for 1024-leaf tree (depth 10), several indices.
    #[test]
    fn test_large_tree_round_trip() {
        let mut v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let leaves: Vec<Vec<u8>> = (0u32..1024).map(|i| i.to_le_bytes().to_vec()).collect();
        let tree = v.build_tree(&leaves);
        assert_eq!(tree.depth, 10);
        for &idx in &[0, 1, 511, 512, 1023] {
            let proof = v.generate_proof(&tree, idx).expect("proof");
            let res = v.verify_proof(&proof);
            assert!(res.valid, "failed at leaf {idx}");
        }
    }

    /// 46. All leaves in an 8-leaf tree can be proved and verified.
    #[test]
    fn test_all_leaves_8() {
        let mut v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let leaves = make_leaves(8);
        let tree = v.build_tree(&leaves);
        for i in 0..8 {
            let proof = v.generate_proof(&tree, i).expect("proof");
            assert!(v.verify_proof(&proof).valid, "leaf {i} failed");
        }
    }

    // ── root_of tests ─────────────────────────────────────────────────────────

    /// 47. root_of empty slice returns [0; 32].
    #[test]
    fn test_root_of_empty() {
        let v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        assert_eq!(v.root_of(&[]), [0u8; 32]);
    }

    /// 48. root_of is consistent across calls.
    #[test]
    fn test_root_of_consistent() {
        let v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let leaves = make_leaves(4);
        assert_eq!(v.root_of(&leaves), v.root_of(&leaves));
    }

    // ── ProofStep enum ────────────────────────────────────────────────────────

    /// 49. ProofStep::Left and Right can be pattern-matched.
    #[test]
    fn test_proof_step_pattern_match() {
        let h = [7u8; 32];
        let step_l = ProofStep::Left(h);
        let step_r = ProofStep::Right(h);
        match step_l {
            ProofStep::Left(inner) => assert_eq!(inner, h),
            ProofStep::Right(_) => panic!("wrong variant"),
        }
        match step_r {
            ProofStep::Right(inner) => assert_eq!(inner, h),
            ProofStep::Left(_) => panic!("wrong variant"),
        }
    }

    // ── MerkleProof round-trip with clone ─────────────────────────────────────

    /// 50. MerkleProof can be cloned and still verifies.
    #[test]
    fn test_proof_clone_verifies() {
        let mut v = MerkleProofVerifier::new(MerkleHashAlgo::Sha256);
        let leaves = make_leaves(4);
        let tree = v.build_tree(&leaves);
        let proof = v.generate_proof(&tree, 2).expect("proof");
        let proof2 = proof.clone();
        assert!(v.verify_proof(&proof2).valid);
    }

    // ── Helper ────────────────────────────────────────────────────────────────

    fn hex_to_bytes(hex: &str) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
            if i >= 32 {
                break;
            }
            let hi = hex_nibble(chunk[0]);
            let lo = hex_nibble(chunk.get(1).copied().unwrap_or(b'0'));
            out[i] = (hi << 4) | lo;
        }
        out
    }

    fn hex_nibble(b: u8) -> u8 {
        match b {
            b'0'..=b'9' => b - b'0',
            b'a'..=b'f' => b - b'a' + 10,
            b'A'..=b'F' => b - b'A' + 10,
            _ => 0,
        }
    }
}
