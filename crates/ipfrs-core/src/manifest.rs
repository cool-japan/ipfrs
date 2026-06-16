//! Content manifests for tracking multi-file add operations
//!
//! This module provides:
//! - [`ManifestEntry`] - metadata for a single content-addressed block in a manifest
//! - [`ContentManifest`] - tracks all blocks for a multi-file operation with Merkle integrity
//! - [`MerkleTree`] - binary Merkle tree over CID strings using FNV-1a hashing
//! - [`ManifestDiff`] - diff two manifests to find added/removed entries
//!
//! ## Design
//!
//! `manifest_id` and `root_cid` are both derived deterministically from the entry CIDs,
//! so the same set of entries always produces the same manifest identity and root.
//!
//! FNV-1a (64-bit) is used throughout as a fast, dependency-free hash primitive.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

// ---------------------------------------------------------------------------
// FNV-1a helpers (no external dependency)
// ---------------------------------------------------------------------------

const FNV_OFFSET_BASIS_64: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME_64: u64 = 1_099_511_628_211;

/// Compute a 64-bit FNV-1a hash of a byte slice.
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS_64;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME_64);
    }
    hash
}

/// Format a u64 FNV-1a hash as a 16-character lowercase hex string.
fn fnv1a_hex(data: &[u8]) -> String {
    format!("{:016x}", fnv1a_64(data))
}

/// Combine two hex-encoded hash strings into one FNV-1a hash.
///
/// This is the *node hash* function used by [`MerkleTree`]:
/// `hash(a, b) = FNV-1a(bytes(a) ++ bytes(b))`.
fn combine_hashes(left: &str, right: &str) -> String {
    let mut combined = Vec::with_capacity(left.len() + right.len());
    combined.extend_from_slice(left.as_bytes());
    combined.extend_from_slice(right.as_bytes());
    fnv1a_hex(&combined)
}

// ---------------------------------------------------------------------------
// ManifestEntry
// ---------------------------------------------------------------------------

/// A single entry inside a [`ContentManifest`].
///
/// An entry corresponds to one content-addressed block within a (possibly
/// chunked) file that was added as part of a multi-file operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestEntry {
    /// Relative path of the file within the manifest (e.g. `"images/photo.jpg"`).
    pub path: String,

    /// Content Identifier for this block.
    pub cid: String,

    /// Size of this block's raw data in bytes.
    pub size_bytes: u64,

    /// Zero-based index of this chunk within its file.
    /// Always `0` for single-chunk files.
    pub chunk_index: u32,

    /// `true` when this entry is the last chunk of the file at `path`.
    pub is_final_chunk: bool,
}

impl ManifestEntry {
    /// Create a new manifest entry.
    pub fn new(
        path: impl Into<String>,
        cid: impl Into<String>,
        size_bytes: u64,
        chunk_index: u32,
        is_final_chunk: bool,
    ) -> Self {
        Self {
            path: path.into(),
            cid: cid.into(),
            size_bytes,
            chunk_index,
            is_final_chunk,
        }
    }
}

// ---------------------------------------------------------------------------
// MerkleTree
// ---------------------------------------------------------------------------

/// Binary Merkle tree whose leaves are CID strings.
///
/// ### Hash function
///
/// Leaf hashes are `FNV-1a(cid.as_bytes())`.
/// Internal nodes use `FNV-1a(left_hex ++ right_hex)`.
/// When a level has an odd number of nodes the last node is *duplicated*
/// (standard Bitcoin/IPFS convention).
///
/// All hash values are stored and returned as 16-character lowercase hex strings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MerkleTree {
    /// Original CID strings used as leaves (in the order they were supplied).
    pub leaves: Vec<String>,

    /// All tree nodes stored level-by-level, leaves first.
    ///
    /// Layout: `[ leaf_hashes … | level-1 hashes … | … | root ]`
    pub nodes: Vec<String>,
}

impl MerkleTree {
    /// Build a Merkle tree from a slice of CID strings.
    ///
    /// Returns an empty tree (no nodes, no root) when `cids` is empty.
    pub fn build(cids: &[String]) -> Self {
        if cids.is_empty() {
            return Self {
                leaves: Vec::new(),
                nodes: Vec::new(),
            };
        }

        // Hash each leaf CID.
        let leaf_hashes: Vec<String> = cids.iter().map(|c| fnv1a_hex(c.as_bytes())).collect();

        // We accumulate all levels into `nodes` (leaves first).
        let mut all_nodes: Vec<String> = leaf_hashes.clone();
        let mut current_level = leaf_hashes;

        while current_level.len() > 1 {
            let mut next_level: Vec<String> = Vec::new();

            let mut i = 0;
            while i < current_level.len() {
                let left = &current_level[i];
                // Duplicate the last node if the level has an odd count.
                let right = if i + 1 < current_level.len() {
                    &current_level[i + 1]
                } else {
                    &current_level[i]
                };
                next_level.push(combine_hashes(left, right));
                i += 2;
            }

            all_nodes.extend(next_level.iter().cloned());
            current_level = next_level;
        }

        Self {
            leaves: cids.to_vec(),
            nodes: all_nodes,
        }
    }

    /// Return the Merkle root as a hex string, or `None` for an empty tree.
    pub fn root(&self) -> Option<&str> {
        self.nodes.last().map(|s| s.as_str())
    }

    /// Return the Merkle proof for the leaf at `index`.
    ///
    /// The proof is a list of sibling hashes from leaf level up to (but not
    /// including) the root.  Returns `None` if `index` is out of range or
    /// the tree is empty.
    pub fn proof_for(&self, index: usize) -> Option<Vec<String>> {
        let n = self.leaves.len();
        if n == 0 || index >= n {
            return None;
        }

        let mut proof = Vec::new();
        let mut current_index = index;
        let mut level_size = n;
        let mut level_start = 0usize;

        while level_size > 1 {
            // Determine sibling index.
            let sibling_index = if current_index.is_multiple_of(2) {
                // We are a left child; sibling is to the right (or self if last).
                (current_index + 1).min(level_size - 1)
            } else {
                // We are a right child; sibling is to the left.
                current_index - 1
            };

            proof.push(self.nodes[level_start + sibling_index].clone());

            level_start += level_size;
            level_size = level_size.div_ceil(2);
            current_index /= 2;
        }

        Some(proof)
    }

    /// Verify a Merkle proof.
    ///
    /// - `leaf` — the raw CID string whose membership is being verified
    /// - `index` — zero-based position of the leaf in the original list
    /// - `proof` — sibling hashes returned by [`MerkleTree::proof_for`]
    /// - `root` — expected Merkle root hex string
    ///
    /// Returns `true` iff the recomputed root matches `root`.
    pub fn verify_proof(leaf: &str, index: usize, proof: &[String], root: &str) -> bool {
        let mut current_hash = fnv1a_hex(leaf.as_bytes());
        let mut current_index = index;

        for sibling in proof {
            current_hash = if current_index.is_multiple_of(2) {
                combine_hashes(&current_hash, sibling)
            } else {
                combine_hashes(sibling, &current_hash)
            };
            current_index /= 2;
        }

        current_hash == root
    }
}

// ---------------------------------------------------------------------------
// ContentManifest
// ---------------------------------------------------------------------------

/// Tracks all blocks belonging to a multi-file add operation.
///
/// The manifest provides:
/// - Ordered entry lookup by path and chunk index
/// - Completeness checks for individual paths
/// - A Merkle root for efficient integrity verification and partial retrieval
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentManifest {
    /// Unique manifest identifier derived as the FNV-1a hash of all entry CIDs
    /// sorted lexicographically and concatenated.
    pub manifest_id: String,

    /// All entries, sorted by `(path, chunk_index)`.
    pub entries: Vec<ManifestEntry>,

    /// Hex string of the Merkle root over all entry CIDs (sorted by path then
    /// chunk_index), or an empty string when there are no entries.
    pub root_cid: String,

    /// Sum of `size_bytes` across all entries.
    pub total_size_bytes: u64,

    /// Unix timestamp in milliseconds at manifest creation time.
    pub created_at_ms: u64,
}

impl ContentManifest {
    /// Create a new manifest from a list of entries.
    ///
    /// Entries are sorted by `(path, chunk_index)` in place.
    /// `manifest_id`, `root_cid`, and `total_size_bytes` are computed
    /// deterministically from the supplied entries.
    pub fn new(mut entries: Vec<ManifestEntry>) -> Self {
        // Canonical sort: path first, then chunk_index.
        entries.sort_by(|a, b| {
            a.path
                .cmp(&b.path)
                .then_with(|| a.chunk_index.cmp(&b.chunk_index))
        });

        let total_size_bytes: u64 = entries.iter().map(|e| e.size_bytes).sum();

        // manifest_id = FNV-1a of all CIDs sorted lexicographically.
        let mut sorted_cids: Vec<&str> = entries.iter().map(|e| e.cid.as_str()).collect();
        sorted_cids.sort_unstable();
        let id_input: String = sorted_cids.join("");
        let manifest_id = fnv1a_hex(id_input.as_bytes());

        // root_cid from Merkle tree over entry CIDs in canonical (path, chunk_index) order.
        let ordered_cids: Vec<String> = entries.iter().map(|e| e.cid.clone()).collect();
        let tree = MerkleTree::build(&ordered_cids);
        let root_cid = tree.root().unwrap_or("").to_string();

        // Timestamp using wasm-compatible platform time.
        let created_at_ms = created_at_ms_now();

        Self {
            manifest_id,
            entries,
            root_cid,
            total_size_bytes,
            created_at_ms,
        }
    }

    /// Return all entries whose `path` equals `path`, in chunk order.
    pub fn get_entries_for_path<'a>(&'a self, path: &str) -> Vec<&'a ManifestEntry> {
        self.entries.iter().filter(|e| e.path == path).collect()
    }

    /// Return the total number of entries (chunks) for a given path.
    pub fn total_chunks_for_path(&self, path: &str) -> usize {
        self.entries.iter().filter(|e| e.path == path).count()
    }

    /// Return `true` if the path is fully represented in this manifest.
    ///
    /// A path is *complete* when:
    /// 1. There is at least one entry for the path.
    /// 2. The entries with the highest `chunk_index` has `is_final_chunk = true`.
    /// 3. All chunk indices from `0` to `max_chunk_index` are present (no gaps).
    pub fn is_complete_for_path(&self, path: &str) -> bool {
        let path_entries: Vec<&ManifestEntry> =
            self.entries.iter().filter(|e| e.path == path).collect();

        if path_entries.is_empty() {
            return false;
        }

        // There must be exactly one entry marked as final.
        let final_entries: Vec<&&ManifestEntry> =
            path_entries.iter().filter(|e| e.is_final_chunk).collect();

        if final_entries.len() != 1 {
            return false;
        }

        let max_index = final_entries[0].chunk_index;
        let expected_count = (max_index as usize) + 1;

        if path_entries.len() != expected_count {
            return false;
        }

        // Verify no gaps: collect all indices and check 0..=max_index are present.
        let indices: BTreeSet<u32> = path_entries.iter().map(|e| e.chunk_index).collect();
        for idx in 0..=max_index {
            if !indices.contains(&idx) {
                return false;
            }
        }

        true
    }

    /// Return the total number of entries in this manifest.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Return a sorted, deduplicated list of all paths in this manifest.
    pub fn paths(&self) -> Vec<String> {
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for entry in &self.entries {
            seen.insert(entry.path.as_str());
        }
        seen.into_iter().map(|s| s.to_string()).collect()
    }
}

// ---------------------------------------------------------------------------
// ManifestDiff
// ---------------------------------------------------------------------------

/// The result of comparing two [`ContentManifest`] instances.
///
/// Comparison is done by CID: an entry present in `new` but absent in `old`
/// (by CID) is *added*; an entry present in `old` but absent in `new` is
/// *removed*.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestDiff {
    /// Entries that exist in the new manifest but not the old one.
    pub added: Vec<ManifestEntry>,

    /// Entries that exist in the old manifest but not the new one.
    pub removed: Vec<ManifestEntry>,
}

impl ManifestDiff {
    /// Compute the diff between `old` and `new` manifests.
    pub fn diff(old: &ContentManifest, new: &ContentManifest) -> Self {
        let old_cids: BTreeMap<&str, &ManifestEntry> =
            old.entries.iter().map(|e| (e.cid.as_str(), e)).collect();
        let new_cids: BTreeMap<&str, &ManifestEntry> =
            new.entries.iter().map(|e| (e.cid.as_str(), e)).collect();

        let added: Vec<ManifestEntry> = new_cids
            .iter()
            .filter(|(cid, _)| !old_cids.contains_key(*cid))
            .map(|(_, e)| (*e).clone())
            .collect();

        let removed: Vec<ManifestEntry> = old_cids
            .iter()
            .filter(|(cid, _)| !new_cids.contains_key(*cid))
            .map(|(_, e)| (*e).clone())
            .collect();

        Self { added, removed }
    }

    /// Return `true` when there are no differences.
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Return the current wall-clock time in milliseconds since Unix epoch.
///
/// On wasm32 targets (where `std::time::SystemTime` is unavailable) this
/// falls back to 0.  Tests that need a stable value should override
/// `created_at_ms` directly after construction.
fn created_at_ms_now() -> u64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
    #[cfg(target_arch = "wasm32")]
    {
        0
    }
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

    fn entry(path: &str, cid: &str, size: u64, idx: u32, final_chunk: bool) -> ManifestEntry {
        ManifestEntry::new(path, cid, size, idx, final_chunk)
    }

    fn sample_entries() -> Vec<ManifestEntry> {
        vec![
            entry("a/b.txt", "cid-ab-0", 100, 0, false),
            entry("a/b.txt", "cid-ab-1", 200, 1, true),
            entry("c/d.bin", "cid-cd-0", 50, 0, true),
        ]
    }

    // -----------------------------------------------------------------------
    // ContentManifest::new
    // -----------------------------------------------------------------------

    #[test]
    fn test_manifest_new_total_size() {
        let m = ContentManifest::new(sample_entries());
        assert_eq!(m.total_size_bytes, 350);
    }

    #[test]
    fn test_manifest_new_entry_count() {
        let m = ContentManifest::new(sample_entries());
        assert_eq!(m.entry_count(), 3);
    }

    #[test]
    fn test_manifest_new_sorted_entries() {
        // Supply entries in reverse order; they must come out sorted.
        let entries = vec![
            entry("z/last.txt", "cid-z", 10, 0, true),
            entry("a/first.txt", "cid-a", 20, 0, true),
        ];
        let m = ContentManifest::new(entries);
        assert_eq!(m.entries[0].path, "a/first.txt");
        assert_eq!(m.entries[1].path, "z/last.txt");
    }

    // -----------------------------------------------------------------------
    // get_entries_for_path
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_entries_for_path_found() {
        let m = ContentManifest::new(sample_entries());
        let entries = m.get_entries_for_path("a/b.txt");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].chunk_index, 0);
        assert_eq!(entries[1].chunk_index, 1);
    }

    #[test]
    fn test_get_entries_for_path_not_found() {
        let m = ContentManifest::new(sample_entries());
        let entries = m.get_entries_for_path("nonexistent.txt");
        assert!(entries.is_empty());
    }

    // -----------------------------------------------------------------------
    // paths()
    // -----------------------------------------------------------------------

    #[test]
    fn test_paths_unique_sorted() {
        let m = ContentManifest::new(sample_entries());
        let paths = m.paths();
        assert_eq!(paths, vec!["a/b.txt", "c/d.bin"]);
    }

    #[test]
    fn test_paths_empty_manifest() {
        let m = ContentManifest::new(vec![]);
        assert!(m.paths().is_empty());
    }

    // -----------------------------------------------------------------------
    // is_complete_for_path
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_complete_for_path_single_chunk() {
        let m = ContentManifest::new(sample_entries());
        assert!(m.is_complete_for_path("c/d.bin"));
    }

    #[test]
    fn test_is_complete_for_path_multi_chunk_complete() {
        let m = ContentManifest::new(sample_entries());
        assert!(m.is_complete_for_path("a/b.txt"));
    }

    #[test]
    fn test_is_complete_for_path_missing_final() {
        // Two chunks, neither is marked final.
        let entries = vec![
            entry("f.txt", "cid-f-0", 10, 0, false),
            entry("f.txt", "cid-f-1", 10, 1, false),
        ];
        let m = ContentManifest::new(entries);
        assert!(!m.is_complete_for_path("f.txt"));
    }

    #[test]
    fn test_is_complete_for_path_gap() {
        // Chunk 0 and chunk 2 present, chunk 1 missing.
        let entries = vec![
            entry("g.bin", "cid-g-0", 10, 0, false),
            entry("g.bin", "cid-g-2", 10, 2, true),
        ];
        let m = ContentManifest::new(entries);
        assert!(!m.is_complete_for_path("g.bin"));
    }

    #[test]
    fn test_is_complete_for_path_nonexistent() {
        let m = ContentManifest::new(sample_entries());
        assert!(!m.is_complete_for_path("does_not_exist.txt"));
    }

    // -----------------------------------------------------------------------
    // manifest_id determinism
    // -----------------------------------------------------------------------

    #[test]
    fn test_manifest_id_deterministic() {
        let m1 = ContentManifest::new(sample_entries());
        let m2 = ContentManifest::new(sample_entries());
        assert_eq!(m1.manifest_id, m2.manifest_id);
    }

    #[test]
    fn test_manifest_id_order_independent() {
        // manifest_id is based on *sorted* CIDs, so entry order must not matter.
        let e1 = vec![
            entry("a.txt", "cid-X", 10, 0, true),
            entry("b.txt", "cid-Y", 20, 0, true),
        ];
        let e2 = vec![
            entry("b.txt", "cid-Y", 20, 0, true),
            entry("a.txt", "cid-X", 10, 0, true),
        ];
        let m1 = ContentManifest::new(e1);
        let m2 = ContentManifest::new(e2);
        assert_eq!(m1.manifest_id, m2.manifest_id);
    }

    // -----------------------------------------------------------------------
    // MerkleTree::build
    // -----------------------------------------------------------------------

    #[test]
    fn test_merkle_tree_single_leaf() {
        let cids = vec!["cid-1".to_string()];
        let tree = MerkleTree::build(&cids);
        assert_eq!(tree.leaves.len(), 1);
        // With one leaf the root IS the leaf hash.
        let leaf_hash = fnv1a_hex(b"cid-1");
        assert_eq!(tree.root(), Some(leaf_hash.as_str()));
    }

    #[test]
    fn test_merkle_tree_two_leaves() {
        let cids = vec!["cid-A".to_string(), "cid-B".to_string()];
        let tree = MerkleTree::build(&cids);
        assert_eq!(tree.leaves.len(), 2);
        // nodes = [hash(A), hash(B), hash(hash(A)+hash(B))]
        assert_eq!(tree.nodes.len(), 3);
        let root = tree.root().expect("root must exist");
        assert_eq!(root.len(), 16); // 16-char hex
    }

    #[test]
    fn test_merkle_tree_three_leaves() {
        let cids: Vec<String> = ["c1", "c2", "c3"].iter().map(|s| s.to_string()).collect();
        let tree = MerkleTree::build(&cids);
        // Level 0: 3 leaves → Level 1: 2 nodes (c3 is duplicated) → Level 2: 1 root
        // Total nodes = 3 + 2 + 1 = 6
        assert_eq!(tree.nodes.len(), 6);
        assert!(tree.root().is_some());
    }

    #[test]
    fn test_merkle_tree_four_leaves() {
        let cids: Vec<String> = ["c1", "c2", "c3", "c4"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let tree = MerkleTree::build(&cids);
        // Level 0: 4 leaves → Level 1: 2 nodes → Level 2: 1 root
        // Total = 4 + 2 + 1 = 7
        assert_eq!(tree.nodes.len(), 7);
    }

    #[test]
    fn test_merkle_tree_empty() {
        let tree = MerkleTree::build(&[]);
        assert!(tree.root().is_none());
        assert!(tree.nodes.is_empty());
    }

    // -----------------------------------------------------------------------
    // MerkleTree::root determinism
    // -----------------------------------------------------------------------

    #[test]
    fn test_merkle_root_deterministic() {
        let cids: Vec<String> = ["x", "y", "z"].iter().map(|s| s.to_string()).collect();
        let t1 = MerkleTree::build(&cids);
        let t2 = MerkleTree::build(&cids);
        assert_eq!(t1.root(), t2.root());
    }

    // -----------------------------------------------------------------------
    // proof_for and verify_proof
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_for_correct_length_two_leaves() {
        let cids = vec!["cid-L".to_string(), "cid-R".to_string()];
        let tree = MerkleTree::build(&cids);
        let proof = tree.proof_for(0).expect("proof must exist");
        // A tree with 2 leaves has depth 1; proof length should be 1.
        assert_eq!(proof.len(), 1);
    }

    #[test]
    fn test_proof_for_correct_length_four_leaves() {
        let cids: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let tree = MerkleTree::build(&cids);
        // Depth of a balanced 4-leaf tree is 2.
        for idx in 0..4usize {
            let proof = tree.proof_for(idx).expect("proof must exist");
            assert_eq!(proof.len(), 2, "leaf {idx} proof length mismatch");
        }
    }

    #[test]
    fn test_verify_proof_valid() {
        let cids: Vec<String> = ["alpha", "beta", "gamma", "delta"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let tree = MerkleTree::build(&cids);
        let root = tree.root().expect("root").to_string();

        for (idx, cid) in cids.iter().enumerate() {
            let proof = tree.proof_for(idx).expect("proof");
            assert!(
                MerkleTree::verify_proof(cid, idx, &proof, &root),
                "valid proof failed for index {idx}"
            );
        }
    }

    #[test]
    fn test_verify_proof_tampered_leaf() {
        let cids: Vec<String> = ["one", "two", "three", "four"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let tree = MerkleTree::build(&cids);
        let root = tree.root().expect("root").to_string();
        let proof = tree.proof_for(0).expect("proof");

        // Use a different leaf than the one at index 0.
        assert!(
            !MerkleTree::verify_proof("tampered", 0, &proof, &root),
            "tampered proof must fail"
        );
    }

    #[test]
    fn test_proof_for_out_of_range() {
        let cids = vec!["only-one".to_string()];
        let tree = MerkleTree::build(&cids);
        assert!(tree.proof_for(5).is_none());
    }

    // -----------------------------------------------------------------------
    // ManifestDiff
    // -----------------------------------------------------------------------

    #[test]
    fn test_manifest_diff_added_removed() {
        let old_entries = vec![
            entry("shared.txt", "cid-shared", 100, 0, true),
            entry("old-only.txt", "cid-old", 50, 0, true),
        ];
        let new_entries = vec![
            entry("shared.txt", "cid-shared", 100, 0, true),
            entry("new-only.txt", "cid-new", 75, 0, true),
        ];
        let old = ContentManifest::new(old_entries);
        let new = ContentManifest::new(new_entries);
        let diff = ManifestDiff::diff(&old, &new);

        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.added[0].cid, "cid-new");
        assert_eq!(diff.removed.len(), 1);
        assert_eq!(diff.removed[0].cid, "cid-old");
    }

    #[test]
    fn test_manifest_diff_empty_when_identical() {
        let m1 = ContentManifest::new(sample_entries());
        let m2 = ContentManifest::new(sample_entries());
        let diff = ManifestDiff::diff(&m1, &m2);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_manifest_diff_all_added() {
        let empty = ContentManifest::new(vec![]);
        let full = ContentManifest::new(sample_entries());
        let diff = ManifestDiff::diff(&empty, &full);
        assert_eq!(diff.added.len(), 3);
        assert!(diff.removed.is_empty());
    }

    #[test]
    fn test_manifest_diff_all_removed() {
        let full = ContentManifest::new(sample_entries());
        let empty = ContentManifest::new(vec![]);
        let diff = ManifestDiff::diff(&full, &empty);
        assert!(diff.added.is_empty());
        assert_eq!(diff.removed.len(), 3);
    }
}
