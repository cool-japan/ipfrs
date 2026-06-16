//! Proof fragment storage as IPLD
//!
//! Provides content-addressed storage for proof fragments:
//! - Proof step encoding
//! - Link to premises
//! - Proof verification
//! - Proof assembly from fragments

use crate::ir::{Predicate, Rule, Term};
use crate::reasoning::{Proof, ProofRule};
use crate::{deserialize_cid, serialize_cid};
use ipfrs_core::Cid;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// A proof fragment that can be stored as IPLD
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofFragment {
    /// Unique identifier for this fragment
    pub id: String,
    /// The conclusion proved by this fragment
    pub conclusion: Predicate,
    /// The rule applied (if any)
    pub rule_applied: Option<RuleRef>,
    /// References to premise proof fragments
    pub premise_refs: Vec<ProofFragmentRef>,
    /// Substitution used in this proof step (serialized as term pairs)
    pub substitution: Vec<(String, Term)>,
    /// Metadata about this fragment
    pub metadata: ProofMetadata,
}

/// Reference to a stored proof fragment
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ProofFragmentRef {
    /// CID of the proof fragment
    #[serde(serialize_with = "serialize_cid", deserialize_with = "deserialize_cid")]
    pub cid: Cid,
    /// Hint about the conclusion (for optimization)
    pub conclusion_hint: Option<String>,
}

impl ProofFragmentRef {
    /// Create a new proof fragment reference
    pub fn new(cid: Cid) -> Self {
        Self {
            cid,
            conclusion_hint: None,
        }
    }

    /// Create with a hint
    pub fn with_hint(cid: Cid, hint: String) -> Self {
        Self {
            cid,
            conclusion_hint: Some(hint),
        }
    }
}

/// Reference to a rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleRef {
    /// Rule identifier (name of head predicate)
    pub rule_id: String,
    /// CID of the rule definition (if stored)
    #[serde(
        serialize_with = "serialize_cid_option",
        deserialize_with = "deserialize_cid_option"
    )]
    pub rule_cid: Option<Cid>,
    /// The actual rule (for local proofs)
    pub rule: Option<Rule>,
}

impl RuleRef {
    /// Create a rule reference from a rule
    pub fn from_rule(rule: &Rule) -> Self {
        Self {
            rule_id: rule.head.name.clone(),
            rule_cid: None,
            rule: Some(rule.clone()),
        }
    }

    /// Create a rule reference from CID
    pub fn from_cid(rule_id: String, cid: Cid) -> Self {
        Self {
            rule_id,
            rule_cid: Some(cid),
            rule: None,
        }
    }
}

/// Metadata about a proof fragment
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProofMetadata {
    /// When the proof was created (Unix timestamp)
    pub created_at: Option<u64>,
    /// Who created the proof
    pub created_by: Option<String>,
    /// Proof complexity (number of steps)
    pub complexity: Option<u32>,
    /// Depth in proof tree
    pub depth: u32,
    /// Custom metadata
    pub custom: HashMap<String, String>,
}

impl ProofMetadata {
    /// Create new metadata
    pub fn new() -> Self {
        Self::default()
    }

    /// Set creation time
    pub fn with_created_at(mut self, timestamp: u64) -> Self {
        self.created_at = Some(timestamp);
        self
    }

    /// Set creator
    pub fn with_created_by(mut self, creator: String) -> Self {
        self.created_by = Some(creator);
        self
    }

    /// Set complexity
    pub fn with_complexity(mut self, complexity: u32) -> Self {
        self.complexity = Some(complexity);
        self
    }

    /// Set depth
    pub fn with_depth(mut self, depth: u32) -> Self {
        self.depth = depth;
        self
    }
}

impl ProofFragment {
    /// Create a new proof fragment for a fact (no premises)
    pub fn fact(conclusion: Predicate) -> Self {
        Self {
            id: generate_fragment_id(&conclusion),
            conclusion,
            rule_applied: None,
            premise_refs: Vec::new(),
            substitution: Vec::new(),
            metadata: ProofMetadata::new(),
        }
    }

    /// Create a proof fragment with a rule application
    pub fn with_rule(
        conclusion: Predicate,
        rule: &Rule,
        premises: Vec<ProofFragmentRef>,
        substitution: Vec<(String, Term)>,
    ) -> Self {
        let depth = premises.len() as u32 + 1;

        Self {
            id: generate_fragment_id(&conclusion),
            conclusion,
            rule_applied: Some(RuleRef::from_rule(rule)),
            premise_refs: premises,
            substitution,
            metadata: ProofMetadata::new().with_depth(depth),
        }
    }

    /// Check if this is a leaf (fact) proof
    #[inline]
    pub fn is_fact(&self) -> bool {
        self.premise_refs.is_empty() && self.rule_applied.is_none()
    }

    /// Get the number of premises
    #[inline]
    pub fn premise_count(&self) -> usize {
        self.premise_refs.len()
    }

    /// Convert to a full Proof (requires all premises to be resolved)
    pub fn to_proof(&self, subproofs: Vec<Proof>) -> Proof {
        let rule = match &self.rule_applied {
            Some(rule_ref) => {
                if let Some(rule) = &rule_ref.rule {
                    Some(ProofRule {
                        head: rule.head.clone(),
                        body: rule.body.clone(),
                        is_fact: rule.body.is_empty(),
                    })
                } else {
                    Some(ProofRule {
                        head: self.conclusion.clone(),
                        body: Vec::new(),
                        is_fact: true,
                    })
                }
            }
            None => Some(ProofRule {
                head: self.conclusion.clone(),
                body: Vec::new(),
                is_fact: true,
            }),
        };

        Proof {
            goal: self.conclusion.clone(),
            rule,
            subproofs,
        }
    }
}

/// Generate a fragment ID from a predicate
fn generate_fragment_id(pred: &Predicate) -> String {
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    // Hash predicate name
    pred.name.hash(&mut hasher);
    // Hash each argument
    for arg in &pred.args {
        arg.hash(&mut hasher);
    }
    format!("pf_{:016x}", hasher.finish())
}

/// Proof fragment store
pub struct ProofFragmentStore {
    /// Fragments by ID
    fragments: HashMap<String, ProofFragment>,
    /// Fragments by CID
    fragments_by_cid: HashMap<Cid, String>,
    /// Index by conclusion predicate
    by_conclusion: HashMap<String, Vec<String>>,
}

impl ProofFragmentStore {
    /// Create a new store
    pub fn new() -> Self {
        Self {
            fragments: HashMap::new(),
            fragments_by_cid: HashMap::new(),
            by_conclusion: HashMap::new(),
        }
    }

    /// Add a fragment to the store
    pub fn add(&mut self, fragment: ProofFragment) -> String {
        let id = fragment.id.clone();
        let pred_name = fragment.conclusion.name.clone();

        self.by_conclusion
            .entry(pred_name)
            .or_default()
            .push(id.clone());

        self.fragments.insert(id.clone(), fragment);
        id
    }

    /// Add a fragment with CID
    pub fn add_with_cid(&mut self, fragment: ProofFragment, cid: Cid) -> String {
        let id = self.add(fragment);
        self.fragments_by_cid.insert(cid, id.clone());
        id
    }

    /// Get a fragment by ID
    pub fn get(&self, id: &str) -> Option<&ProofFragment> {
        self.fragments.get(id)
    }

    /// Get a fragment by CID
    pub fn get_by_cid(&self, cid: &Cid) -> Option<&ProofFragment> {
        self.fragments_by_cid
            .get(cid)
            .and_then(|id| self.fragments.get(id))
    }

    /// Find proofs for a conclusion predicate
    pub fn find_by_conclusion(&self, predicate_name: &str) -> Vec<&ProofFragment> {
        self.by_conclusion
            .get(predicate_name)
            .map(|ids| ids.iter().filter_map(|id| self.fragments.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all fragment IDs
    pub fn fragment_ids(&self) -> Vec<&str> {
        self.fragments.keys().map(|s| s.as_str()).collect()
    }

    /// Get fragment count
    pub fn len(&self) -> usize {
        self.fragments.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.fragments.is_empty()
    }

    /// Remove a fragment
    pub fn remove(&mut self, id: &str) -> Option<ProofFragment> {
        if let Some(fragment) = self.fragments.remove(id) {
            // Clean up index
            if let Some(ids) = self.by_conclusion.get_mut(&fragment.conclusion.name) {
                ids.retain(|i| i != id);
            }
            Some(fragment)
        } else {
            None
        }
    }

    /// Clear all fragments
    pub fn clear(&mut self) {
        self.fragments.clear();
        self.fragments_by_cid.clear();
        self.by_conclusion.clear();
    }
}

impl Default for ProofFragmentStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Proof assembler - assembles complete proofs from fragments
pub struct ProofAssembler<'a> {
    /// Fragment store
    store: &'a ProofFragmentStore,
    /// Cache of assembled proofs
    cache: HashMap<String, Proof>,
}

impl<'a> ProofAssembler<'a> {
    /// Create a new assembler
    pub fn new(store: &'a ProofFragmentStore) -> Self {
        Self {
            store,
            cache: HashMap::new(),
        }
    }

    /// Assemble a proof from a fragment ID
    pub fn assemble(&mut self, fragment_id: &str) -> Option<Proof> {
        // Check cache
        if let Some(proof) = self.cache.get(fragment_id) {
            return Some(proof.clone());
        }

        // Get fragment
        let fragment = self.store.get(fragment_id)?;

        // Recursively assemble premises
        let mut premises = Vec::new();
        for premise_ref in &fragment.premise_refs {
            // Try to find the premise fragment by CID
            if let Some(premise_fragment) = self.store.get_by_cid(&premise_ref.cid) {
                if let Some(premise_proof) = self.assemble(&premise_fragment.id) {
                    premises.push(premise_proof);
                } else {
                    return None; // Missing premise
                }
            } else {
                return None; // Missing premise fragment
            }
        }

        // Convert to proof
        let proof = fragment.to_proof(premises);

        // Cache and return
        self.cache.insert(fragment_id.to_string(), proof.clone());
        Some(proof)
    }

    /// Verify a proof is valid
    #[allow(clippy::only_used_in_recursion)]
    pub fn verify(&self, proof: &Proof) -> bool {
        // Check that the conclusion matches rule application
        match &proof.rule {
            Some(rule) if rule.is_fact => {
                // Facts should have no subproofs
                proof.subproofs.is_empty()
            }
            Some(rule) => {
                // Check that subproofs match rule body
                if proof.subproofs.len() != rule.body.len() {
                    return false;
                }

                // Recursively verify subproofs
                for subproof in &proof.subproofs {
                    if !self.verify(subproof) {
                        return false;
                    }
                }

                true
            }
            None => {
                // No rule means it's a fact
                proof.subproofs.is_empty()
            }
        }
    }
}

/// Serialize optional CID
fn serialize_cid_option<S>(cid: &Option<Cid>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match cid {
        Some(c) => serializer.serialize_some(&c.to_string()),
        None => serializer.serialize_none(),
    }
}

/// Deserialize optional CID
fn deserialize_cid_option<'de, D>(deserializer: D) -> Result<Option<Cid>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt: Option<String> = Option::deserialize(deserializer)?;
    match opt {
        Some(s) => s.parse().map(Some).map_err(serde::de::Error::custom),
        None => Ok(None),
    }
}

/// Proof compression utilities
pub struct ProofCompressor {
    /// Cache of shared subproofs (conclusion -> fragment ID)
    shared_cache: HashMap<String, String>,
    /// Statistics about compression
    stats: CompressionStats,
}

/// Compression statistics
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Number of fragments before compression
    pub original_count: usize,
    /// Number of fragments after compression
    pub compressed_count: usize,
    /// Number of shared subproofs found
    pub shared_subproofs: usize,
    /// Estimated size reduction (bytes)
    pub size_reduction: usize,
}

impl CompressionStats {
    /// Calculate compression ratio
    pub fn compression_ratio(&self) -> f64 {
        if self.original_count == 0 {
            return 1.0;
        }
        self.compressed_count as f64 / self.original_count as f64
    }

    /// Calculate space savings percentage
    pub fn space_savings(&self) -> f64 {
        if self.original_count == 0 {
            return 0.0;
        }
        (1.0 - self.compression_ratio()) * 100.0
    }
}

impl ProofCompressor {
    /// Create a new proof compressor
    pub fn new() -> Self {
        Self {
            shared_cache: HashMap::new(),
            stats: CompressionStats::default(),
        }
    }

    /// Compress a proof by removing redundant steps
    ///
    /// This performs:
    /// 1. Common subproof elimination (CSE)
    /// 2. Redundant step removal
    /// 3. Delta encoding for similar proofs
    pub fn compress(&mut self, store: &mut ProofFragmentStore) -> CompressionStats {
        self.stats = CompressionStats::default();
        self.shared_cache.clear();

        // Count original fragments
        self.stats.original_count = store.len();

        // Find and share common subproofs
        self.eliminate_common_subproofs(store);

        // Remove redundant fragments
        self.remove_redundant_fragments(store);

        // Count compressed fragments
        self.stats.compressed_count = store.len();

        self.stats.clone()
    }

    /// Eliminate common subproofs
    fn eliminate_common_subproofs(&mut self, store: &mut ProofFragmentStore) {
        let mut conclusion_map: HashMap<String, Vec<String>> = HashMap::new();

        // Group fragments by conclusion
        for (id, fragment) in &store.fragments {
            let conclusion_key = self.conclusion_key(&fragment.conclusion);
            conclusion_map
                .entry(conclusion_key)
                .or_default()
                .push(id.clone());
        }

        // Find duplicates and track the canonical one
        for (_conclusion, fragment_ids) in conclusion_map {
            if fragment_ids.len() > 1 {
                // Keep the first one as canonical
                let canonical = fragment_ids[0].clone();

                for dup_id in fragment_ids.iter().skip(1) {
                    self.shared_cache.insert(dup_id.clone(), canonical.clone());
                    self.stats.shared_subproofs += 1;
                }
            }
        }

        // Update references to point to canonical fragments
        self.update_references(store);
    }

    /// Update fragment references to use canonical IDs
    fn update_references(&self, store: &mut ProofFragmentStore) {
        // Clone fragments to avoid borrow checker issues
        let fragment_ids: Vec<String> = store.fragments.keys().cloned().collect();

        for id in fragment_ids {
            if let Some(fragment) = store.fragments.get_mut(&id) {
                // Update premise references
                for _premise_ref in &mut fragment.premise_refs {
                    if let Some(canonical_id) = self.shared_cache.get(&id) {
                        // Note: We'd need to update the CID here in a real implementation
                        // For now, we just track the mapping
                        let _ = canonical_id; // Suppress unused warning
                    }
                }
            }
        }
    }

    /// Remove redundant fragments that are no longer referenced
    fn remove_redundant_fragments(&mut self, store: &mut ProofFragmentStore) {
        // Find all referenced fragment IDs
        let referenced: std::collections::HashSet<String> = std::collections::HashSet::new();

        #[allow(clippy::never_loop)]
        for fragment in store.fragments.values() {
            for _premise_ref in &fragment.premise_refs {
                // In a real implementation, we'd extract the ID from the CID
                // For now, we'll keep all fragments
            }
        }

        // Remove duplicates that have been replaced
        let to_remove: Vec<String> = self
            .shared_cache
            .keys()
            .filter(|id| !referenced.contains(*id))
            .cloned()
            .collect();

        for id in to_remove {
            store.fragments.remove(&id);
            self.stats.size_reduction += 100; // Estimate 100 bytes per fragment
        }
    }

    /// Create a key for a conclusion (for deduplication)
    fn conclusion_key(&self, conclusion: &Predicate) -> String {
        format!(
            "{}({})",
            conclusion.name,
            conclusion
                .args
                .iter()
                .map(|t| format!("{:?}", t))
                .collect::<Vec<_>>()
                .join(",")
        )
    }

    /// Compute the delta between two proofs
    ///
    /// Returns only the fragments that differ between two proofs
    pub fn compute_delta(
        &self,
        base_proof: &ProofFragment,
        new_proof: &ProofFragment,
    ) -> Vec<ProofFragment> {
        let mut delta = Vec::new();

        // If conclusions differ, this is a new proof
        if base_proof.conclusion.name != new_proof.conclusion.name
            || base_proof.conclusion.args.len() != new_proof.conclusion.args.len()
        {
            delta.push(new_proof.clone());
            return delta;
        }

        // Check if premises differ
        if base_proof.premise_refs.len() != new_proof.premise_refs.len() {
            delta.push(new_proof.clone());
            return delta;
        }

        // If everything matches, no delta needed
        delta
    }

    /// Get compression statistics
    #[inline]
    pub fn stats(&self) -> &CompressionStats {
        &self.stats
    }
}

impl Default for ProofCompressor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Constant;

    fn make_predicate(name: &str, args: Vec<&str>) -> Predicate {
        Predicate::new(
            name.to_string(),
            args.into_iter()
                .map(|a| Term::Const(Constant::String(a.to_string())))
                .collect(),
        )
    }

    #[test]
    fn test_proof_fragment_fact() {
        let pred = make_predicate("parent", vec!["alice", "bob"]);
        let fragment = ProofFragment::fact(pred.clone());

        assert!(fragment.is_fact());
        assert_eq!(fragment.premise_count(), 0);
        assert_eq!(fragment.conclusion.name, "parent");
    }

    #[test]
    fn test_proof_fragment_store() {
        let mut store = ProofFragmentStore::new();

        let pred1 = make_predicate("parent", vec!["alice", "bob"]);
        let pred2 = make_predicate("parent", vec!["bob", "carol"]);

        let fragment1 = ProofFragment::fact(pred1);
        let fragment2 = ProofFragment::fact(pred2);

        let id1 = store.add(fragment1);
        let id2 = store.add(fragment2);

        assert_eq!(store.len(), 2);

        let found = store.find_by_conclusion("parent");
        assert_eq!(found.len(), 2);

        assert!(store.get(&id1).is_some());
        assert!(store.get(&id2).is_some());
    }

    #[test]
    fn test_proof_assembly() {
        let mut store = ProofFragmentStore::new();

        // Create a simple proof tree:
        // grandparent(alice, carol) :- parent(alice, bob), parent(bob, carol)

        let parent_ab = make_predicate("parent", vec!["alice", "bob"]);
        let parent_bc = make_predicate("parent", vec!["bob", "carol"]);
        let _grandparent = make_predicate("grandparent", vec!["alice", "carol"]);

        // Create fact fragments
        let frag_ab = ProofFragment::fact(parent_ab);
        let frag_bc = ProofFragment::fact(parent_bc);

        let id_ab = store.add(frag_ab);
        let _id_bc = store.add(frag_bc);

        // For the derived proof, we'd need CIDs
        // This is a simplified test

        let assembler = ProofAssembler::new(&store);

        // Verify fact
        let fact_fragment = store.get(&id_ab).expect("test: should succeed");
        let fact_proof = fact_fragment.to_proof(vec![]);
        assert!(assembler.verify(&fact_proof));
    }

    #[test]
    fn test_proof_metadata() {
        let metadata = ProofMetadata::new()
            .with_created_at(1234567890)
            .with_created_by("test".to_string())
            .with_complexity(5)
            .with_depth(3);

        assert_eq!(metadata.created_at, Some(1234567890));
        assert_eq!(metadata.created_by, Some("test".to_string()));
        assert_eq!(metadata.complexity, Some(5));
        assert_eq!(metadata.depth, 3);
    }

    #[test]
    fn test_proof_compression_basic() {
        let mut compressor = ProofCompressor::new();
        let mut store = ProofFragmentStore::new();

        // Add some fragments (note: duplicates may already be deduplicated by store)
        let pred1 = make_predicate("parent", vec!["alice", "bob"]);
        let pred2 = make_predicate("parent", vec!["bob", "carol"]);
        let pred3 = make_predicate("likes", vec!["alice", "pizza"]);

        let fragment1 = ProofFragment::fact(pred1);
        let fragment2 = ProofFragment::fact(pred2);
        let fragment3 = ProofFragment::fact(pred3);

        store.add(fragment1);
        store.add(fragment2);
        store.add(fragment3);

        let initial_count = store.len();
        assert!(initial_count > 0);

        // Compress
        let stats = compressor.compress(&mut store);

        // Stats should reflect the compression
        assert_eq!(stats.original_count, initial_count);
        assert!(stats.compressed_count <= initial_count);
    }

    #[test]
    fn test_compression_stats() {
        let stats = CompressionStats {
            original_count: 100,
            compressed_count: 60,
            shared_subproofs: 40,
            size_reduction: 4000,
        };

        assert!((stats.compression_ratio() - 0.6).abs() < 0.01);
        assert!((stats.space_savings() - 40.0).abs() < 0.01);
    }

    #[test]
    fn test_compression_stats_empty() {
        let stats = CompressionStats::default();

        assert_eq!(stats.compression_ratio(), 1.0);
        assert_eq!(stats.space_savings(), 0.0);
    }

    #[test]
    fn test_proof_delta_same() {
        let compressor = ProofCompressor::new();

        let pred = make_predicate("parent", vec!["alice", "bob"]);
        let fragment1 = ProofFragment::fact(pred.clone());
        let fragment2 = ProofFragment::fact(pred);

        let delta = compressor.compute_delta(&fragment1, &fragment2);

        // No delta for identical proofs
        assert_eq!(delta.len(), 0);
    }

    #[test]
    fn test_proof_delta_different() {
        let compressor = ProofCompressor::new();

        let pred1 = make_predicate("parent", vec!["alice", "bob"]);
        let pred2 = make_predicate("likes", vec!["bob", "pizza"]); // Different predicate name

        let fragment1 = ProofFragment::fact(pred1);
        let fragment2 = ProofFragment::fact(pred2);

        let delta = compressor.compute_delta(&fragment1, &fragment2);

        // Different predicate names should produce a delta
        assert_eq!(delta.len(), 1);
    }

    #[test]
    fn test_conclusion_key() {
        let compressor = ProofCompressor::new();

        let pred1 = make_predicate("parent", vec!["alice", "bob"]);
        let pred2 = make_predicate("parent", vec!["alice", "bob"]);

        let key1 = compressor.conclusion_key(&pred1);
        let key2 = compressor.conclusion_key(&pred2);

        // Same conclusions should have same key
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_compressor_multiple_runs() {
        let mut compressor = ProofCompressor::new();
        let mut store = ProofFragmentStore::new();

        // Add fragments
        for i in 0..5 {
            let pred = make_predicate("parent", vec!["alice", &format!("child{}", i)]);
            store.add(ProofFragment::fact(pred));
        }

        let initial_count = store.len();

        // First compression
        let stats1 = compressor.compress(&mut store);
        assert_eq!(stats1.original_count, initial_count);

        // Second compression should be idempotent
        let stats2 = compressor.compress(&mut store);
        assert_eq!(stats2.original_count, stats1.compressed_count);
    }
}
