//! Knowledge Base Federation
//!
//! Provides functions for merging, exporting, and importing knowledge bases
//! across the IPFRS network using content-addressed IPLD blocks.
//!
//! # Overview
//!
//! - [`merge_knowledge_bases`] – merge two KBs, deduplicating by content hash.
//! - [`export_kb_as_cid`] – serialize a store's KB as IPLD, return the root CID.
//! - [`import_remote_kb`] – fetch a KB snapshot by CID and merge it into a local store.

use crate::ipld_codec::{
    block_to_kb, fact_ipld_to_predicate, kb_to_block, predicate_to_fact_ipld, rule_cid,
    rule_ipld_to_rule, rule_to_rule_ipld, KnowledgeBaseIpld,
};
use crate::ir::{KnowledgeBase, Predicate, Rule};
use ipfrs_core::{Block, Cid, Result};
use ipfrs_storage::traits::BlockStore;
use serde::{Deserialize, Serialize};

// ─── Public types ────────────────────────────────────────────────────────────

/// Summary of changes produced by merging two knowledge bases.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KbMergeDiff {
    /// Number of facts added from the remote KB
    pub facts_added: usize,
    /// Number of facts skipped because they already existed (by content hash)
    pub facts_skipped_duplicate: usize,
    /// Number of rules added from the remote KB
    pub rules_added: usize,
    /// Number of rules skipped because they already existed (by content hash)
    pub rules_skipped_duplicate: usize,
    /// List of predicate conflicts (same predicate/arity, different bodies)
    pub conflicts: Vec<KbConflict>,
}

/// A conflict detected when merging two knowledge bases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KbConflict {
    /// Same predicate name / arity but different rule body.
    /// Both versions are retained with a version prefix to distinguish them.
    RuleBodyConflict {
        /// The predicate name involved in the conflict
        predicate: String,
        /// Display form of the local rule
        local_rule: String,
        /// Display form of the remote rule
        remote_rule: String,
    },
}

// ─── Core merge logic ─────────────────────────────────────────────────────────

/// Compute the FNV-1a hash of a serialized predicate (used for deduplication).
fn fact_content_hash(fact: &Predicate) -> u64 {
    match predicate_to_fact_ipld(fact) {
        Ok(ipld) => match serde_json::to_vec(&ipld) {
            Ok(bytes) => fnv1a(&bytes),
            Err(_) => {
                // Fallback: hash the debug representation
                fnv1a(format!("{:?}", fact).as_bytes())
            }
        },
        Err(_) => fnv1a(format!("{:?}", fact).as_bytes()),
    }
}

fn fnv1a(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;
    let mut hash = FNV_OFFSET;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Compute the content hash of a rule via its IPLD representation.
fn rule_content_hash(rule: &Rule) -> Option<u64> {
    let ipld = rule_to_rule_ipld(rule).ok()?;
    let bytes = serde_json::to_vec(&ipld).ok()?;
    Some(fnv1a(&bytes))
}

/// Merge two knowledge bases, deduplicating facts and rules by content hash.
///
/// Rules with the same head predicate name and arity but a different body
/// produce a [`KbConflict::RuleBodyConflict`] entry.  Both versions are kept
/// in the merged output (the local version is already present; the remote
/// version is renamed with a `"remote_"` prefix on the head predicate).
///
/// # Returns
/// A tuple of `(merged_kb, diff_summary)`.
pub fn merge_knowledge_bases(
    local: &KnowledgeBase,
    remote: &KnowledgeBase,
) -> (KnowledgeBase, KbMergeDiff) {
    let mut merged = local.clone();
    let mut diff = KbMergeDiff::default();

    // ── Fact deduplication ───────────────────────────────────────────────────
    // Build a set of content hashes for existing local facts.
    let local_fact_hashes: std::collections::HashSet<u64> =
        local.facts.iter().map(fact_content_hash).collect();

    for remote_fact in &remote.facts {
        let h = fact_content_hash(remote_fact);
        if local_fact_hashes.contains(&h) {
            diff.facts_skipped_duplicate += 1;
        } else {
            merged.add_fact(remote_fact.clone());
            diff.facts_added += 1;
        }
    }

    // ── Rule deduplication & conflict detection ──────────────────────────────
    // Build a map from content hash → rule for fast duplicate lookup.
    let local_rule_hashes: std::collections::HashMap<u64, &Rule> = local
        .rules
        .iter()
        .filter_map(|r| rule_content_hash(r).map(|h| (h, r)))
        .collect();

    // Build a map from (head_name, arity) → rule body display for conflict detection.
    let local_signature_map: std::collections::HashMap<(String, usize), String> = local
        .rules
        .iter()
        .map(|r| {
            let key = (r.head.name.clone(), r.head.args.len());
            let body_repr = rule_body_repr(r);
            (key, body_repr)
        })
        .collect();

    for remote_rule in &remote.rules {
        let remote_hash = match rule_content_hash(remote_rule) {
            Some(h) => h,
            None => {
                // Cannot compute hash; add unconditionally
                merged.add_rule(remote_rule.clone());
                diff.rules_added += 1;
                continue;
            }
        };

        if local_rule_hashes.contains_key(&remote_hash) {
            diff.rules_skipped_duplicate += 1;
            continue;
        }

        // Check for body conflict: same predicate/arity but different body.
        let sig = (remote_rule.head.name.clone(), remote_rule.head.args.len());
        if let Some(local_body_repr) = local_signature_map.get(&sig) {
            let remote_body_repr = rule_body_repr(remote_rule);
            if *local_body_repr != remote_body_repr {
                // Conflict – record it and add the remote rule with a versioned prefix
                diff.conflicts.push(KbConflict::RuleBodyConflict {
                    predicate: remote_rule.head.name.clone(),
                    local_rule: format!(
                        "{}/{}: {}",
                        remote_rule.head.name,
                        remote_rule.head.args.len(),
                        local_body_repr
                    ),
                    remote_rule: format!(
                        "{}/{}: {}",
                        remote_rule.head.name,
                        remote_rule.head.args.len(),
                        remote_body_repr
                    ),
                });

                // Keep both: create a renamed copy of the remote rule so it does
                // not shadow the local one.
                let mut renamed = remote_rule.clone();
                renamed.head.name = format!("remote_{}", remote_rule.head.name);
                merged.add_rule(renamed);
                diff.rules_added += 1;
                continue;
            }
        }

        merged.add_rule(remote_rule.clone());
        diff.rules_added += 1;
    }

    (merged, diff)
}

/// Produce a compact string representation of a rule's body for conflict detection.
fn rule_body_repr(rule: &Rule) -> String {
    rule.body
        .iter()
        .map(|p| format!("{}({})", p.name, p.args.len()))
        .collect::<Vec<_>>()
        .join(",")
}

// ─── IPLD export / import ─────────────────────────────────────────────────────

/// Export the current knowledge base of `store` as an IPLD block DAG.
///
/// Each rule is stored as a separate block (deduplication-friendly). Facts are
/// inlined in the root `KnowledgeBaseIpld` block.
///
/// Returns the CID of the root block.
pub async fn export_kb_as_cid<S: BlockStore>(
    store: &crate::storage::TensorLogicStore<S>,
    block_store: &dyn BlockStore,
) -> Result<Cid> {
    // Snapshot the KB
    let kb = store.snapshot_kb()?;

    // Store each rule individually
    let mut rule_cids: Vec<String> = Vec::with_capacity(kb.rules.len());
    for rule in &kb.rules {
        let rule_ipld = rule_to_rule_ipld(rule)?;
        let cid = rule_cid(&rule_ipld)?;

        // Build the block manually so we can store it in the caller's block_store
        let json_bytes = serde_json::to_vec(&rule_ipld)
            .map_err(|e| ipfrs_core::Error::Serialization(format!("rule IPLD: {}", e)))?;
        let block = Block::from_parts(cid, bytes::Bytes::from(json_bytes));
        block_store.put(&block).await?;

        rule_cids.push(cid.to_string());
    }

    // Inline facts
    let fact_iplds = kb
        .facts
        .iter()
        .map(predicate_to_fact_ipld)
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let kb_ipld = KnowledgeBaseIpld {
        rules: rule_cids,
        facts: fact_iplds,
        version: "1.0.0".to_string(),
    };

    let root_block = kb_to_block(&kb_ipld)?;
    let root_cid = *root_block.cid();
    block_store.put(&root_block).await?;

    Ok(root_cid)
}

/// Import a KB snapshot identified by `remote_cid` into `store`, merging with
/// the existing local KB.
///
/// The function fetches the root `KnowledgeBaseIpld` block and all linked rule
/// blocks from `block_store`, reconstructs the remote [`KnowledgeBase`], and
/// then calls [`merge_knowledge_bases`] to produce the merged result.
///
/// # Errors
/// Returns an error if any block is missing from `block_store` or if decoding
/// fails.
pub async fn import_remote_kb<S: BlockStore>(
    remote_cid: &Cid,
    store: &crate::storage::TensorLogicStore<S>,
    block_store: &dyn BlockStore,
) -> Result<KbMergeDiff> {
    // Fetch root block
    let root_block = block_store
        .get(remote_cid)
        .await?
        .ok_or_else(|| ipfrs_core::Error::BlockNotFound(remote_cid.to_string()))?;

    let kb_ipld = block_to_kb(&root_block)?;

    // Reconstruct remote KB
    let mut remote_kb = KnowledgeBase::new();

    // Load rules
    for rule_cid_str in &kb_ipld.rules {
        let rule_cid: Cid = rule_cid_str
            .parse()
            .map_err(|e| ipfrs_core::Error::Cid(format!("invalid CID {}: {}", rule_cid_str, e)))?;

        let rule_block = block_store
            .get(&rule_cid)
            .await?
            .ok_or_else(|| ipfrs_core::Error::BlockNotFound(rule_cid.to_string()))?;

        let rule_ipld = crate::ipld_codec::block_to_rule(&rule_block)?;
        let rule = rule_ipld_to_rule(&rule_ipld)?;
        remote_kb.add_rule(rule);
    }

    // Load facts from inline ipld
    for fact_ipld in &kb_ipld.facts {
        let predicate = fact_ipld_to_predicate(fact_ipld)?;
        remote_kb.add_fact(predicate);
    }

    // Snapshot local KB
    let local_kb = store.snapshot_kb()?;

    // Merge
    let (merged_kb, diff) = merge_knowledge_bases(&local_kb, &remote_kb);

    // Apply merged KB back into store
    for fact in &merged_kb.facts {
        // Only add facts that are new (not already in local)
        let local_fact_hashes: std::collections::HashSet<u64> =
            local_kb.facts.iter().map(fact_content_hash).collect();
        if !local_fact_hashes.contains(&fact_content_hash(fact)) {
            store.add_fact(fact.clone())?;
        }
    }

    for rule in &merged_kb.rules {
        let local_rule_hashes: std::collections::HashSet<u64> = local_kb
            .rules
            .iter()
            .filter_map(rule_content_hash)
            .collect();
        if let Some(h) = rule_content_hash(rule) {
            if !local_rule_hashes.contains(&h) {
                store.add_rule(rule.clone())?;
            }
        }
    }

    Ok(diff)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Constant, Predicate, Rule, Term};
    use crate::storage::TensorLogicStore;
    use ipfrs_storage::{BlockStoreConfig, SledBlockStore};
    use std::sync::Arc;

    fn make_store(suffix: &str) -> Arc<TensorLogicStore<SledBlockStore>> {
        let path = std::env::temp_dir().join(format!("ipfrs-test-kb-fed-{}", suffix));
        let _ = std::fs::remove_dir_all(&path);
        let config = BlockStoreConfig {
            path,
            cache_size: 32 * 1024 * 1024,
        };
        let sled = Arc::new(SledBlockStore::new(config).expect("sled store"));
        Arc::new(TensorLogicStore::new(sled).expect("tensorlogic store"))
    }

    fn make_sled_store(suffix: &str) -> Arc<SledBlockStore> {
        let path = std::env::temp_dir().join(format!("ipfrs-test-sled-fed-{}", suffix));
        let _ = std::fs::remove_dir_all(&path);
        let config = BlockStoreConfig {
            path,
            cache_size: 16 * 1024 * 1024,
        };
        Arc::new(SledBlockStore::new(config).expect("sled store"))
    }

    fn parent_fact(a: &str, b: &str) -> Predicate {
        Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String(a.to_string())),
                Term::Const(Constant::String(b.to_string())),
            ],
        )
    }

    fn ancestor_rule_base() -> Rule {
        Rule::new(
            Predicate::new(
                "ancestor".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            vec![Predicate::new(
                "parent".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            )],
        )
    }

    fn sibling_rule() -> Rule {
        Rule::new(
            Predicate::new(
                "sibling".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            vec![
                Predicate::new(
                    "parent".to_string(),
                    vec![Term::Var("Z".to_string()), Term::Var("X".to_string())],
                ),
                Predicate::new(
                    "parent".to_string(),
                    vec![Term::Var("Z".to_string()), Term::Var("Y".to_string())],
                ),
            ],
        )
    }

    #[test]
    fn test_merge_disjoint_kbs() {
        let mut local_kb = KnowledgeBase::new();
        local_kb.add_fact(parent_fact("alice", "bob"));
        local_kb.add_rule(ancestor_rule_base());

        let mut remote_kb = KnowledgeBase::new();
        remote_kb.add_fact(parent_fact("bob", "charlie"));
        remote_kb.add_rule(sibling_rule());

        let (merged, diff) = merge_knowledge_bases(&local_kb, &remote_kb);

        assert_eq!(merged.facts.len(), 2, "should have both facts");
        assert_eq!(merged.rules.len(), 2, "should have both rules");
        assert_eq!(diff.facts_added, 1);
        assert_eq!(diff.facts_skipped_duplicate, 0);
        assert_eq!(diff.rules_added, 1);
        assert_eq!(diff.rules_skipped_duplicate, 0);
        assert!(diff.conflicts.is_empty());
    }

    #[test]
    fn test_merge_duplicate_facts() {
        let mut local_kb = KnowledgeBase::new();
        local_kb.add_fact(parent_fact("alice", "bob"));

        let mut remote_kb = KnowledgeBase::new();
        // Same fact in remote
        remote_kb.add_fact(parent_fact("alice", "bob"));
        // Plus a new one
        remote_kb.add_fact(parent_fact("bob", "charlie"));

        let (merged, diff) = merge_knowledge_bases(&local_kb, &remote_kb);

        assert_eq!(merged.facts.len(), 2);
        assert_eq!(
            diff.facts_skipped_duplicate, 1,
            "duplicate should be skipped"
        );
        assert_eq!(diff.facts_added, 1);
    }

    #[test]
    fn test_merge_conflicting_rules() {
        let mut local_kb = KnowledgeBase::new();
        // ancestor(X, Y) :- parent(X, Y)
        local_kb.add_rule(ancestor_rule_base());

        let mut remote_kb = KnowledgeBase::new();
        // ancestor(X, Y) :- ancestor(X, Z), ancestor(Z, Y)  (different body)
        let conflict_rule = Rule::new(
            Predicate::new(
                "ancestor".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            vec![
                Predicate::new(
                    "ancestor".to_string(),
                    vec![Term::Var("X".to_string()), Term::Var("Z".to_string())],
                ),
                Predicate::new(
                    "ancestor".to_string(),
                    vec![Term::Var("Z".to_string()), Term::Var("Y".to_string())],
                ),
            ],
        );
        remote_kb.add_rule(conflict_rule);

        let (_merged, diff) = merge_knowledge_bases(&local_kb, &remote_kb);

        assert_eq!(diff.conflicts.len(), 1, "should have one conflict");
        match &diff.conflicts[0] {
            KbConflict::RuleBodyConflict { predicate, .. } => {
                assert_eq!(predicate, "ancestor");
            }
        }
        // Both rules are kept: local + remote renamed
        assert_eq!(diff.rules_added, 1);
    }

    #[tokio::test]
    async fn test_export_import_roundtrip() {
        let local_store = make_store("roundtrip-local");
        let block_store = make_sled_store("roundtrip-blocks");

        // Populate local store
        local_store
            .add_fact(parent_fact("alice", "bob"))
            .expect("add fact");
        local_store
            .add_rule(ancestor_rule_base())
            .expect("add rule");

        // Export
        let cid = export_kb_as_cid(&local_store, block_store.as_ref())
            .await
            .expect("export");

        // Import into a fresh store
        let fresh_store = make_store("roundtrip-fresh");
        let diff = import_remote_kb(&cid, &fresh_store, block_store.as_ref())
            .await
            .expect("import");

        assert_eq!(diff.facts_added, 1, "should have imported 1 fact");
        assert_eq!(diff.rules_added, 1, "should have imported 1 rule");

        let kb = fresh_store.snapshot_kb().expect("snapshot");
        assert_eq!(kb.facts.len(), 1);
        assert_eq!(kb.rules.len(), 1);
        assert_eq!(kb.facts[0].name, "parent");
        assert_eq!(kb.rules[0].head.name, "ancestor");
    }
}
