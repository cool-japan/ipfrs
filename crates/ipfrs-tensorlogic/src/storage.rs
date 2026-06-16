//! Storage for TensorLogic IR
//!
//! Provides content-addressed storage for logical terms, predicates, and rules

use crate::inference_cache::InferenceCache;
use crate::ir::{KnowledgeBase, Predicate, Rule, Term};
use crate::reasoning::{InferenceEngine, Proof, Substitution};
use ipfrs_core::{Block, Cid, Result};
use ipfrs_storage::traits::BlockStore;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Extended statistics for a [`TensorLogicStore`] including CID accounting.
///
/// Produced by [`TensorLogicStore::kb_stats_with_cids`].
#[derive(Debug, Clone)]
pub struct TensorLogicStoreStats {
    /// Number of rules currently loaded in the in-memory knowledge base
    pub rule_count: usize,
    /// Number of ground facts currently loaded in the in-memory knowledge base
    pub fact_count: usize,
    /// How many rules already have a corresponding DAG-CBOR block in the store
    /// (i.e., content-addressed deduplication is active for those rules)
    pub cid_indexed_rules: usize,
}

/// Configuration for automatic snapshot persistence
pub struct TensorLogicPersistenceConfig {
    /// Path where snapshots are saved
    pub snapshot_path: std::path::PathBuf,
    /// Whether to auto-save when changes are detected
    pub auto_save: bool,
    /// Interval between automatic saves
    pub snapshot_interval: std::time::Duration,
}

/// Persistent snapshot of the knowledge base
///
/// Captures all rules and facts at a point in time for cross-restart persistence.
#[derive(Debug, Serialize, Deserialize)]
pub struct KnowledgeBaseSnapshot {
    /// Schema/format version
    pub version: u32,
    /// Serialized rules
    pub rules: Vec<RuleSnapshot>,
    /// Serialized facts
    pub facts: Vec<FactSnapshot>,
    /// Unix timestamp (seconds) when the snapshot was created
    pub created_at: u64,
    /// Number of rules in this snapshot
    pub rule_count: usize,
    /// Number of facts in this snapshot
    pub fact_count: usize,
}

/// Snapshot of a single rule
#[derive(Debug, Serialize, Deserialize)]
pub struct RuleSnapshot {
    /// Name of the head predicate
    pub head_predicate: String,
    /// String representations of head arguments
    pub head_args: Vec<String>,
    /// String representations of body goals
    pub body_goals: Vec<String>,
    /// Optional CID string (set if rule was previously stored as a block)
    pub cid: Option<String>,
}

/// Snapshot of a single ground fact
#[derive(Debug, Serialize, Deserialize)]
pub struct FactSnapshot {
    /// Predicate name
    pub predicate: String,
    /// String representations of arguments
    pub args: Vec<String>,
}

/// Errors specific to TensorLogic persistence operations
#[derive(Debug, thiserror::Error)]
pub enum TensorLogicError {
    #[error("Snapshot IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Snapshot serialization error: {0}")]
    Serialization(String),
    #[error("Lock poisoned: {0}")]
    LockPoisoned(String),
}

/// Storage manager for TensorLogic IR
///
/// Stores terms, predicates, and rules as content-addressed blocks
pub struct TensorLogicStore<S: BlockStore> {
    /// Underlying block store
    store: Arc<S>,
    /// In-memory knowledge base for inference
    knowledge_base: std::sync::RwLock<KnowledgeBase>,
    /// Inference engine
    engine: InferenceEngine,
    /// Whether knowledge base has unsaved changes since last snapshot
    dirty: AtomicBool,
    /// Rolling window of recent inference durations (capped at 100 entries).
    ///
    /// Guarded by a `parking_lot::Mutex` for contention-free appends from
    /// concurrent callers.
    inference_times: Mutex<VecDeque<Duration>>,
    /// Monotonic counter incremented on every KB mutation (add_rule, add_fact, retract).
    kb_version: AtomicU64,
    /// Inference memoization cache keyed by (goal_hash, kb_version).
    inference_cache: Mutex<InferenceCache>,
}

impl<S: BlockStore> TensorLogicStore<S> {
    /// Create a new TensorLogic store
    pub fn new(store: Arc<S>) -> Result<Self> {
        Ok(Self {
            store,
            knowledge_base: std::sync::RwLock::new(KnowledgeBase::new()),
            engine: InferenceEngine::new(),
            dirty: AtomicBool::new(false),
            inference_times: Mutex::new(VecDeque::with_capacity(100)),
            kb_version: AtomicU64::new(0),
            inference_cache: Mutex::new(InferenceCache::new(1024)),
        })
    }

    /// Return the current KB version counter (monotonically increasing).
    pub fn kb_version(&self) -> u64 {
        self.kb_version.load(Ordering::Acquire)
    }

    /// Access the inference cache statistics.
    pub fn inference_cache_stats(&self) -> crate::inference_cache::CacheStats {
        self.inference_cache.lock().stats()
    }

    /// Invalidate all inference cache entries for the current (pre-bump) KB version.
    fn bump_kb_version_and_invalidate(&self) {
        let old_version = self.kb_version.fetch_add(1, Ordering::AcqRel);
        self.inference_cache
            .lock()
            .invalidate_for_kb_version(old_version);
    }

    /// Store a term and return its CID
    pub async fn store_term(&self, term: &Term) -> Result<Cid> {
        let json = serde_json::to_vec(term)
            .map_err(|e| ipfrs_core::Error::Serialization(format!("Term serialization: {}", e)))?;

        let block = Block::new(json.into())?;
        let cid = *block.cid();

        self.store.put(&block).await?;

        Ok(cid)
    }

    /// Retrieve a term by CID
    pub async fn get_term(&self, cid: &Cid) -> Result<Option<Term>> {
        match self.store.get(cid).await? {
            Some(block) => {
                let term = serde_json::from_slice(block.data()).map_err(|e| {
                    ipfrs_core::Error::Deserialization(format!("Term deserialization: {}", e))
                })?;
                Ok(Some(term))
            }
            None => Ok(None),
        }
    }

    /// Store a predicate and return its CID
    pub async fn store_predicate(&self, predicate: &Predicate) -> Result<Cid> {
        let json = serde_json::to_vec(predicate).map_err(|e| {
            ipfrs_core::Error::Serialization(format!("Predicate serialization: {}", e))
        })?;

        let block = Block::new(json.into())?;
        let cid = *block.cid();

        self.store.put(&block).await?;

        Ok(cid)
    }

    /// Retrieve a predicate by CID
    pub async fn get_predicate(&self, cid: &Cid) -> Result<Option<Predicate>> {
        match self.store.get(cid).await? {
            Some(block) => {
                let predicate = serde_json::from_slice(block.data()).map_err(|e| {
                    ipfrs_core::Error::Deserialization(format!("Predicate deserialization: {}", e))
                })?;
                Ok(Some(predicate))
            }
            None => Ok(None),
        }
    }

    /// Store a rule and return its CID
    pub async fn store_rule(&self, rule: &Rule) -> Result<Cid> {
        let json = serde_json::to_vec(rule)
            .map_err(|e| ipfrs_core::Error::Serialization(format!("Rule serialization: {}", e)))?;

        let block = Block::new(json.into())?;
        let cid = *block.cid();

        self.store.put(&block).await?;

        Ok(cid)
    }

    /// Retrieve a rule by CID
    pub async fn get_rule(&self, cid: &Cid) -> Result<Option<Rule>> {
        match self.store.get(cid).await? {
            Some(block) => {
                let rule = serde_json::from_slice(block.data()).map_err(|e| {
                    ipfrs_core::Error::Deserialization(format!("Rule deserialization: {}", e))
                })?;
                Ok(Some(rule))
            }
            None => Ok(None),
        }
    }

    /// Check if a CID exists in storage
    pub async fn has(&self, cid: &Cid) -> Result<bool> {
        self.store.has(cid).await
    }

    /// Delete a stored item by CID
    pub async fn delete(&self, cid: &Cid) -> Result<()> {
        self.store.delete(cid).await
    }

    /// Add a fact to the knowledge base
    pub fn add_fact(&self, fact: Predicate) -> Result<()> {
        let mut kb = self.knowledge_base.write().map_err(|_| {
            ipfrs_core::Error::Storage("KB write lock poisoned in add_fact".to_string())
        })?;
        kb.add_fact(fact);
        self.dirty.store(true, Ordering::Release);
        drop(kb);
        self.bump_kb_version_and_invalidate();
        Ok(())
    }

    /// Add a rule to the knowledge base
    pub fn add_rule(&self, rule: Rule) -> Result<()> {
        let mut kb = self.knowledge_base.write().map_err(|_| {
            ipfrs_core::Error::Storage("KB write lock poisoned in add_rule".to_string())
        })?;
        kb.add_rule(rule);
        self.dirty.store(true, Ordering::Release);
        drop(kb);
        self.bump_kb_version_and_invalidate();
        Ok(())
    }

    /// Retract a fact from the knowledge base (removes first matching fact).
    pub fn retract_fact(&self, fact: &Predicate) -> Result<bool> {
        let mut kb = self.knowledge_base.write().map_err(|_| {
            ipfrs_core::Error::Storage("KB write lock poisoned in retract_fact".to_string())
        })?;
        let before = kb.facts.len();
        kb.facts.retain(|f| f != fact);
        let removed = kb.facts.len() < before;
        if removed {
            self.dirty.store(true, Ordering::Release);
            drop(kb);
            self.bump_kb_version_and_invalidate();
        }
        Ok(removed)
    }

    /// Run inference query on the knowledge base
    ///
    /// Records the wall-clock duration of each call in an internal rolling
    /// window of the last 100 durations so that `avg_inference_ms` can report
    /// a meaningful average.
    pub fn infer(&self, goal: &Predicate) -> Result<Vec<Substitution>> {
        let t0 = Instant::now();
        let kb = self.knowledge_base.read().map_err(|_| {
            ipfrs_core::Error::Storage("KB read lock poisoned in infer".to_string())
        })?;
        let result = self.engine.query(goal, &kb)?;
        let elapsed = t0.elapsed();
        drop(kb);
        let mut times = self.inference_times.lock();
        if times.len() >= 100 {
            times.pop_front();
        }
        times.push_back(elapsed);
        Ok(result)
    }

    /// Average inference duration in milliseconds over the last (up to 100) calls.
    ///
    /// Returns `None` when no inference has been performed yet.
    pub fn avg_inference_ms(&self) -> Option<f64> {
        let times = self.inference_times.lock();
        if times.is_empty() {
            return None;
        }
        let total_us: u128 = times.iter().map(|d| d.as_micros()).sum();
        Some(total_us as f64 / times.len() as f64 / 1000.0)
    }

    /// Generate a proof for a goal
    pub fn prove(&self, goal: &Predicate) -> Result<Option<Proof>> {
        let kb = self.knowledge_base.read().map_err(|_| {
            ipfrs_core::Error::Storage("KB read lock poisoned in prove".to_string())
        })?;
        self.engine.prove(goal, &kb)
    }

    /// Store a proof and return its CID
    pub async fn store_proof(&self, proof: &Proof) -> Result<Cid> {
        let json = serde_json::to_vec(proof)
            .map_err(|e| ipfrs_core::Error::Serialization(format!("Proof serialization: {}", e)))?;

        let block = Block::new(json.into())?;
        let cid = *block.cid();

        self.store.put(&block).await?;

        Ok(cid)
    }

    /// Retrieve a proof by CID
    pub async fn get_proof(&self, cid: &Cid) -> Result<Option<Proof>> {
        match self.store.get(cid).await? {
            Some(block) => {
                let proof = serde_json::from_slice(block.data()).map_err(|e| {
                    ipfrs_core::Error::Deserialization(format!("Proof deserialization: {}", e))
                })?;
                Ok(Some(proof))
            }
            None => Ok(None),
        }
    }

    /// Verify that a proof is valid against the current knowledge base
    pub fn verify_proof(&self, proof: &Proof) -> Result<bool> {
        let kb = self.knowledge_base.read().map_err(|_| {
            ipfrs_core::Error::Storage("KB read lock poisoned in verify_proof".to_string())
        })?;
        self.engine.verify(proof, &kb)
    }

    /// Get knowledge base statistics
    pub fn kb_stats(&self) -> crate::ir::KnowledgeBaseStats {
        match self.knowledge_base.read() {
            Ok(kb) => kb.stats(),
            Err(_) => KnowledgeBase::new().stats(),
        }
    }

    /// Estimated memory usage in bytes for the in-memory knowledge base.
    ///
    /// Uses a conservative heuristic:
    /// - Each rule  ≈ 500 bytes (head + body terms + serialised form)
    /// - Each fact  ≈ 200 bytes (predicate name + argument terms)
    pub fn estimated_memory_bytes(&self) -> usize {
        let stats = self.kb_stats();
        stats.num_rules * 500 + stats.num_facts * 200
    }

    /// Return a snapshot (clone) of the current in-memory knowledge base.
    ///
    /// The returned value is independent of the store; subsequent mutations are
    /// not reflected in the snapshot.
    pub fn snapshot_kb(&self) -> Result<KnowledgeBase> {
        let kb = self
            .knowledge_base
            .read()
            .map_err(|_| ipfrs_core::Error::Storage("KB lock poisoned".to_string()))?;
        Ok(kb.clone())
    }

    // ─── IPLD codec integration ───────────────────────────────────────────────

    /// Store a rule as a content-addressed IPLD block using the DAG-CBOR codec.
    ///
    /// Unlike `store_rule` which serialises via JSON directly, this method
    /// goes through the [`crate::ipld_codec`] pipeline:
    ///
    /// ```text
    /// Rule → RuleIpld → Block (DAG-CBOR codec, SHA-256 CID)
    /// ```
    ///
    /// Identical rules always produce the same CID, enabling deduplication
    /// across the network without reading the block contents.
    pub async fn store_rule_as_block(&self, rule: &Rule) -> Result<Cid> {
        use crate::ipld_codec::{rule_to_block, rule_to_rule_ipld};

        let rule_ipld = rule_to_rule_ipld(rule)?;
        let block = rule_to_block(&rule_ipld)?;
        let cid = *block.cid();

        self.store.put(&block).await?;

        Ok(cid)
    }

    /// Load a rule that was stored via `store_rule_as_block` (or any IPLD
    /// block whose content is a valid `RuleIpld` JSON blob).
    ///
    /// Returns [`ipfrs_core::Error::BlockNotFound`] when the CID is absent.
    pub async fn load_rule_from_block(&self, cid: &Cid) -> Result<Rule> {
        use crate::ipld_codec::{block_to_rule, rule_ipld_to_rule};

        let block = self
            .store
            .get(cid)
            .await?
            .ok_or_else(|| ipfrs_core::Error::BlockNotFound(cid.to_string()))?;

        let rule_ipld = block_to_rule(&block)?;
        rule_ipld_to_rule(&rule_ipld)
    }

    /// Snapshot the in-memory knowledge base as an IPLD DAG.
    ///
    /// Each rule is stored as an individual block (via `store_rule_as_block`)
    /// so that rules already present in the store are not re-written.  Facts
    /// are stored inline in the root `KnowledgeBaseIpld` block.
    ///
    /// Returns the CID of the root KB block.  The caller can pin this CID to
    /// prevent garbage-collection of the whole DAG.
    pub async fn store_kb_as_ipld(&self) -> Result<Cid> {
        use crate::ipld_codec::{kb_to_block, predicate_to_fact_ipld, KnowledgeBaseIpld};

        // Snapshot the in-memory KB (hold the lock only long enough to clone)
        let (rules, facts) = {
            let kb = self
                .knowledge_base
                .read()
                .map_err(|_| ipfrs_core::Error::Storage("KB lock poisoned".to_string()))?;
            (kb.rules.clone(), kb.facts.clone())
        };

        // Store each rule as a separate block; collect CID strings for the
        // root node.  We deliberately call store_rule_as_block so that the
        // idempotent put semantics of the underlying BlockStore handle
        // deduplication for us.
        let mut rule_cids: Vec<String> = Vec::with_capacity(rules.len());
        for rule in &rules {
            let cid = self.store_rule_as_block(rule).await?;
            rule_cids.push(cid.to_string());
        }

        // Convert facts to IPLD representation (inline in the root block)
        let fact_iplds = facts
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
        self.store.put(&root_block).await?;

        Ok(root_cid)
    }

    /// Check whether a rule is already present in the block store by computing
    /// its content-addressed CID and calling `has()`.
    ///
    /// This is a pure deduplication check: it does **not** add the rule to the
    /// in-memory knowledge base or write any block.  The conversion is the same
    /// pipeline used by `store_rule_as_block`, so the CID is guaranteed to
    /// match.
    pub async fn rule_exists_by_cid(&self, rule: &Rule) -> Result<bool> {
        use crate::ipld_codec::{rule_cid, rule_to_rule_ipld};

        let rule_ipld = rule_to_rule_ipld(rule)?;
        let cid = rule_cid(&rule_ipld)?;
        self.store.has(&cid).await
    }

    /// Extended knowledge base statistics that include content-addressed block
    /// accounting.
    ///
    /// `cid_indexed_rules` reflects how many rules in the in-memory KB are
    /// already persisted as DAG-CBOR blocks (i.e., `has()` returns `true` for
    /// their computed CID).  The async check is done concurrently for all
    /// rules so this method scales acceptably for large knowledge bases.
    pub async fn kb_stats_with_cids(&self) -> Result<TensorLogicStoreStats> {
        use crate::ipld_codec::{rule_cid, rule_to_rule_ipld};
        use futures::future;

        // Snapshot
        let (rules, facts) = {
            let kb = self
                .knowledge_base
                .read()
                .map_err(|_| ipfrs_core::Error::Storage("KB lock poisoned".to_string()))?;
            (kb.rules.clone(), kb.facts.clone())
        };

        // Fire off concurrent `has()` checks for every rule
        let has_futures: Vec<_> = rules
            .iter()
            .map(|rule| async move {
                let rule_ipld = rule_to_rule_ipld(rule)?;
                let cid = rule_cid(&rule_ipld)?;
                self.store.has(&cid).await
            })
            .collect();

        let has_results = future::join_all(has_futures).await;

        let mut cid_indexed_rules = 0usize;
        for res in has_results {
            if res? {
                cid_indexed_rules += 1;
            }
        }

        Ok(TensorLogicStoreStats {
            rule_count: rules.len(),
            fact_count: facts.len(),
            cid_indexed_rules,
        })
    }

    /// Build a predicate-name → CID index for all rules currently in the
    /// knowledge base by computing each rule's DAG-CBOR CID.
    ///
    /// Rules that fail CID computation are silently skipped.  The index is
    /// built from the current in-memory snapshot; changes made after this call
    /// are not reflected.
    ///
    /// This method is used by `DistributedBackwardChainer` to find DHT
    /// providers for relevant predicates without scanning all rules on every
    /// query.
    pub async fn index_rules_by_predicate(
        &self,
    ) -> Result<std::collections::HashMap<String, Vec<Cid>>> {
        use crate::ipld_codec::{rule_cid, rule_to_rule_ipld};

        let rules = {
            let kb = self
                .knowledge_base
                .read()
                .map_err(|_| ipfrs_core::Error::Storage("KB lock poisoned".to_string()))?;
            kb.rules.clone()
        };

        let mut cid_map: std::collections::HashMap<usize, Cid> =
            std::collections::HashMap::with_capacity(rules.len());

        for (idx, rule) in rules.iter().enumerate() {
            if let Ok(rule_ipld) = rule_to_rule_ipld(rule) {
                if let Ok(cid) = rule_cid(&rule_ipld) {
                    cid_map.insert(idx, cid);
                }
            }
        }

        // Re-acquire to build the final index using KnowledgeBase::index_rules_by_predicate
        let kb = self
            .knowledge_base
            .read()
            .map_err(|_| ipfrs_core::Error::Storage("KB lock poisoned".to_string()))?;

        Ok(kb.index_rules_by_predicate(&cid_map))
    }

    // ─── Snapshot persistence ─────────────────────────────────────────────────

    /// Save all rules and facts to a snapshot file.
    ///
    /// The snapshot is stored as pretty-printed JSON so it is human-readable
    /// and portable across restarts.  After a successful save the dirty flag
    /// is cleared.
    pub fn save_snapshot(
        &self,
        path: &std::path::Path,
    ) -> std::result::Result<(), TensorLogicError> {
        use std::io::Write;

        let (rules, facts) = {
            let kb = self
                .knowledge_base
                .read()
                .map_err(|e| TensorLogicError::LockPoisoned(e.to_string()))?;
            (kb.rules.clone(), kb.facts.clone())
        };

        let rule_snapshots: Vec<RuleSnapshot> = rules
            .iter()
            .map(|r| RuleSnapshot {
                head_predicate: r.head.name.clone(),
                head_args: r.head.args.iter().map(|a| format!("{:?}", a)).collect(),
                body_goals: r
                    .body
                    .iter()
                    .map(|g| format!("{}({:?})", g.name, g.args))
                    .collect(),
                cid: None,
            })
            .collect();

        let fact_snapshots: Vec<FactSnapshot> = facts
            .iter()
            .map(|f| FactSnapshot {
                predicate: f.name.clone(),
                args: f.args.iter().map(|a| format!("{:?}", a)).collect(),
            })
            .collect();

        let rule_count = rule_snapshots.len();
        let fact_count = fact_snapshots.len();

        let snapshot = KnowledgeBaseSnapshot {
            version: 1,
            rules: rule_snapshots,
            facts: fact_snapshots,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            rule_count,
            fact_count,
        };

        let json = serde_json::to_vec_pretty(&snapshot)
            .map_err(|e| TensorLogicError::Serialization(e.to_string()))?;

        let mut file = std::fs::File::create(path)?;
        file.write_all(&json)?;

        self.dirty.store(false, Ordering::Release);
        Ok(())
    }

    /// Load rules and facts from a previously saved snapshot file.
    ///
    /// The snapshot metadata (counts, timestamp, version) is returned.
    /// Note: the snapshot stores rules/facts as debug strings for
    /// human-readable audit trails.  The raw [`KnowledgeBase`] is
    /// persisted separately via `save_kb` / `load_kb` for full
    /// round-trip fidelity; this method populates snapshot metadata.
    ///
    /// After a successful load the dirty flag is cleared.
    pub fn load_snapshot(
        &mut self,
        path: &std::path::Path,
    ) -> std::result::Result<KnowledgeBaseSnapshot, TensorLogicError> {
        use std::io::Read;

        let mut file = std::fs::File::open(path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;

        let snapshot: KnowledgeBaseSnapshot = serde_json::from_slice(&buf)
            .map_err(|e| TensorLogicError::Serialization(e.to_string()))?;

        self.dirty.store(false, Ordering::Release);
        Ok(snapshot)
    }

    /// Whether the store has unsaved changes since the last snapshot save.
    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    /// Save the knowledge base to a file
    ///
    /// Serializes the entire knowledge base (facts and rules) to a file
    /// for later loading.
    ///
    /// # Arguments
    /// * `path` - Path to save the knowledge base file
    pub async fn save_kb<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let kb = self.knowledge_base.read().map_err(|_| {
            ipfrs_core::Error::Storage("KB read lock poisoned in save_kb".to_string())
        })?;

        // Serialize to oxicode
        let encoded =
            oxicode::serde::encode_to_vec(&*kb, oxicode::config::standard()).map_err(|e| {
                ipfrs_core::Error::Serialization(format!("Failed to serialize KB: {}", e))
            })?;

        // Write to file
        let mut file = File::create(path.as_ref())
            .map_err(|e| ipfrs_core::Error::Storage(format!("Failed to create KB file: {}", e)))?;

        file.write_all(&encoded)
            .map_err(|e| ipfrs_core::Error::Storage(format!("Failed to write KB file: {}", e)))?;

        Ok(())
    }

    /// Load a knowledge base from a file
    ///
    /// Loads a previously saved knowledge base from disk, replacing the current KB.
    ///
    /// # Arguments
    /// * `path` - Path to the saved knowledge base file
    pub async fn load_kb<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        use std::fs::File;
        use std::io::Read;

        // Read file
        let mut file = File::open(path.as_ref())
            .map_err(|e| ipfrs_core::Error::Storage(format!("Failed to open KB file: {}", e)))?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| ipfrs_core::Error::Storage(format!("Failed to read KB file: {}", e)))?;

        // Deserialize
        let kb: KnowledgeBase =
            oxicode::serde::decode_owned_from_slice(&buffer, oxicode::config::standard())
                .map(|(v, _)| v)
                .map_err(|e| {
                    ipfrs_core::Error::Deserialization(format!("Failed to deserialize KB: {}", e))
                })?;

        // Replace current KB
        let mut guard = self.knowledge_base.write().map_err(|_| {
            ipfrs_core::Error::Storage("KB write lock poisoned in load_kb".to_string())
        })?;
        *guard = kb;
        drop(guard);
        self.bump_kb_version_and_invalidate();

        Ok(())
    }
}

#[cfg(test)]
mod ipld_integration_tests {
    use super::*;
    use crate::ir::{Constant, Predicate, Rule, Term};
    use ipfrs_storage::{BlockStoreConfig, SledBlockStore};

    fn make_store(suffix: &str) -> TensorLogicStore<SledBlockStore> {
        let path = std::env::temp_dir().join(format!("ipfrs-test-tl-ipld-{}", suffix));
        let _ = std::fs::remove_dir_all(&path);
        let config = BlockStoreConfig {
            path,
            cache_size: 32 * 1024 * 1024,
        };
        let store = Arc::new(SledBlockStore::new(config).expect("test: should succeed"));
        TensorLogicStore::new(store).expect("test: should succeed")
    }

    fn grandparent_rule() -> Rule {
        // grandparent(X, Z) :- parent(X, Y), parent(Y, Z)
        let head = Predicate::new(
            "grandparent".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Z".to_string())],
        );
        let body = vec![
            Predicate::new(
                "parent".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            Predicate::new(
                "parent".to_string(),
                vec![Term::Var("Y".to_string()), Term::Var("Z".to_string())],
            ),
        ];
        Rule::new(head, body)
    }

    #[tokio::test]
    async fn test_store_and_load_rule_as_block() {
        let tl = make_store("store-load");
        let rule = grandparent_rule();

        let cid = tl
            .store_rule_as_block(&rule)
            .await
            .expect("test: should succeed");

        // Round-trip: load by CID and verify structure
        let loaded = tl
            .load_rule_from_block(&cid)
            .await
            .expect("test: should succeed");
        assert_eq!(loaded.head.name, rule.head.name);
        assert_eq!(loaded.head.args.len(), rule.head.args.len());
        assert_eq!(loaded.body.len(), rule.body.len());
        assert_eq!(loaded.body[0].name, rule.body[0].name);
    }

    #[tokio::test]
    async fn test_rule_deduplication_by_cid() {
        let tl = make_store("dedup");
        let rule = grandparent_rule();

        // Before storing: rule should not be in the block store
        let exists_before = tl
            .rule_exists_by_cid(&rule)
            .await
            .expect("test: should succeed");
        assert!(!exists_before, "Rule must not exist before first store");

        // Store once
        let cid1 = tl
            .store_rule_as_block(&rule)
            .await
            .expect("test: should succeed");

        // After first store: rule is present
        let exists_after = tl
            .rule_exists_by_cid(&rule)
            .await
            .expect("test: should succeed");
        assert!(exists_after, "Rule must exist after first store");

        // Store again: identical CID must be returned without error
        let cid2 = tl
            .store_rule_as_block(&rule)
            .await
            .expect("test: should succeed");
        assert_eq!(cid1, cid2, "Storing same rule twice must yield same CID");
    }

    #[tokio::test]
    async fn test_load_rule_from_block_not_found() {
        use crate::ipld_codec::{rule_cid, rule_to_rule_ipld};

        let tl = make_store("not-found");
        let rule = grandparent_rule();

        // Compute CID without storing
        let rule_ipld = rule_to_rule_ipld(&rule).expect("test: should succeed");
        let cid = rule_cid(&rule_ipld).expect("test: should succeed");

        let result = tl.load_rule_from_block(&cid).await;
        assert!(result.is_err(), "Loading unstored CID must return Err");
    }

    #[tokio::test]
    async fn test_kb_snapshot_as_ipld() {
        let tl = make_store("kb-snapshot");

        // Populate in-memory KB
        let rule = grandparent_rule();
        tl.add_rule(rule).expect("test: should succeed");
        tl.add_fact(Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String("alice".to_string())),
                Term::Const(Constant::String("bob".to_string())),
            ],
        ))
        .expect("test: should succeed");
        tl.add_fact(Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String("bob".to_string())),
                Term::Const(Constant::String("charlie".to_string())),
            ],
        ))
        .expect("test: should succeed");

        // Snapshot to IPLD DAG
        let root_cid = tl.store_kb_as_ipld().await.expect("test: should succeed");

        // Root block must be present
        let root_block = tl.store.get(&root_cid).await.expect("test: should succeed");
        assert!(root_block.is_some(), "Root KB block must be stored");

        // The rule block must also be present (individual rule blocks are stored)
        let stats = tl.kb_stats_with_cids().await.expect("test: should succeed");
        assert_eq!(stats.rule_count, 1);
        assert_eq!(stats.fact_count, 2);
        assert_eq!(
            stats.cid_indexed_rules, 1,
            "After kb_snapshot, all rules should have CID blocks"
        );
    }

    #[tokio::test]
    async fn test_kb_stats_with_cids_no_blocks() {
        let tl = make_store("stats-no-blocks");

        // Add rules to in-memory KB but do NOT store them as blocks
        tl.add_rule(grandparent_rule())
            .expect("test: should succeed");

        let stats = tl.kb_stats_with_cids().await.expect("test: should succeed");
        assert_eq!(stats.rule_count, 1);
        assert_eq!(stats.fact_count, 0);
        assert_eq!(
            stats.cid_indexed_rules, 0,
            "Rules added to KB but not as blocks should not be counted"
        );
    }

    #[tokio::test]
    async fn test_multiple_rules_partial_cid_coverage() {
        let tl = make_store("partial-cids");

        let rule_a = grandparent_rule();
        let rule_b = Rule::fact(Predicate::new(
            "mortal".to_string(),
            vec![Term::Var("X".to_string())],
        ));

        tl.add_rule(rule_a.clone()).expect("test: should succeed");
        tl.add_rule(rule_b.clone()).expect("test: should succeed");

        // Only store rule_a as a block
        tl.store_rule_as_block(&rule_a)
            .await
            .expect("test: should succeed");

        let stats = tl.kb_stats_with_cids().await.expect("test: should succeed");
        assert_eq!(stats.rule_count, 2);
        assert_eq!(
            stats.cid_indexed_rules, 1,
            "Only one rule is block-stored; cid_indexed_rules must be 1"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Constant;
    use ipfrs_storage::{BlockStoreConfig, SledBlockStore};

    #[tokio::test]
    async fn test_term_storage() {
        let config = BlockStoreConfig {
            path: std::path::PathBuf::from("/tmp/ipfrs-test-tensorlogic-term"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("test: should succeed"));
        let tl_store = TensorLogicStore::new(store).expect("test: should succeed");

        let term = Term::Const(Constant::String("Alice".to_string()));
        let cid = tl_store
            .store_term(&term)
            .await
            .expect("test: should succeed");

        let retrieved = tl_store.get_term(&cid).await.expect("test: should succeed");
        assert_eq!(retrieved, Some(term));
    }

    #[tokio::test]
    async fn test_predicate_storage() {
        let config = BlockStoreConfig {
            path: std::path::PathBuf::from("/tmp/ipfrs-test-tensorlogic-pred"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("test: should succeed"));
        let tl_store = TensorLogicStore::new(store).expect("test: should succeed");

        let predicate = Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String("Alice".to_string())),
                Term::Const(Constant::String("Bob".to_string())),
            ],
        );

        let cid = tl_store
            .store_predicate(&predicate)
            .await
            .expect("test: should succeed");
        let retrieved = tl_store
            .get_predicate(&cid)
            .await
            .expect("test: should succeed");
        assert_eq!(retrieved, Some(predicate));
    }

    #[tokio::test]
    async fn test_rule_storage() {
        let config = BlockStoreConfig {
            path: std::path::PathBuf::from("/tmp/ipfrs-test-tensorlogic-rule"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).expect("test: should succeed"));
        let tl_store = TensorLogicStore::new(store).expect("test: should succeed");

        let rule = Rule::fact(Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String("Alice".to_string())),
                Term::Const(Constant::String("Bob".to_string())),
            ],
        ));

        let cid = tl_store
            .store_rule(&rule)
            .await
            .expect("test: should succeed");
        let retrieved = tl_store.get_rule(&cid).await.expect("test: should succeed");
        assert!(retrieved.is_some());
    }
}

#[cfg(test)]
mod snapshot_tests {
    use super::*;
    use crate::ir::{Constant, Predicate, Rule, Term};
    use ipfrs_storage::{BlockStoreConfig, SledBlockStore};

    fn make_store(suffix: &str) -> TensorLogicStore<SledBlockStore> {
        let path = std::env::temp_dir().join(format!("ipfrs-snap-test-{}", suffix));
        let _ = std::fs::remove_dir_all(&path);
        let config = BlockStoreConfig {
            path,
            cache_size: 8 * 1024 * 1024,
        };
        let store = Arc::new(SledBlockStore::new(config).expect("test: should succeed"));
        TensorLogicStore::new(store).expect("test: should succeed")
    }

    fn grandparent_rule() -> Rule {
        let head = Predicate::new(
            "grandparent".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Z".to_string())],
        );
        let body = vec![
            Predicate::new(
                "parent".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            Predicate::new(
                "parent".to_string(),
                vec![Term::Var("Y".to_string()), Term::Var("Z".to_string())],
            ),
        ];
        Rule::new(head, body)
    }

    #[test]
    fn test_snapshot_save_load_roundtrip() {
        let store = make_store("roundtrip");

        store
            .add_rule(grandparent_rule())
            .expect("test: should succeed");
        store
            .add_fact(Predicate::new(
                "parent".to_string(),
                vec![
                    Term::Const(Constant::String("alice".to_string())),
                    Term::Const(Constant::String("bob".to_string())),
                ],
            ))
            .expect("test: should succeed");

        let snap_path = std::env::temp_dir().join("ipfrs-snap-roundtrip.json");
        store
            .save_snapshot(&snap_path)
            .expect("save_snapshot failed");

        // Dirty flag must be cleared after save
        assert!(!store.is_dirty(), "Dirty flag should be cleared after save");

        // Load into a fresh store instance (note: load_snapshot returns metadata)
        let mut store2 = make_store("roundtrip-load");
        let snapshot = store2
            .load_snapshot(&snap_path)
            .expect("load_snapshot failed");

        assert_eq!(snapshot.version, 1);
        assert_eq!(snapshot.rule_count, 1);
        assert_eq!(snapshot.fact_count, 1);
        assert_eq!(snapshot.rules[0].head_predicate, "grandparent");
        assert_eq!(snapshot.facts[0].predicate, "parent");
    }

    #[test]
    fn test_snapshot_empty_kb() {
        let store = make_store("empty");
        let snap_path = std::env::temp_dir().join("ipfrs-snap-empty.json");

        store
            .save_snapshot(&snap_path)
            .expect("save empty snapshot");
        let mut store2 = make_store("empty-load");
        let snapshot = store2
            .load_snapshot(&snap_path)
            .expect("load empty snapshot");

        assert_eq!(snapshot.rule_count, 0);
        assert_eq!(snapshot.fact_count, 0);
        assert!(snapshot.rules.is_empty());
        assert!(snapshot.facts.is_empty());
    }

    #[test]
    fn test_is_dirty_tracking() {
        let store = make_store("dirty");

        // Fresh store: not dirty
        assert!(!store.is_dirty(), "Fresh store should not be dirty");

        // Adding a fact marks it dirty
        store
            .add_fact(Predicate::new(
                "test".to_string(),
                vec![Term::Const(Constant::String("x".to_string()))],
            ))
            .expect("test: should succeed");
        assert!(store.is_dirty(), "Store should be dirty after add_fact");

        // Saving clears the dirty flag
        let snap_path = std::env::temp_dir().join("ipfrs-snap-dirty.json");
        store.save_snapshot(&snap_path).expect("save snapshot");
        assert!(!store.is_dirty(), "Store should not be dirty after save");

        // Adding a rule marks it dirty again
        store
            .add_rule(grandparent_rule())
            .expect("test: should succeed");
        assert!(store.is_dirty(), "Store should be dirty after add_rule");
    }
}

#[cfg(test)]
mod inference_tracking_tests {
    use super::*;
    use crate::ir::{Constant, Predicate, Term};
    use ipfrs_storage::{BlockStoreConfig, SledBlockStore};
    use std::sync::Arc;

    fn make_store(suffix: &str) -> TensorLogicStore<SledBlockStore> {
        let path = std::env::temp_dir().join(format!("ipfrs-inf-track-{}", suffix));
        let _ = std::fs::remove_dir_all(&path);
        let config = BlockStoreConfig {
            path,
            cache_size: 8 * 1024 * 1024,
        };
        let store = Arc::new(SledBlockStore::new(config).expect("test: should succeed"));
        TensorLogicStore::new(store).expect("test: should succeed")
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

    #[test]
    fn test_inference_time_tracking() {
        let store = make_store("infer-time");

        // No inferences yet — avg should be None.
        assert!(
            store.avg_inference_ms().is_none(),
            "avg_inference_ms should be None before any infer() call"
        );

        // Add a fact and run an inference.
        store
            .add_fact(parent_fact("alice", "bob"))
            .expect("test: should succeed");

        let goal = Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String("alice".to_string())),
                Term::Var("X".to_string()),
            ],
        );
        let solutions = store.infer(&goal).expect("test: should succeed");
        assert!(!solutions.is_empty(), "should find at least one solution");

        // avg_inference_ms should now be Some.
        let avg = store.avg_inference_ms();
        assert!(
            avg.is_some(),
            "avg_inference_ms should be Some after infer() call"
        );
        assert!(
            avg.expect("test: should succeed") >= 0.0,
            "avg_inference_ms should be non-negative"
        );
    }

    #[test]
    fn test_memory_bytes_nonzero() {
        let store = make_store("mem-bytes");

        // Empty KB → 0 estimated bytes.
        assert_eq!(
            store.estimated_memory_bytes(),
            0,
            "empty KB should report 0 estimated bytes"
        );

        // Add a fact → estimate should be > 0.
        store
            .add_fact(parent_fact("alice", "bob"))
            .expect("test: should succeed");
        let estimate = store.estimated_memory_bytes();
        assert!(
            estimate > 0,
            "estimated_memory_bytes should be > 0 after adding a fact (got {})",
            estimate
        );
    }
}
