//! Storage for TensorLogic IR
//!
//! Provides content-addressed storage for logical terms, predicates, and rules

use crate::ir::{KnowledgeBase, Predicate, Rule, Term};
use crate::reasoning::{InferenceEngine, Proof, Substitution};
use ipfrs_core::{Block, Cid, Result};
use ipfrs_storage::traits::BlockStore;
use serde_json;
use std::sync::Arc;

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
}

impl<S: BlockStore> TensorLogicStore<S> {
    /// Create a new TensorLogic store
    pub fn new(store: Arc<S>) -> Result<Self> {
        Ok(Self {
            store,
            knowledge_base: std::sync::RwLock::new(KnowledgeBase::new()),
            engine: InferenceEngine::new(),
        })
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
        let mut kb = self.knowledge_base.write().unwrap();
        kb.add_fact(fact);
        Ok(())
    }

    /// Add a rule to the knowledge base
    pub fn add_rule(&self, rule: Rule) -> Result<()> {
        let mut kb = self.knowledge_base.write().unwrap();
        kb.add_rule(rule);
        Ok(())
    }

    /// Run inference query on the knowledge base
    pub fn infer(&self, goal: &Predicate) -> Result<Vec<Substitution>> {
        let kb = self.knowledge_base.read().unwrap();
        self.engine.query(goal, &kb)
    }

    /// Generate a proof for a goal
    pub fn prove(&self, goal: &Predicate) -> Result<Option<Proof>> {
        let kb = self.knowledge_base.read().unwrap();
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
        let kb = self.knowledge_base.read().unwrap();
        self.engine.verify(proof, &kb)
    }

    /// Get knowledge base statistics
    pub fn kb_stats(&self) -> crate::ir::KnowledgeBaseStats {
        let kb = self.knowledge_base.read().unwrap();
        kb.stats()
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

        let kb = self.knowledge_base.read().unwrap();

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
        *self.knowledge_base.write().unwrap() = kb;

        Ok(())
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
        let store = Arc::new(SledBlockStore::new(config).unwrap());
        let tl_store = TensorLogicStore::new(store).unwrap();

        let term = Term::Const(Constant::String("Alice".to_string()));
        let cid = tl_store.store_term(&term).await.unwrap();

        let retrieved = tl_store.get_term(&cid).await.unwrap();
        assert_eq!(retrieved, Some(term));
    }

    #[tokio::test]
    async fn test_predicate_storage() {
        let config = BlockStoreConfig {
            path: std::path::PathBuf::from("/tmp/ipfrs-test-tensorlogic-pred"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).unwrap());
        let tl_store = TensorLogicStore::new(store).unwrap();

        let predicate = Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String("Alice".to_string())),
                Term::Const(Constant::String("Bob".to_string())),
            ],
        );

        let cid = tl_store.store_predicate(&predicate).await.unwrap();
        let retrieved = tl_store.get_predicate(&cid).await.unwrap();
        assert_eq!(retrieved, Some(predicate));
    }

    #[tokio::test]
    async fn test_rule_storage() {
        let config = BlockStoreConfig {
            path: std::path::PathBuf::from("/tmp/ipfrs-test-tensorlogic-rule"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).unwrap());
        let tl_store = TensorLogicStore::new(store).unwrap();

        let rule = Rule::fact(Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String("Alice".to_string())),
                Term::Const(Constant::String("Bob".to_string())),
            ],
        ));

        let cid = tl_store.store_rule(&rule).await.unwrap();
        let retrieved = tl_store.get_rule(&cid).await.unwrap();
        assert!(retrieved.is_some());
    }
}
