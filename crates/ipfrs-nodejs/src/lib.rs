//! Node.js bindings for IPFRS
//!
//! This module provides JavaScript/TypeScript bindings for IPFRS using NAPI-RS.

use ipfrs::{Node as RustNode, NodeConfig as RustNodeConfig, QueryFilter as RustQueryFilter};
use ipfrs_core::{Block as RustBlock, Cid as RustCid};
use ipfrs_tensorlogic::ir::{Constant, Predicate as RustPredicate, Rule as RustRule, Term as RustTerm};
use ipfrs_tensorlogic::reasoning::{Proof as RustProof, Substitution as RustSubstitution};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::path::PathBuf;
use std::sync::Arc;
use parking_lot::Mutex;

/// Node configuration
#[napi(object)]
#[derive(Clone)]
pub struct NodeConfig {
    /// Path to storage directory
    pub storage_path: Option<String>,
    /// Enable semantic search
    pub enable_semantic: Option<bool>,
    /// Enable TensorLogic
    pub enable_tensorlogic: Option<bool>,
}

impl NodeConfig {
    fn to_rust_config(&self) -> RustNodeConfig {
        let mut config = RustNodeConfig::default();
        if let Some(ref path) = self.storage_path {
            config.storage.path = PathBuf::from(path);
        }
        if let Some(enable_semantic) = self.enable_semantic {
            config.enable_semantic = enable_semantic;
        }
        if let Some(enable_tensorlogic) = self.enable_tensorlogic {
            config.enable_tensorlogic = enable_tensorlogic;
        }
        config
    }
}

/// IPFRS Node - main interface for all operations
#[napi]
pub struct Node {
    inner: Arc<Mutex<RustNode>>,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[napi]
impl Node {
    /// Create a new IPFRS node
    #[napi(constructor)]
    pub fn new(config: Option<NodeConfig>) -> Result<Self> {
        let rust_config = config
            .map(|c| c.to_rust_config())
            .unwrap_or_else(RustNodeConfig::default);

        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| Error::from_reason(format!("Failed to create runtime: {}", e)))?;

        let inner = RustNode::new(rust_config)
            .map_err(|e| Error::from_reason(format!("Failed to create node: {}", e)))?;

        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
            runtime: Arc::new(runtime),
        })
    }

    /// Start the node
    #[napi]
    pub async fn start(&self) -> Result<()> {
        let inner = self.inner.clone();
        let result = self.runtime.block_on(async move {
            let mut node = inner.lock();
            node.start().await
        });
        result.map_err(|e| Error::from_reason(format!("Failed to start: {}", e)))
    }

    /// Stop the node
    #[napi]
    pub async fn stop(&self) -> Result<()> {
        let inner = self.inner.clone();
        let result = self.runtime.block_on(async move {
            let mut node = inner.lock();
            node.stop().await
        });
        result.map_err(|e| Error::from_reason(format!("Failed to stop: {}", e)))
    }

    /// Add a block to storage
    #[napi]
    pub async fn put_block(&self, data: Buffer) -> Result<String> {
        let data_vec = data.to_vec();
        let block = RustBlock::new(data_vec.into())
            .map_err(|e| Error::from_reason(format!("Failed to create block: {}", e)))?;
        let cid = *block.cid();

        let inner = self.inner.clone();
        let result = self.runtime.block_on(async move {
            let node = inner.lock();
            node.put_block(&block).await
        });
        result.map_err(|e| Error::from_reason(format!("Failed to put block: {}", e)))?;

        Ok(cid.to_string())
    }

    /// Get a block from storage
    #[napi]
    pub async fn get_block(&self, cid: String) -> Result<Option<Buffer>> {
        let cid: RustCid = cid.parse()
            .map_err(|_| Error::from_reason("Invalid CID"))?;

        let inner = self.inner.clone();
        let result = self.runtime.block_on(async move {
            let node = inner.lock();
            node.get_block(&cid).await
        });

        match result {
            Ok(Some(block)) => Ok(Some(block.data().to_vec().into())),
            Ok(None) => Ok(None),
            Err(e) => Err(Error::from_reason(format!("Failed to get block: {}", e))),
        }
    }

    /// Check if a block exists
    #[napi]
    pub async fn has_block(&self, cid: String) -> Result<bool> {
        let cid: RustCid = cid.parse()
            .map_err(|_| Error::from_reason("Invalid CID"))?;

        let inner = self.inner.clone();
        let result = self.runtime.block_on(async move {
            let node = inner.lock();
            node.has_block(&cid).await
        });

        result.map_err(|e| Error::from_reason(format!("Failed to check block: {}", e)))
    }

    /// Delete a block from storage
    #[napi]
    pub async fn delete_block(&self, cid: String) -> Result<()> {
        let cid: RustCid = cid.parse()
            .map_err(|_| Error::from_reason("Invalid CID"))?;

        let inner = self.inner.clone();
        let result = self.runtime.block_on(async move {
            let node = inner.lock();
            node.delete_block(&cid).await
        });

        result.map_err(|e| Error::from_reason(format!("Failed to delete block: {}", e)))
    }

    /// Index content for semantic search
    #[napi]
    pub async fn index_content(&self, cid: String, embedding: Vec<f32>) -> Result<()> {
        let cid: RustCid = cid.parse()
            .map_err(|_| Error::from_reason("Invalid CID"))?;

        let inner = self.inner.clone();
        let result = self.runtime.block_on(async move {
            let node = inner.lock();
            node.index_content(&cid, &embedding).await
        });

        result.map_err(|e| Error::from_reason(format!("Failed to index: {}", e)))
    }

    /// Search for similar content
    #[napi]
    pub async fn search_similar(&self, query: Vec<f32>, k: u32) -> Result<Vec<SearchResult>> {
        let inner = self.inner.clone();
        let k = k as usize;
        let result = self.runtime.block_on(async move {
            let node = inner.lock();
            node.search_similar(&query, k).await
        });

        match result {
            Ok(results) => Ok(results.into_iter().map(|r| SearchResult {
                cid: r.cid.to_string(),
                score: r.score,
            }).collect()),
            Err(e) => Err(Error::from_reason(format!("Search failed: {}", e))),
        }
    }

    /// Search with filters
    #[napi]
    pub async fn search_filtered(
        &self,
        query: Vec<f32>,
        k: u32,
        filter: Option<QueryFilter>,
    ) -> Result<Vec<SearchResult>> {
        let rust_filter = filter
            .map(|f| f.to_rust_filter())
            .unwrap_or_else(RustQueryFilter::default);

        let inner = self.inner.clone();
        let k = k as usize;
        let result = self.runtime.block_on(async move {
            let node = inner.lock();
            node.search_hybrid(&query, k, rust_filter).await
        });

        match result {
            Ok(results) => Ok(results.into_iter().map(|r| SearchResult {
                cid: r.cid.to_string(),
                score: r.score,
            }).collect()),
            Err(e) => Err(Error::from_reason(format!("Search failed: {}", e))),
        }
    }

    /// Add a fact to the knowledge base
    #[napi]
    pub fn add_fact(&self, fact: &Predicate) -> Result<()> {
        let node = self.inner.lock();
        let rust_fact = fact.to_rust_predicate()?;
        node.add_fact(rust_fact)
            .map_err(|e| Error::from_reason(format!("Failed to add fact: {}", e)))
    }

    /// Add a rule to the knowledge base
    #[napi]
    pub fn add_rule(&self, rule: &Rule) -> Result<()> {
        let node = self.inner.lock();
        let rust_rule = rule.to_rust_rule()?;
        node.add_rule(rust_rule)
            .map_err(|e| Error::from_reason(format!("Failed to add rule: {}", e)))
    }

    /// Run inference query
    #[napi]
    pub fn infer(&self, goal: &Predicate) -> Result<Vec<String>> {
        let node = self.inner.lock();
        let rust_goal = goal.to_rust_predicate()?;
        let results = node.infer(&rust_goal)
            .map_err(|e| Error::from_reason(format!("Inference failed: {}", e)))?;

        Ok(results.into_iter().map(|s| format!("{:?}", s)).collect())
    }

    /// Generate a proof for a goal
    #[napi]
    pub fn prove(&self, goal: &Predicate) -> Result<Option<String>> {
        let node = self.inner.lock();
        let rust_goal = goal.to_rust_predicate()?;
        let proof = node.prove(&rust_goal)
            .map_err(|e| Error::from_reason(format!("Proof generation failed: {}", e)))?;

        Ok(proof.map(|p| format!("{:?}", p)))
    }

    /// Verify a proof (simplified - takes proof as string)
    #[napi]
    pub fn verify_proof_str(&self, _proof_str: String) -> Result<bool> {
        // Simplified version - in production would deserialize proof
        Err(Error::from_reason("Proof verification not yet implemented for Node.js"))
    }

    /// Get knowledge base statistics
    #[napi]
    pub fn kb_stats(&self) -> Result<KbStats> {
        let node = self.inner.lock();
        let stats = node.tensorlogic_stats()
            .map_err(|e| Error::from_reason(format!("Failed to get stats: {}", e)))?;

        Ok(KbStats {
            num_facts: stats.num_facts as u32,
            num_rules: stats.num_rules as u32,
        })
    }

    /// Save semantic index to disk
    #[napi]
    pub async fn save_semantic_index(&self, path: String) -> Result<()> {
        let inner = self.inner.clone();
        let result = self.runtime.block_on(async move {
            let node = inner.lock();
            node.save_semantic_index(PathBuf::from(path)).await
        });

        result.map_err(|e| Error::from_reason(format!("Failed to save index: {}", e)))
    }

    /// Load semantic index from disk
    #[napi]
    pub async fn load_semantic_index(&self, path: String) -> Result<()> {
        let inner = self.inner.clone();
        let result = self.runtime.block_on(async move {
            let node = inner.lock();
            node.load_semantic_index(PathBuf::from(path)).await
        });

        result.map_err(|e| Error::from_reason(format!("Failed to load index: {}", e)))
    }

    /// Save knowledge base to disk
    #[napi]
    pub async fn save_kb(&self, path: String) -> Result<()> {
        let inner = self.inner.clone();
        let result = self.runtime.block_on(async move {
            let node = inner.lock();
            node.save_knowledge_base(PathBuf::from(path)).await
        });

        result.map_err(|e| Error::from_reason(format!("Failed to save KB: {}", e)))
    }

    /// Load knowledge base from disk
    #[napi]
    pub async fn load_kb(&self, path: String) -> Result<()> {
        let inner = self.inner.clone();
        let result = self.runtime.block_on(async move {
            let node = inner.lock();
            node.load_knowledge_base(PathBuf::from(path)).await
        });

        result.map_err(|e| Error::from_reason(format!("Failed to load KB: {}", e)))
    }
}

/// Search result
#[napi(object)]
pub struct SearchResult {
    pub cid: String,
    pub score: f32,
}

/// Query filter
#[napi(object)]
#[derive(Clone)]
pub struct QueryFilter {
    pub min_score: Option<f32>,
    pub max_score: Option<f32>,
    pub max_results: Option<u32>,
    pub cid_prefix: Option<String>,
}

impl QueryFilter {
    fn to_rust_filter(&self) -> RustQueryFilter {
        RustQueryFilter {
            min_score: self.min_score,
            max_score: self.max_score,
            max_results: self.max_results.map(|n| n as usize),
            cid_prefix: self.cid_prefix.clone(),
        }
    }
}

/// Knowledge base statistics
#[napi(object)]
pub struct KbStats {
    pub num_facts: u32,
    pub num_rules: u32,
}

/// Logical term
#[napi(object)]
#[derive(Clone)]
pub struct Term {
    pub kind: String,
    pub value: String,
}

impl Term {
    fn to_rust_term(&self) -> Result<RustTerm> {
        match self.kind.as_str() {
            "int" => {
                let val: i64 = self.value.parse()
                    .map_err(|_| Error::from_reason("Invalid integer"))?;
                Ok(RustTerm::Const(Constant::Int(val)))
            }
            "float" => {
                Ok(RustTerm::Const(Constant::Float(self.value.clone())))
            }
            "string" => {
                Ok(RustTerm::Const(Constant::String(self.value.clone())))
            }
            "bool" => {
                let val: bool = self.value.parse()
                    .map_err(|_| Error::from_reason("Invalid boolean"))?;
                Ok(RustTerm::Const(Constant::Bool(val)))
            }
            "var" => {
                Ok(RustTerm::Var(self.value.clone()))
            }
            _ => Err(Error::from_reason(format!("Unknown term kind: {}", self.kind))),
        }
    }
}

/// Logical predicate
#[napi(object)]
#[derive(Clone)]
pub struct Predicate {
    pub name: String,
    pub args: Vec<Term>,
}

impl Predicate {
    fn to_rust_predicate(&self) -> Result<RustPredicate> {
        let rust_args: Result<Vec<RustTerm>> = self.args.iter()
            .map(|t| t.to_rust_term())
            .collect();

        Ok(RustPredicate::new(self.name.clone(), rust_args?))
    }
}

/// Logical rule
#[napi(object)]
#[derive(Clone)]
pub struct Rule {
    pub head: Predicate,
    pub body: Vec<Predicate>,
}

impl Rule {
    fn to_rust_rule(&self) -> Result<RustRule> {
        let rust_head = self.head.to_rust_predicate()?;
        let rust_body: Result<Vec<RustPredicate>> = self.body.iter()
            .map(|p| p.to_rust_predicate())
            .collect();

        if self.body.is_empty() {
            Ok(RustRule::fact(rust_head))
        } else {
            Ok(RustRule::new(rust_head, rust_body?))
        }
    }
}
