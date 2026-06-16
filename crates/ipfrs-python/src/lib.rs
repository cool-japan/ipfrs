//! Python bindings for IPFRS
//!
//! This module provides Python bindings for IPFRS using PyO3.

use ipfrs::{Node as RustNode, NodeConfig as RustNodeConfig, QueryFilter as RustQueryFilter};
use ipfrs_core::{Block as RustBlock, Cid as RustCid, Error as RustError};
use ipfrs_tensorlogic::ir::{
    Constant, Predicate as RustPredicate, Rule as RustRule, Term as RustTerm,
};
use ipfrs_tensorlogic::reasoning::{Proof as RustProof, Substitution as RustSubstitution};
use parking_lot::Mutex;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;
use std::sync::Arc;

/// Python module for IPFRS
#[pymodule]
fn ipfrs_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Node>()?;
    m.add_class::<NodeConfig>()?;
    m.add_class::<Block>()?;
    m.add_class::<Cid>()?;
    m.add_class::<Term>()?;
    m.add_class::<Predicate>()?;
    m.add_class::<Rule>()?;
    m.add_class::<Proof>()?;
    m.add_class::<Substitution>()?;
    m.add_class::<Filter>()?;
    Ok(())
}

/// Convert Rust errors to Python exceptions
fn to_py_err(err: RustError) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

/// Node configuration
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct NodeConfig {
    inner: RustNodeConfig,
}

#[pymethods]
impl NodeConfig {
    /// Create a new node configuration
    #[new]
    #[pyo3(signature = (storage_path=None, enable_semantic=true, enable_tensorlogic=true))]
    fn new(storage_path: Option<String>, enable_semantic: bool, enable_tensorlogic: bool) -> Self {
        let mut config = RustNodeConfig::default();
        if let Some(path) = storage_path {
            config.storage.path = PathBuf::from(path);
        }
        config.enable_semantic = enable_semantic;
        config.enable_tensorlogic = enable_tensorlogic;
        Self { inner: config }
    }

    /// Create default configuration
    #[staticmethod]
    fn default() -> Self {
        Self {
            inner: RustNodeConfig::default(),
        }
    }
}

/// IPFRS Node - main interface for all operations
#[pyclass]
pub struct Node {
    inner: Arc<Mutex<RustNode>>,
    runtime: Arc<tokio::runtime::Runtime>,
}

// The parking_lot::Mutex guard is intentionally held across await points here because
// all async work is driven through `block_on`, which creates a synchronous barrier.
// The guard is always released when `block_on` returns, so there is no real deadlock risk.
#[allow(clippy::await_holding_lock)]
#[pymethods]
impl Node {
    /// Create a new IPFRS node
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<NodeConfig>) -> PyResult<Self> {
        let config = config.map(|c| c.inner).unwrap_or_default();

        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        let inner = RustNode::new(config).map_err(to_py_err)?;

        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
            runtime: Arc::new(runtime),
        })
    }

    /// Start the node
    fn start(&self) -> PyResult<()> {
        let inner = self.inner.clone();
        self.runtime
            .block_on(async move {
                let mut node = inner.lock();
                node.start().await
            })
            .map_err(to_py_err)
    }

    /// Stop the node
    fn stop(&self) -> PyResult<()> {
        let inner = self.inner.clone();
        self.runtime
            .block_on(async move {
                let mut node = inner.lock();
                node.stop().await
            })
            .map_err(to_py_err)
    }

    /// Add a block to storage
    fn put_block(&self, data: Vec<u8>) -> PyResult<Cid> {
        let block = RustBlock::new(data.into()).map_err(to_py_err)?;
        let cid = *block.cid();
        let inner = self.inner.clone();
        self.runtime
            .block_on(async move {
                let node = inner.lock();
                node.put_block(&block).await
            })
            .map_err(to_py_err)?;
        Ok(Cid { inner: cid })
    }

    /// Get a block from storage
    fn get_block(&self, cid: &Cid) -> PyResult<Option<Block>> {
        let inner = self.inner.clone();
        let cid = cid.inner;
        let block = self
            .runtime
            .block_on(async move {
                let node = inner.lock();
                node.get_block(&cid).await
            })
            .map_err(to_py_err)?;
        Ok(block.map(|b| Block { inner: b }))
    }

    /// Check if a block exists
    fn has_block(&self, cid: &Cid) -> PyResult<bool> {
        let inner = self.inner.clone();
        let cid = cid.inner;
        self.runtime
            .block_on(async move {
                let node = inner.lock();
                node.has_block(&cid).await
            })
            .map_err(to_py_err)
    }

    /// Delete a block from storage
    fn delete_block(&self, cid: &Cid) -> PyResult<()> {
        let inner = self.inner.clone();
        let cid = cid.inner;
        self.runtime
            .block_on(async move {
                let node = inner.lock();
                node.delete_block(&cid).await
            })
            .map_err(to_py_err)
    }

    /// Index content for semantic search
    fn index_content(&self, cid: &Cid, embedding: Vec<f32>) -> PyResult<()> {
        let inner = self.inner.clone();
        let cid = cid.inner;
        self.runtime
            .block_on(async move {
                let node = inner.lock();
                node.index_content(&cid, &embedding).await
            })
            .map_err(to_py_err)
    }

    /// Search for similar content
    fn search_similar(&self, query: Vec<f32>, k: usize) -> PyResult<Vec<(Cid, f32)>> {
        let inner = self.inner.clone();
        let results = self
            .runtime
            .block_on(async move {
                let node = inner.lock();
                node.search_similar(&query, k).await
            })
            .map_err(to_py_err)?;
        Ok(results
            .into_iter()
            .map(|r| (Cid { inner: r.cid }, r.score))
            .collect())
    }

    /// Search with filters
    #[pyo3(signature = (query, k, filter=None))]
    fn search_filtered(
        &self,
        query: Vec<f32>,
        k: usize,
        filter: Option<&Filter>,
    ) -> PyResult<Vec<(Cid, f32)>> {
        let rust_filter = filter.map(|f| f.inner.clone()).unwrap_or_default();
        let inner = self.inner.clone();
        let results = self
            .runtime
            .block_on(async move {
                let node = inner.lock();
                node.search_hybrid(&query, k, rust_filter).await
            })
            .map_err(to_py_err)?;
        Ok(results
            .into_iter()
            .map(|r| (Cid { inner: r.cid }, r.score))
            .collect())
    }

    /// Add a fact to the knowledge base
    fn add_fact(&self, fact: &Predicate) -> PyResult<()> {
        let node = self.inner.lock();
        node.add_fact(fact.inner.clone()).map_err(to_py_err)
    }

    /// Add a rule to the knowledge base
    fn add_rule(&self, rule: &Rule) -> PyResult<()> {
        let node = self.inner.lock();
        node.add_rule(rule.inner.clone()).map_err(to_py_err)
    }

    /// Run inference query
    fn infer(&self, goal: &Predicate) -> PyResult<Vec<Substitution>> {
        let node = self.inner.lock();
        let results = node.infer(&goal.inner).map_err(to_py_err)?;
        Ok(results
            .into_iter()
            .map(|s| Substitution { inner: s })
            .collect())
    }

    /// Generate a proof for a goal
    fn prove(&self, goal: &Predicate) -> PyResult<Option<Proof>> {
        let node = self.inner.lock();
        let proof = node.prove(&goal.inner).map_err(to_py_err)?;
        Ok(proof.map(|p| Proof { inner: p }))
    }

    /// Verify a proof
    fn verify_proof(&self, proof: &Proof) -> PyResult<bool> {
        let node = self.inner.lock();
        node.verify_proof(&proof.inner).map_err(to_py_err)
    }

    /// Get knowledge base statistics
    fn kb_stats(&self) -> PyResult<Py<PyDict>> {
        let node = self.inner.lock();
        let stats = node.tensorlogic_stats().map_err(to_py_err)?;
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("num_facts", stats.num_facts)?;
            dict.set_item("num_rules", stats.num_rules)?;
            Ok(dict.into())
        })
    }

    /// Save semantic index to disk
    fn save_semantic_index(&self, path: String) -> PyResult<()> {
        let inner = self.inner.clone();
        self.runtime
            .block_on(async move {
                let node = inner.lock();
                node.save_semantic_index(PathBuf::from(path)).await
            })
            .map_err(to_py_err)
    }

    /// Load semantic index from disk
    fn load_semantic_index(&self, path: String) -> PyResult<()> {
        let inner = self.inner.clone();
        self.runtime
            .block_on(async move {
                let node = inner.lock();
                node.load_semantic_index(PathBuf::from(path)).await
            })
            .map_err(to_py_err)
    }

    /// Save knowledge base to disk
    fn save_kb(&self, path: String) -> PyResult<()> {
        let inner = self.inner.clone();
        self.runtime
            .block_on(async move {
                let node = inner.lock();
                node.save_knowledge_base(PathBuf::from(path)).await
            })
            .map_err(to_py_err)
    }

    /// Load knowledge base from disk
    fn load_kb(&self, path: String) -> PyResult<()> {
        let inner = self.inner.clone();
        self.runtime
            .block_on(async move {
                let node = inner.lock();
                node.load_knowledge_base(PathBuf::from(path)).await
            })
            .map_err(to_py_err)
    }
}

/// Content-addressed block
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Block {
    inner: RustBlock,
}

#[pymethods]
impl Block {
    /// Create a new block from data
    #[new]
    fn new(data: Vec<u8>) -> PyResult<Self> {
        let inner = RustBlock::new(data.into()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Get block data
    fn data(&self) -> Vec<u8> {
        self.inner.data().to_vec()
    }

    /// Get block CID
    fn cid(&self) -> Cid {
        Cid {
            inner: *self.inner.cid(),
        }
    }

    /// Get block size
    fn size(&self) -> usize {
        self.inner.data().len()
    }
}

/// Content Identifier
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Cid {
    inner: RustCid,
}

#[pymethods]
impl Cid {
    /// Parse CID from string
    #[staticmethod]
    fn parse(s: &str) -> PyResult<Self> {
        let inner = s
            .parse()
            .map_err(|_| PyRuntimeError::new_err("Invalid CID string"))?;
        Ok(Self { inner })
    }

    /// Convert CID to string
    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    /// Convert CID to repr
    fn __repr__(&self) -> String {
        format!("Cid('{}')", self.inner)
    }
}

/// Logical term
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Term {
    inner: RustTerm,
}

#[pymethods]
impl Term {
    /// Create a constant integer term
    #[staticmethod]
    fn int(value: i64) -> Self {
        Self {
            inner: RustTerm::Const(Constant::Int(value)),
        }
    }

    /// Create a constant float term
    #[staticmethod]
    fn float(value: f64) -> Self {
        Self {
            inner: RustTerm::Const(Constant::Float(value.to_string())),
        }
    }

    /// Create a constant string term
    #[staticmethod]
    fn string(value: String) -> Self {
        Self {
            inner: RustTerm::Const(Constant::String(value)),
        }
    }

    /// Create a constant boolean term
    #[staticmethod]
    fn bool(value: bool) -> Self {
        Self {
            inner: RustTerm::Const(Constant::Bool(value)),
        }
    }

    /// Create a variable term
    #[staticmethod]
    fn var(name: String) -> Self {
        Self {
            inner: RustTerm::Var(name),
        }
    }

    /// String representation
    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// Logical predicate
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Predicate {
    inner: RustPredicate,
}

#[pymethods]
impl Predicate {
    /// Create a new predicate
    #[new]
    fn new(name: String, args: Vec<Py<Term>>) -> PyResult<Self> {
        Python::attach(|py| {
            let rust_args: Vec<RustTerm> = args
                .into_iter()
                .map(|t| t.borrow(py).inner.clone())
                .collect();
            Ok(Self {
                inner: RustPredicate::new(name, rust_args),
            })
        })
    }

    /// String representation
    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// Logical rule
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Rule {
    inner: RustRule,
}

#[pymethods]
impl Rule {
    /// Create a fact (rule with no body)
    #[staticmethod]
    fn fact(head: &Predicate) -> Self {
        Self {
            inner: RustRule::fact(head.inner.clone()),
        }
    }

    /// Create a rule with body
    #[staticmethod]
    #[allow(clippy::self_named_constructors)]
    fn rule(head: &Predicate, body: Vec<Py<Predicate>>) -> PyResult<Self> {
        Python::attach(|py| {
            let rust_body: Vec<RustPredicate> = body
                .into_iter()
                .map(|p| p.borrow(py).inner.clone())
                .collect();
            Ok(Self {
                inner: RustRule::new(head.inner.clone(), rust_body),
            })
        })
    }

    /// String representation
    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// Proof tree
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Proof {
    inner: RustProof,
}

#[pymethods]
impl Proof {
    /// String representation
    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// Variable substitution
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Substitution {
    inner: RustSubstitution,
}

#[pymethods]
impl Substitution {
    /// Get bindings as dictionary
    fn bindings(&self) -> PyResult<Py<PyDict>> {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            for (var, term) in self.inner.iter() {
                dict.set_item(var, format!("{:?}", term))?;
            }
            Ok(dict.into())
        })
    }

    /// String representation
    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// Search filter
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Filter {
    inner: RustQueryFilter,
}

#[pymethods]
impl Filter {
    /// Create a filter with minimum score threshold
    #[staticmethod]
    fn min_score(min_score: f32) -> Self {
        Self {
            inner: RustQueryFilter {
                min_score: Some(min_score),
                max_score: None,
                max_results: None,
                cid_prefix: None,
            },
        }
    }

    /// Create a filter with maximum score threshold
    #[staticmethod]
    fn max_score(max_score: f32) -> Self {
        Self {
            inner: RustQueryFilter {
                min_score: None,
                max_score: Some(max_score),
                max_results: None,
                cid_prefix: None,
            },
        }
    }

    /// Create a filter with maximum results limit
    #[staticmethod]
    fn max_results(max_results: usize) -> Self {
        Self {
            inner: RustQueryFilter {
                min_score: None,
                max_score: None,
                max_results: Some(max_results),
                cid_prefix: None,
            },
        }
    }
}
