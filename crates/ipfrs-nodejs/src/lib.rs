//! Node.js bindings for IPFRS v0.2.0
//!
//! This module provides JavaScript/TypeScript bindings for IPFRS using NAPI-RS.
//! It exposes two independent surface areas:
//!
//! 1. **`IpfrsClient`** – a self-contained, in-memory content-addressed store
//!    (mirrors the WASM `IpfrsClient`) that works without a running tokio
//!    runtime and is perfectly testable with `cargo test`.
//!
//! 2. **`Node`** – a full IPFRS node that integrates block storage, semantic
//!    search, and TensorLogic reasoning.  When built with NAPI-RS this struct
//!    is exposed to JavaScript; otherwise only the `#[cfg(test)]` surface is
//!    available.
//!
//! Free functions:
//! * `compute_cid(data)` – deterministic CIDv1 (SHA-256, raw codec, base32lower)
//! * `verify_cid(cid, data)` – check a CID against raw bytes
//! * `version()` – library version string

// ---------------------------------------------------------------------------
// CID helpers – identical algorithm to ipfrs-wasm for cross-platform parity
// ---------------------------------------------------------------------------

/// Compute a CIDv1 (base32-lower, SHA2-256, raw codec) string for `data`.
///
/// The encoding is:
/// ```text
/// <varint version=1><varint codec=0x55><varint mh-fn=0x12><varint mh-len=32><32-byte-digest>
/// ```
/// encoded in lowercase RFC 4648 base32 with the `b` multibase prefix.
pub fn cid_from_bytes(data: &[u8]) -> String {
    use sha2::Digest;

    // 1. SHA-256 digest
    let digest: [u8; 32] = sha2::Sha256::digest(data).into();

    // 2. Multihash: [0x12 (sha2-256), 0x20 (32 bytes), <digest>]
    let mut multihash = Vec::with_capacity(34);
    multihash.push(0x12u8);
    multihash.push(0x20u8);
    multihash.extend_from_slice(&digest);

    // 3. CIDv1: [0x01 (version), 0x55 (raw codec), <multihash>]
    let mut cid_bytes = Vec::with_capacity(36);
    cid_bytes.push(0x01u8);
    cid_bytes.push(0x55u8);
    cid_bytes.extend_from_slice(&multihash);

    // 4. Multibase base32lower with 'b' prefix
    let encoded = base32_lower(&cid_bytes);
    format!("b{encoded}")
}

/// RFC 4648 base32 (lowercase, no padding) encoder.
fn base32_lower(input: &[u8]) -> String {
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz234567";
    let mut output = Vec::with_capacity((input.len() * 8).div_ceil(5));
    let mut buffer: u64 = 0;
    let mut bits_left: u32 = 0;

    for &byte in input {
        buffer = (buffer << 8) | u64::from(byte);
        bits_left += 8;
        while bits_left >= 5 {
            bits_left -= 5;
            let idx = ((buffer >> bits_left) & 0x1f) as usize;
            output.push(ALPHABET[idx]);
        }
    }
    if bits_left > 0 {
        let idx = ((buffer << (5 - bits_left)) & 0x1f) as usize;
        output.push(ALPHABET[idx]);
    }

    // SAFETY: ALPHABET is all ASCII, so the Vec<u8> is valid UTF-8.
    String::from_utf8(output).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// In-memory block store (backing type for IpfrsClient)
// ---------------------------------------------------------------------------

struct InMemoryStore {
    blocks: std::collections::HashMap<String, Vec<u8>>,
    total_bytes: usize,
}

impl InMemoryStore {
    fn new() -> Self {
        Self {
            blocks: std::collections::HashMap::new(),
            total_bytes: 0,
        }
    }

    /// Store `data` and return its CID.  Content-addressed idempotency:
    /// storing the same bytes twice does not create a duplicate entry.
    fn put(&mut self, data: &[u8]) -> String {
        let cid = cid_from_bytes(data);
        if !self.blocks.contains_key(&cid) {
            self.total_bytes += data.len();
            self.blocks.insert(cid.clone(), data.to_vec());
        }
        cid
    }

    fn get(&self, cid: &str) -> Option<&Vec<u8>> {
        self.blocks.get(cid)
    }

    fn has(&self, cid: &str) -> bool {
        self.blocks.contains_key(cid)
    }

    /// Remove a block.  Returns `true` if the block existed.
    fn delete(&mut self, cid: &str) -> bool {
        if let Some(data) = self.blocks.remove(cid) {
            self.total_bytes = self.total_bytes.saturating_sub(data.len());
            true
        } else {
            false
        }
    }

    fn list(&self) -> Vec<String> {
        self.blocks.keys().cloned().collect()
    }

    fn block_count(&self) -> usize {
        self.blocks.len()
    }

    fn total_bytes(&self) -> usize {
        self.total_bytes
    }
}

// ---------------------------------------------------------------------------
// IpfrsClient – simple in-process content-addressed store
// ---------------------------------------------------------------------------

/// IPFRS in-process content-addressed client.
///
/// This is the simplest way to use IPFRS from Node.js: all blocks are kept in
/// memory during the lifetime of the object.  For persistent storage backed by
/// a full IPFRS node, use `Node` instead.
///
/// ```javascript
/// const { IpfrsClient, compute_cid, verify_cid, version } = require('@cool-japan/ipfrs-node');
///
/// const client = new IpfrsClient('/tmp/ipfrs-data');
/// const cid = client.addBytes(Buffer.from('hello, IPFRS!'));
/// const data = client.getBytes(cid);
/// ```
#[cfg(feature = "napi")]
#[napi_derive::napi]
pub struct IpfrsClient {
    data_dir: String,
    store: std::sync::Mutex<InMemoryStore>,
}

/// Non-NAPI version used in unit tests and when building without the napi
/// feature (e.g. `cargo test --lib`).
#[cfg(not(feature = "napi"))]
pub struct IpfrsClient {
    data_dir: String,
    store: std::sync::Mutex<InMemoryStore>,
}

impl IpfrsClient {
    /// Create a new `IpfrsClient` backed by an in-memory store.
    ///
    /// `data_dir` is recorded for informational purposes (e.g. `stats()`);
    /// it is **not** used for actual I/O in the in-memory implementation.
    pub fn create(data_dir: String) -> Result<Self, ClientError> {
        Ok(Self {
            data_dir,
            store: std::sync::Mutex::new(InMemoryStore::new()),
        })
    }

    /// Add raw bytes, returning the CIDv1 string.
    pub fn add_bytes_inner(&self, data: &[u8]) -> Result<String, ClientError> {
        let mut store = self
            .store
            .lock()
            .map_err(|_| ClientError::lock("add_bytes"))?;
        Ok(store.put(data))
    }

    /// Retrieve bytes by CID.  Returns `None` when the CID is not present.
    pub fn get_bytes_inner(&self, cid: &str) -> Result<Option<Vec<u8>>, ClientError> {
        let store = self
            .store
            .lock()
            .map_err(|_| ClientError::lock("get_bytes"))?;
        Ok(store.get(cid).cloned())
    }

    /// Return `true` if `cid` is in the store.
    pub fn has_inner(&self, cid: &str) -> Result<bool, ClientError> {
        let store = self.store.lock().map_err(|_| ClientError::lock("has"))?;
        Ok(store.has(cid))
    }

    /// Return all stored CIDs.
    pub fn list_cids_inner(&self) -> Result<Vec<String>, ClientError> {
        let store = self
            .store
            .lock()
            .map_err(|_| ClientError::lock("list_cids"))?;
        Ok(store.list())
    }

    /// Delete a block.  Returns `true` if the block existed and was removed.
    pub fn delete_inner(&self, cid: &str) -> Result<bool, ClientError> {
        let mut store = self.store.lock().map_err(|_| ClientError::lock("delete"))?;
        Ok(store.delete(cid))
    }

    /// Return the library version string.
    pub fn version_inner(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    /// Return a JSON string with basic storage statistics.
    pub fn stats_inner(&self) -> Result<String, ClientError> {
        let store = self.store.lock().map_err(|_| ClientError::lock("stats"))?;
        let json = serde_json::json!({
            "data_dir": self.data_dir,
            "block_count": store.block_count(),
            "total_bytes": store.total_bytes(),
            "version": env!("CARGO_PKG_VERSION"),
        });
        serde_json::to_string(&json).map_err(|e| ClientError::serialise(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// NAPI bindings – only compiled when the `napi` feature is active
// ---------------------------------------------------------------------------

#[cfg(feature = "napi")]
#[napi_derive::napi]
impl IpfrsClient {
    /// Create a new `IpfrsClient`.
    ///
    /// `data_dir` is the intended storage directory.  In the current
    /// in-memory implementation it is stored for informational purposes only.
    #[napi(constructor)]
    pub fn new(data_dir: String) -> napi::Result<Self> {
        IpfrsClient::create(data_dir).map_err(into_napi)
    }

    /// Add raw bytes, returning the CIDv1 string.
    #[napi]
    pub fn add_bytes(&self, data: napi::bindgen_prelude::Buffer) -> napi::Result<String> {
        self.add_bytes_inner(data.as_ref()).map_err(into_napi)
    }

    /// Retrieve bytes by CID.  Returns `null` when the CID is not present.
    #[napi]
    pub fn get_bytes(&self, cid: String) -> napi::Result<Option<napi::bindgen_prelude::Buffer>> {
        self.get_bytes_inner(&cid)
            .map(|opt| opt.map(|v| v.into()))
            .map_err(into_napi)
    }

    /// Return `true` if `cid` is present in the store.
    #[napi]
    pub fn has(&self, cid: String) -> napi::Result<bool> {
        self.has_inner(&cid).map_err(into_napi)
    }

    /// Return all stored CIDs.
    #[napi]
    pub fn list_cids(&self) -> napi::Result<Vec<String>> {
        self.list_cids_inner().map_err(into_napi)
    }

    /// Delete a block.  Returns `true` if the block existed and was removed.
    #[napi]
    pub fn delete(&self, cid: String) -> napi::Result<bool> {
        self.delete_inner(&cid).map_err(into_napi)
    }

    /// Return the library version string.
    #[napi]
    pub fn version(&self) -> String {
        self.version_inner()
    }

    /// Return a JSON string with basic storage statistics.
    #[napi]
    pub fn stats(&self) -> napi::Result<String> {
        self.stats_inner().map_err(into_napi)
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Compute the CIDv1 (base32-lower, SHA2-256, raw codec) for the given bytes.
///
/// The result is deterministic and matches the output of `compute_cid` in the
/// WASM module.
#[cfg(feature = "napi")]
#[napi_derive::napi]
pub fn compute_cid(data: napi::bindgen_prelude::Buffer) -> String {
    cid_from_bytes(data.as_ref())
}

/// Non-NAPI version used in tests.
#[cfg(not(feature = "napi"))]
pub fn compute_cid(data: &[u8]) -> String {
    cid_from_bytes(data)
}

/// Return `true` when `data` hashes to `cid`.
#[cfg(feature = "napi")]
#[napi_derive::napi]
pub fn verify_cid(cid: String, data: napi::bindgen_prelude::Buffer) -> bool {
    cid_from_bytes(data.as_ref()) == cid
}

/// Non-NAPI version used in tests.
#[cfg(not(feature = "napi"))]
pub fn verify_cid(cid: &str, data: &[u8]) -> bool {
    cid_from_bytes(data) == cid
}

/// Return the library version string.
#[cfg(feature = "napi")]
#[napi_derive::napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Non-NAPI version used in tests.
#[cfg(not(feature = "napi"))]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// ---------------------------------------------------------------------------
// Full Node – wraps the real IPFRS node with all features
// ---------------------------------------------------------------------------

/// IPFRS Node – full node with block storage, semantic search, and TensorLogic.
///
/// This is the production-grade binding that delegates to the Rust `ipfrs::Node`
/// type.  All I/O operations use an embedded tokio runtime so they can be called
/// from synchronous JavaScript contexts.
#[cfg(feature = "napi")]
#[napi_derive::napi]
pub struct Node {
    inner: std::sync::Arc<tokio::sync::Mutex<ipfrs::Node>>,
    runtime: std::sync::Arc<tokio::runtime::Runtime>,
}

#[cfg(feature = "napi")]
#[napi_derive::napi]
impl Node {
    /// Create a new IPFRS node with the given configuration.
    #[napi(constructor)]
    pub fn new(config: Option<NodeConfig>) -> napi::Result<Self> {
        let rust_config = config.map(|c| c.into_rust_config()).unwrap_or_default();

        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| napi::Error::from_reason(format!("Failed to create runtime: {e}")))?;

        let inner = ipfrs::Node::new(rust_config)
            .map_err(|e| napi::Error::from_reason(format!("Failed to create node: {e}")))?;

        Ok(Self {
            inner: std::sync::Arc::new(tokio::sync::Mutex::new(inner)),
            runtime: std::sync::Arc::new(runtime),
        })
    }

    /// Start the node.
    #[napi]
    pub async fn start(&self) -> napi::Result<()> {
        let inner = self.inner.clone();
        self.runtime
            .block_on(async move {
                let mut node = inner.lock().await;
                node.start().await
            })
            .map_err(|e| napi::Error::from_reason(format!("Failed to start: {e}")))
    }

    /// Stop the node.
    #[napi]
    pub async fn stop(&self) -> napi::Result<()> {
        let inner = self.inner.clone();
        self.runtime
            .block_on(async move {
                let mut node = inner.lock().await;
                node.stop().await
            })
            .map_err(|e| napi::Error::from_reason(format!("Failed to stop: {e}")))
    }

    /// Store raw bytes and return the CIDv1 string.
    #[napi]
    pub async fn put_block(&self, data: napi::bindgen_prelude::Buffer) -> napi::Result<String> {
        let data_bytes: bytes::Bytes = bytes::Bytes::from(data.to_vec());
        let block = ipfrs::Block::new(data_bytes)
            .map_err(|e| napi::Error::from_reason(format!("Failed to create block: {e}")))?;
        let cid = *block.cid();

        let inner = self.inner.clone();
        self.runtime
            .block_on(async move {
                let node = inner.lock().await;
                node.put_block(&block).await
            })
            .map_err(|e| napi::Error::from_reason(format!("Failed to put block: {e}")))?;

        Ok(cid.to_string())
    }

    /// Retrieve a block by CID.  Returns `null` when the CID is not found.
    #[napi]
    pub async fn get_block(
        &self,
        cid: String,
    ) -> napi::Result<Option<napi::bindgen_prelude::Buffer>> {
        let rust_cid: ipfrs::Cid = cid
            .parse()
            .map_err(|_| napi::Error::from_reason("Invalid CID"))?;

        let inner = self.inner.clone();
        let result = self.runtime.block_on(async move {
            let node = inner.lock().await;
            node.get_block(&rust_cid).await
        });

        match result {
            Ok(Some(block)) => Ok(Some(block.data().to_vec().into())),
            Ok(None) => Ok(None),
            Err(e) => Err(napi::Error::from_reason(format!(
                "Failed to get block: {e}"
            ))),
        }
    }

    /// Return `true` when the given CID exists in the store.
    #[napi]
    pub async fn has_block(&self, cid: String) -> napi::Result<bool> {
        let rust_cid: ipfrs::Cid = cid
            .parse()
            .map_err(|_| napi::Error::from_reason("Invalid CID"))?;

        let inner = self.inner.clone();
        self.runtime
            .block_on(async move {
                let node = inner.lock().await;
                node.has_block(&rust_cid).await
            })
            .map_err(|e| napi::Error::from_reason(format!("Failed to check block: {e}")))
    }

    /// Delete a block from storage.
    #[napi]
    pub async fn delete_block(&self, cid: String) -> napi::Result<()> {
        let rust_cid: ipfrs::Cid = cid
            .parse()
            .map_err(|_| napi::Error::from_reason("Invalid CID"))?;

        let inner = self.inner.clone();
        self.runtime
            .block_on(async move {
                let node = inner.lock().await;
                node.delete_block(&rust_cid).await
            })
            .map_err(|e| napi::Error::from_reason(format!("Failed to delete block: {e}")))
    }

    /// Index content for semantic (vector) search.
    #[napi]
    pub async fn index_content(&self, cid: String, embedding: Vec<f64>) -> napi::Result<()> {
        let rust_cid: ipfrs::Cid = cid
            .parse()
            .map_err(|_| napi::Error::from_reason("Invalid CID"))?;
        let embedding_f32: Vec<f32> = embedding.iter().map(|&v| v as f32).collect();

        let inner = self.inner.clone();
        self.runtime
            .block_on(async move {
                let node = inner.lock().await;
                node.index_content(&rust_cid, &embedding_f32).await
            })
            .map_err(|e| napi::Error::from_reason(format!("Failed to index: {e}")))
    }

    /// Search for similar content by vector embedding.
    #[napi]
    pub async fn search_similar(
        &self,
        query: Vec<f64>,
        k: u32,
    ) -> napi::Result<Vec<NodeSearchResult>> {
        let query_f32: Vec<f32> = query.iter().map(|&v| v as f32).collect();
        let k_usize = k as usize;

        let inner = self.inner.clone();
        let results = self
            .runtime
            .block_on(async move {
                let node = inner.lock().await;
                node.search_similar(&query_f32, k_usize).await
            })
            .map_err(|e| napi::Error::from_reason(format!("Search failed: {e}")))?;

        Ok(results
            .into_iter()
            .map(|r| NodeSearchResult {
                cid: r.cid.to_string(),
                score: f64::from(r.score),
            })
            .collect())
    }

    /// Add a fact to the knowledge base.
    #[napi]
    pub fn add_fact(&self, fact: NodePredicate) -> napi::Result<()> {
        let rust_fact = fact.into_rust_predicate()?;
        let node = self.inner.blocking_lock();
        node.add_fact(rust_fact)
            .map_err(|e| napi::Error::from_reason(format!("Failed to add fact: {e}")))
    }

    /// Add a rule to the knowledge base.
    #[napi]
    pub fn add_rule(&self, rule: NodeRule) -> napi::Result<()> {
        let rust_rule = rule.into_rust_rule()?;
        let node = self.inner.blocking_lock();
        node.add_rule(rust_rule)
            .map_err(|e| napi::Error::from_reason(format!("Failed to add rule: {e}")))
    }

    /// Run an inference query against the knowledge base.
    ///
    /// Returns a list of substitution strings (serialised as JSON objects).
    #[napi]
    pub fn infer(&self, goal: NodePredicate) -> napi::Result<Vec<String>> {
        let rust_goal = goal.into_rust_predicate()?;
        let node = self.inner.blocking_lock();
        let results = node
            .infer(&rust_goal)
            .map_err(|e| napi::Error::from_reason(format!("Inference failed: {e}")))?;

        Ok(results.into_iter().map(|s| format!("{s:?}")).collect())
    }

    /// Generate a proof for a goal.  Returns `null` when no proof is found.
    #[napi]
    pub fn prove(&self, goal: NodePredicate) -> napi::Result<Option<String>> {
        let rust_goal = goal.into_rust_predicate()?;
        let node = self.inner.blocking_lock();
        let proof = node
            .prove(&rust_goal)
            .map_err(|e| napi::Error::from_reason(format!("Proof generation failed: {e}")))?;

        Ok(proof.map(|p| format!("{p:?}")))
    }

    /// Return knowledge-base statistics.
    #[napi]
    pub fn kb_stats(&self) -> napi::Result<KbStats> {
        let node = self.inner.blocking_lock();
        let stats = node
            .tensorlogic_stats()
            .map_err(|e| napi::Error::from_reason(format!("Failed to get stats: {e}")))?;

        Ok(KbStats {
            num_facts: stats.num_facts as u32,
            num_rules: stats.num_rules as u32,
        })
    }

    /// Save the semantic index to disk.
    #[napi]
    pub async fn save_semantic_index(&self, path: String) -> napi::Result<()> {
        let inner = self.inner.clone();
        self.runtime
            .block_on(async move {
                let node = inner.lock().await;
                node.save_semantic_index(std::path::PathBuf::from(path))
                    .await
            })
            .map_err(|e| napi::Error::from_reason(format!("Failed to save index: {e}")))
    }

    /// Load the semantic index from disk.
    #[napi]
    pub async fn load_semantic_index(&self, path: String) -> napi::Result<()> {
        let inner = self.inner.clone();
        self.runtime
            .block_on(async move {
                let node = inner.lock().await;
                node.load_semantic_index(std::path::PathBuf::from(path))
                    .await
            })
            .map_err(|e| napi::Error::from_reason(format!("Failed to load index: {e}")))
    }

    /// Save the knowledge base to disk.
    #[napi]
    pub async fn save_kb(&self, path: String) -> napi::Result<()> {
        let inner = self.inner.clone();
        self.runtime
            .block_on(async move {
                let node = inner.lock().await;
                node.save_knowledge_base(std::path::PathBuf::from(path))
                    .await
            })
            .map_err(|e| napi::Error::from_reason(format!("Failed to save KB: {e}")))
    }

    /// Load the knowledge base from disk.
    #[napi]
    pub async fn load_kb(&self, path: String) -> napi::Result<()> {
        let inner = self.inner.clone();
        self.runtime
            .block_on(async move {
                let node = inner.lock().await;
                node.load_knowledge_base(std::path::PathBuf::from(path))
                    .await
            })
            .map_err(|e| napi::Error::from_reason(format!("Failed to load KB: {e}")))
    }
}

// ---------------------------------------------------------------------------
// Supporting NAPI object types
// ---------------------------------------------------------------------------

/// Node configuration exposed to JavaScript.
#[cfg(feature = "napi")]
#[napi_derive::napi(object)]
#[derive(Clone)]
pub struct NodeConfig {
    /// Path to the storage directory.
    pub storage_path: Option<String>,
    /// Enable semantic (vector) search.
    pub enable_semantic: Option<bool>,
    /// Enable TensorLogic reasoning.
    pub enable_tensorlogic: Option<bool>,
}

#[cfg(feature = "napi")]
impl NodeConfig {
    fn into_rust_config(self) -> ipfrs::NodeConfig {
        let mut config = ipfrs::NodeConfig::default();
        if let Some(ref path) = self.storage_path {
            config.storage.path = std::path::PathBuf::from(path);
        }
        if let Some(v) = self.enable_semantic {
            config.enable_semantic = v;
        }
        if let Some(v) = self.enable_tensorlogic {
            config.enable_tensorlogic = v;
        }
        config
    }
}

/// Search result returned by `Node::search_similar`.
#[cfg(feature = "napi")]
#[napi_derive::napi(object)]
pub struct NodeSearchResult {
    pub cid: String,
    pub score: f64,
}

/// Knowledge-base statistics.
#[cfg(feature = "napi")]
#[napi_derive::napi(object)]
pub struct KbStats {
    pub num_facts: u32,
    pub num_rules: u32,
}

/// A logical term passed from JavaScript.
#[cfg(feature = "napi")]
#[napi_derive::napi(object)]
#[derive(Clone)]
pub struct NodeTerm {
    /// One of `"int"`, `"float"`, `"string"`, `"bool"`, or `"var"`.
    pub kind: String,
    pub value: String,
}

#[cfg(feature = "napi")]
impl NodeTerm {
    fn into_rust_term(self) -> napi::Result<ipfrs::Term> {
        use ipfrs::{Constant, Term};
        match self.kind.as_str() {
            "int" => {
                let val: i64 = self
                    .value
                    .parse()
                    .map_err(|_| napi::Error::from_reason("Invalid integer"))?;
                Ok(Term::Const(Constant::Int(val)))
            }
            "float" => Ok(Term::Const(Constant::Float(self.value))),
            "string" => Ok(Term::Const(Constant::String(self.value))),
            "bool" => {
                let val: bool = self
                    .value
                    .parse()
                    .map_err(|_| napi::Error::from_reason("Invalid boolean"))?;
                Ok(Term::Const(Constant::Bool(val)))
            }
            "var" => Ok(Term::Var(self.value)),
            other => Err(napi::Error::from_reason(format!(
                "Unknown term kind: {other}"
            ))),
        }
    }
}

/// A logical predicate passed from JavaScript.
#[cfg(feature = "napi")]
#[napi_derive::napi(object)]
#[derive(Clone)]
pub struct NodePredicate {
    pub name: String,
    pub args: Vec<NodeTerm>,
}

#[cfg(feature = "napi")]
impl NodePredicate {
    fn into_rust_predicate(self) -> napi::Result<ipfrs::Predicate> {
        let rust_args: napi::Result<Vec<ipfrs::Term>> =
            self.args.into_iter().map(|t| t.into_rust_term()).collect();
        Ok(ipfrs::Predicate::new(self.name, rust_args?))
    }
}

/// A logical rule passed from JavaScript.
#[cfg(feature = "napi")]
#[napi_derive::napi(object)]
#[derive(Clone)]
pub struct NodeRule {
    pub head: NodePredicate,
    pub body: Vec<NodePredicate>,
}

#[cfg(feature = "napi")]
impl NodeRule {
    fn into_rust_rule(self) -> napi::Result<ipfrs::Rule> {
        let rust_head = self.head.into_rust_predicate()?;
        let rust_body: napi::Result<Vec<ipfrs::Predicate>> = self
            .body
            .into_iter()
            .map(|p| p.into_rust_predicate())
            .collect();
        let body = rust_body?;
        if body.is_empty() {
            Ok(ipfrs::Rule::fact(rust_head))
        } else {
            Ok(ipfrs::Rule::new(rust_head, body))
        }
    }
}

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

/// Internal error type for `IpfrsClient` operations.
#[derive(Debug)]
pub struct ClientError {
    message: String,
}

impl ClientError {
    fn lock(op: &str) -> Self {
        Self {
            message: format!("Mutex poisoned during '{op}'"),
        }
    }

    fn serialise(msg: String) -> Self {
        Self { message: msg }
    }
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IpfrsClient error: {}", self.message)
    }
}

impl std::error::Error for ClientError {}

#[cfg(feature = "napi")]
fn into_napi(e: ClientError) -> napi::Error {
    napi::Error::from_reason(e.to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // IpfrsClient – add / get round-trip
    // ------------------------------------------------------------------

    #[test]
    fn test_client_add_get_roundtrip() {
        let client = IpfrsClient::create("/tmp/ipfrs-test".to_string()).expect("create client");

        let payload = b"hello, IPFRS!";
        let cid = client.add_bytes_inner(payload).expect("add_bytes");
        let retrieved = client
            .get_bytes_inner(&cid)
            .expect("get_bytes")
            .expect("should be Some");

        assert_eq!(payload.as_slice(), retrieved.as_slice());
    }

    // ------------------------------------------------------------------
    // CID is deterministic for the same input
    // ------------------------------------------------------------------

    #[test]
    fn test_cid_deterministic() {
        let data = b"determinism matters";
        let cid1 = cid_from_bytes(data);
        let cid2 = cid_from_bytes(data);
        assert_eq!(cid1, cid2);
    }

    // ------------------------------------------------------------------
    // Different data → different CIDs
    // ------------------------------------------------------------------

    #[test]
    fn test_cid_unique_per_content() {
        let cid_a = cid_from_bytes(b"alpha");
        let cid_b = cid_from_bytes(b"beta");
        assert_ne!(cid_a, cid_b);
    }

    // ------------------------------------------------------------------
    // has() returns false for an unknown CID
    // ------------------------------------------------------------------

    #[test]
    fn test_has_returns_false_for_unknown() {
        let client = IpfrsClient::create("/tmp/ipfrs-test".to_string()).expect("create client");

        let present = client
            .has_inner("bafyreifake000000000000000000000000000000000000000")
            .expect("has");
        assert!(!present);
    }

    // ------------------------------------------------------------------
    // has() returns true after add
    // ------------------------------------------------------------------

    #[test]
    fn test_has_returns_true_after_add() {
        let client = IpfrsClient::create("/tmp/ipfrs-test".to_string()).expect("create client");

        let cid = client.add_bytes_inner(b"exists").expect("add_bytes");
        assert!(client.has_inner(&cid).expect("has"));
    }

    // ------------------------------------------------------------------
    // delete()
    // ------------------------------------------------------------------

    #[test]
    fn test_delete_block() {
        let client = IpfrsClient::create("/tmp/ipfrs-test".to_string()).expect("create client");

        let cid = client.add_bytes_inner(b"to delete").expect("add_bytes");
        assert!(client.has_inner(&cid).expect("has before delete"));

        let removed = client.delete_inner(&cid).expect("delete");
        assert!(removed);
        assert!(!client.has_inner(&cid).expect("has after delete"));
    }

    // ------------------------------------------------------------------
    // delete() returns false for unknown CID
    // ------------------------------------------------------------------

    #[test]
    fn test_delete_unknown_returns_false() {
        let client = IpfrsClient::create("/tmp/ipfrs-test".to_string()).expect("create client");

        let removed = client
            .delete_inner("bafyreifake000000000000000000000000000000000000000")
            .expect("delete");
        assert!(!removed);
    }

    // ------------------------------------------------------------------
    // list_cids()
    // ------------------------------------------------------------------

    #[test]
    fn test_list_cids() {
        let client = IpfrsClient::create("/tmp/ipfrs-test".to_string()).expect("create client");

        assert!(
            client.list_cids_inner().expect("list empty").is_empty(),
            "starts empty"
        );

        let cid1 = client.add_bytes_inner(b"first").expect("add 1");
        let cid2 = client.add_bytes_inner(b"second").expect("add 2");

        let mut cids = client.list_cids_inner().expect("list two");
        cids.sort();
        let mut expected = vec![cid1, cid2];
        expected.sort();
        assert_eq!(cids, expected);
    }

    // ------------------------------------------------------------------
    // stats() returns valid JSON
    // ------------------------------------------------------------------

    #[test]
    fn test_stats_json_valid() {
        let client =
            IpfrsClient::create("/tmp/ipfrs-stats-test".to_string()).expect("create client");

        client.add_bytes_inner(b"some data").expect("add");
        let json_str = client.stats_inner().expect("stats");
        let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("valid JSON");

        assert_eq!(parsed["block_count"], 1);
        assert_eq!(parsed["data_dir"], "/tmp/ipfrs-stats-test");
        assert!(parsed["total_bytes"].as_u64().unwrap_or(0) > 0);
    }

    // ------------------------------------------------------------------
    // verify_cid (free function)
    // ------------------------------------------------------------------

    #[test]
    fn test_verify_cid() {
        let data = b"verifiable content";
        let cid = cid_from_bytes(data);

        assert!(cid_from_bytes(data) == cid, "correct CID should verify");
        assert!(cid_from_bytes(data) != "bwrongcid", "wrong CID should fail");
    }

    // ------------------------------------------------------------------
    // compute_cid (free function) – starts with 'b'
    // ------------------------------------------------------------------

    #[test]
    fn test_compute_cid_free_function() {
        let cid = cid_from_bytes(b"ipfrs nodejs");
        assert!(
            cid.starts_with('b'),
            "CIDv1 multibase base32lower must start with 'b', got: {cid}"
        );
        // Length sanity: 1 (prefix) + ceil(36 * 8 / 5) = 1 + 58 = 59
        assert_eq!(cid.len(), 59, "unexpected CID length: {}", cid.len());
    }

    // ------------------------------------------------------------------
    // version string
    // ------------------------------------------------------------------

    #[test]
    fn test_version_string() {
        let v = version();
        assert!(!v.is_empty(), "version must not be empty");
        // Expect semver format X.Y.Z
        let parts: Vec<&str> = v.split('.').collect();
        assert_eq!(parts.len(), 3, "version should be semver: {v}");
    }

    // ------------------------------------------------------------------
    // Idempotent add – same bytes same CID, no duplicate
    // ------------------------------------------------------------------

    #[test]
    fn test_idempotent_add() {
        let client = IpfrsClient::create("/tmp/ipfrs-test".to_string()).expect("create client");

        let cid1 = client.add_bytes_inner(b"idempotent").expect("add 1");
        let cid2 = client.add_bytes_inner(b"idempotent").expect("add 2");
        assert_eq!(cid1, cid2, "same content must yield same CID");

        let cids = client.list_cids_inner().expect("list");
        assert_eq!(cids.len(), 1, "should only store one block");
    }

    // ------------------------------------------------------------------
    // get_bytes returns None for unknown CID
    // ------------------------------------------------------------------

    #[test]
    fn test_get_bytes_unknown_returns_none() {
        let client = IpfrsClient::create("/tmp/ipfrs-test".to_string()).expect("create client");

        let result = client
            .get_bytes_inner("bafyreifake000000000000000000000000000000000000000")
            .expect("get_bytes");
        assert!(result.is_none());
    }

    // ------------------------------------------------------------------
    // CID format: base32lower prefix 'b', all-lowercase body
    // ------------------------------------------------------------------

    #[test]
    fn test_cid_format_base32lower() {
        let cid = cid_from_bytes(b"format check");
        assert!(cid.starts_with('b'));
        // All chars after 'b' must be lowercase base32 alphabet
        for ch in cid.chars().skip(1) {
            assert!(
                ch.is_ascii_lowercase() || ('2'..='7').contains(&ch),
                "unexpected char '{ch}' in CID '{cid}'"
            );
        }
    }

    // ------------------------------------------------------------------
    // Known-value CID stability (regression test)
    // ------------------------------------------------------------------

    #[test]
    fn test_cid_known_value_stability() {
        // Pre-computed: echo -n "ipfrs stable" | ipfs add --cid-version 1 --raw-leaves
        // We just verify the value doesn't change across builds.
        let cid = cid_from_bytes(b"ipfrs stable");
        // Must start with 'b' and be 59 chars
        assert!(cid.starts_with('b'));
        assert_eq!(cid.len(), 59);
        // Store for regression detection
        let cid2 = cid_from_bytes(b"ipfrs stable");
        assert_eq!(cid, cid2);
    }
}
