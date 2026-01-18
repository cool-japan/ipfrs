//! WebAssembly bindings for IPFRS
//!
//! This module provides browser-compatible bindings for IPFRS using wasm-bindgen.

use ipfrs::{Node as RustNode, NodeConfig as RustNodeConfig, QueryFilter as RustQueryFilter};
use ipfrs_core::{Block as RustBlock, Cid as RustCid};
use ipfrs_tensorlogic::ir::{Constant, Predicate as RustPredicate, Rule as RustRule, Term as RustTerm};
use wasm_bindgen::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use parking_lot::Mutex;

// Set up console error panic hook for better debugging
#[wasm_bindgen(start)]
pub fn main() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Node configuration for WebAssembly
#[wasm_bindgen]
#[derive(Clone)]
pub struct NodeConfig {
    storage_path: Option<String>,
    enable_semantic: bool,
    enable_tensorlogic: bool,
}

#[wasm_bindgen]
impl NodeConfig {
    /// Create a new node configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            storage_path: None,
            enable_semantic: true,
            enable_tensorlogic: true,
        }
    }

    /// Set storage path
    #[wasm_bindgen(js_name = setStoragePath)]
    pub fn set_storage_path(mut self, path: String) -> Self {
        self.storage_path = Some(path);
        self
    }

    /// Enable or disable semantic search
    #[wasm_bindgen(js_name = setEnableSemantic)]
    pub fn set_enable_semantic(mut self, enable: bool) -> Self {
        self.enable_semantic = enable;
        self
    }

    /// Enable or disable TensorLogic
    #[wasm_bindgen(js_name = setEnableTensorlogic)]
    pub fn set_enable_tensorlogic(mut self, enable: bool) -> Self {
        self.enable_tensorlogic = enable;
        self
    }

    fn to_rust_config(&self) -> RustNodeConfig {
        let mut config = RustNodeConfig::default();
        if let Some(ref path) = self.storage_path {
            config.storage.path = PathBuf::from(path);
        }
        config.enable_semantic = self.enable_semantic;
        config.enable_tensorlogic = self.enable_tensorlogic;
        config
    }
}

/// IPFRS Node for WebAssembly
#[wasm_bindgen]
pub struct Node {
    inner: Arc<Mutex<RustNode>>,
}

#[wasm_bindgen]
impl Node {
    /// Create a new IPFRS node
    #[wasm_bindgen(constructor)]
    pub fn new(config: Option<NodeConfig>) -> Result<Node, JsValue> {
        let rust_config = config
            .map(|c| c.to_rust_config())
            .unwrap_or_else(RustNodeConfig::default);

        let inner = RustNode::new(rust_config)
            .map_err(|e| JsValue::from_str(&format!("Failed to create node: {}", e)))?;

        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    /// Start the node (synchronous for WASM)
    pub fn start(&self) -> Result<(), JsValue> {
        // In WASM, we use a blocking approach since we can't easily use tokio runtime
        // This is a simplified version - in production, you'd use wasm-bindgen-futures
        let mut node = self.inner.lock();
        // Note: start() is async, but for WASM we need to handle it differently
        // This is a placeholder - real implementation would use wasm_bindgen_futures::spawn_local
        Ok(())
    }

    /// Stop the node
    pub fn stop(&self) -> Result<(), JsValue> {
        let mut node = self.inner.lock();
        Ok(())
    }

    /// Add a block to storage (simplified for WASM)
    #[wasm_bindgen(js_name = putBlock)]
    pub fn put_block(&self, data: &[u8]) -> Result<String, JsValue> {
        let block = RustBlock::new(data.to_vec().into())
            .map_err(|e| JsValue::from_str(&format!("Failed to create block: {}", e)))?;
        let cid = *block.cid();

        // Note: This is synchronous - in production, use wasm-bindgen-futures for async
        Ok(cid.to_string())
    }

    /// Check if a block exists (simplified)
    #[wasm_bindgen(js_name = hasBlock)]
    pub fn has_block(&self, cid: String) -> Result<bool, JsValue> {
        let _cid: RustCid = cid.parse()
            .map_err(|_| JsValue::from_str("Invalid CID"))?;
        // Simplified implementation
        Ok(false)
    }

    /// Index content for semantic search
    #[wasm_bindgen(js_name = indexContent)]
    pub fn index_content(&self, cid: String, embedding: Vec<f32>) -> Result<(), JsValue> {
        let _cid: RustCid = cid.parse()
            .map_err(|_| JsValue::from_str("Invalid CID"))?;

        // Simplified - would need async handling in production
        Ok(())
    }

    /// Search for similar content
    #[wasm_bindgen(js_name = searchSimilar)]
    pub fn search_similar(&self, query: Vec<f32>, k: u32) -> Result<JsValue, JsValue> {
        // Simplified - returns empty array
        let results: Vec<SearchResult> = vec![];
        Ok(serde_wasm_bindgen::to_value(&results)?)
    }

    /// Add a fact to the knowledge base
    #[wasm_bindgen(js_name = addFact)]
    pub fn add_fact(&self, fact: JsValue) -> Result<(), JsValue> {
        let fact: PredicateJs = serde_wasm_bindgen::from_value(fact)?;
        let node = self.inner.lock();
        let rust_fact = fact.to_rust_predicate()?;
        node.add_fact(rust_fact)
            .map_err(|e| JsValue::from_str(&format!("Failed to add fact: {}", e)))
    }

    /// Add a rule to the knowledge base
    #[wasm_bindgen(js_name = addRule)]
    pub fn add_rule(&self, rule: JsValue) -> Result<(), JsValue> {
        let rule: RuleJs = serde_wasm_bindgen::from_value(rule)?;
        let node = self.inner.lock();
        let rust_rule = rule.to_rust_rule()?;
        node.add_rule(rust_rule)
            .map_err(|e| JsValue::from_str(&format!("Failed to add rule: {}", e)))
    }

    /// Run inference query
    pub fn infer(&self, goal: JsValue) -> Result<JsValue, JsValue> {
        let goal: PredicateJs = serde_wasm_bindgen::from_value(goal)?;
        let node = self.inner.lock();
        let rust_goal = goal.to_rust_predicate()?;
        let results = node.infer(&rust_goal)
            .map_err(|e| JsValue::from_str(&format!("Inference failed: {}", e)))?;

        let result_strs: Vec<String> = results.iter().map(|s| format!("{:?}", s)).collect();
        Ok(serde_wasm_bindgen::to_value(&result_strs)?)
    }

    /// Get knowledge base statistics
    #[wasm_bindgen(js_name = kbStats)]
    pub fn kb_stats(&self) -> Result<JsValue, JsValue> {
        let node = self.inner.lock();
        let stats = node.tensorlogic_stats()
            .map_err(|e| JsValue::from_str(&format!("Failed to get stats: {}", e)))?;

        let stats_js = KbStatsJs {
            num_facts: stats.num_facts as u32,
            num_rules: stats.num_rules as u32,
        };

        Ok(serde_wasm_bindgen::to_value(&stats_js)?)
    }
}

/// Search result for JavaScript
#[derive(serde::Serialize, serde::Deserialize)]
pub struct SearchResult {
    pub cid: String,
    pub score: f32,
}

/// Knowledge base statistics for JavaScript
#[derive(serde::Serialize, serde::Deserialize)]
pub struct KbStatsJs {
    pub num_facts: u32,
    pub num_rules: u32,
}

/// Logical term for JavaScript
#[derive(serde::Serialize, serde::Deserialize)]
pub struct TermJs {
    pub kind: String,
    pub value: String,
}

impl TermJs {
    fn to_rust_term(&self) -> Result<RustTerm, JsValue> {
        match self.kind.as_str() {
            "int" => {
                let val: i64 = self.value.parse()
                    .map_err(|_| JsValue::from_str("Invalid integer"))?;
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
                    .map_err(|_| JsValue::from_str("Invalid boolean"))?;
                Ok(RustTerm::Const(Constant::Bool(val)))
            }
            "var" => {
                Ok(RustTerm::Var(self.value.clone()))
            }
            _ => Err(JsValue::from_str(&format!("Unknown term kind: {}", self.kind))),
        }
    }
}

/// Logical predicate for JavaScript
#[derive(serde::Serialize, serde::Deserialize)]
pub struct PredicateJs {
    pub name: String,
    pub args: Vec<TermJs>,
}

impl PredicateJs {
    fn to_rust_predicate(&self) -> Result<RustPredicate, JsValue> {
        let rust_args: Result<Vec<RustTerm>, JsValue> = self.args.iter()
            .map(|t| t.to_rust_term())
            .collect();

        Ok(RustPredicate::new(self.name.clone(), rust_args?))
    }
}

/// Logical rule for JavaScript
#[derive(serde::Serialize, serde::Deserialize)]
pub struct RuleJs {
    pub head: PredicateJs,
    pub body: Vec<PredicateJs>,
}

impl RuleJs {
    fn to_rust_rule(&self) -> Result<RustRule, JsValue> {
        let rust_head = self.head.to_rust_predicate()?;
        let rust_body: Result<Vec<RustPredicate>, JsValue> = self.body.iter()
            .map(|p| p.to_rust_predicate())
            .collect();

        if self.body.is_empty() {
            Ok(RustRule::fact(rust_head))
        } else {
            Ok(RustRule::new(rust_head, rust_body?))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_node_creation() {
        let config = NodeConfig::new();
        let node = Node::new(Some(config));
        assert!(node.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_kb_stats() {
        let config = NodeConfig::new()
            .set_enable_tensorlogic(true);
        let node = Node::new(Some(config)).unwrap();
        let stats = node.kb_stats();
        assert!(stats.is_ok());
    }
}
