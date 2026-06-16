//! Einsum expression parser and dependency-aware computation graph.

use ipfrs_core::{Cid, Result};
use std::collections::HashMap;

use super::streaming::TensorMetadata;

/// Einsum expression parser and dependency resolver
///
/// Parses einsum expressions to identify tensor dependencies and optimal fetch order
#[derive(Debug, Clone)]
pub struct EinsumExpression {
    /// Raw einsum string (e.g., "ij,jk->ik")
    pub expression: String,
    /// Input tensor identifiers
    pub inputs: Vec<String>,
    /// Output tensor identifier
    pub output: String,
}

impl EinsumExpression {
    /// Parse an einsum expression
    ///
    /// Format: "input1_indices,input2_indices->output_indices"
    /// Example: "ij,jk->ik" for matrix multiplication
    pub fn parse(expression: impl Into<String>) -> Result<Self> {
        let expr = expression.into();

        // Split on "->"
        let parts: Vec<&str> = expr.split("->").collect();
        if parts.len() != 2 {
            return Err(ipfrs_core::error::Error::InvalidInput(
                "Invalid einsum expression: missing '->'".to_string(),
            ));
        }

        let inputs_str = parts[0];
        let output_str = parts[1];

        // Parse inputs (comma-separated)
        let inputs: Vec<String> = inputs_str
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();

        if inputs.is_empty() {
            return Err(ipfrs_core::error::Error::InvalidInput(
                "Invalid einsum expression: no inputs".to_string(),
            ));
        }

        let output = output_str.trim().to_string();

        Ok(Self {
            expression: expr,
            inputs,
            output,
        })
    }

    /// Get number of input tensors
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Check if this is a reduction operation
    pub fn is_reduction(&self) -> bool {
        // If output has fewer indices than any input, it's a reduction
        let output_len = self.output.len();
        self.inputs.iter().any(|input| input.len() > output_len)
    }

    /// Check if this is a transpose operation
    pub fn is_transpose(&self) -> bool {
        self.inputs.len() == 1 && self.output.len() == self.inputs[0].len()
    }

    /// Get shared indices between inputs (for contraction)
    pub fn shared_indices(&self) -> Vec<char> {
        if self.inputs.len() < 2 {
            return Vec::new();
        }

        let first: std::collections::HashSet<char> = self.inputs[0].chars().collect();
        let second: std::collections::HashSet<char> = self.inputs[1].chars().collect();

        first.intersection(&second).copied().collect()
    }
}

/// Einsum computation graph
///
/// Manages dependencies and scheduling for einsum operations
#[derive(Debug)]
pub struct EinsumGraph {
    /// Einsum expressions to execute
    expressions: Vec<EinsumExpression>,
    /// Tensor name to CID mapping
    tensor_cids: HashMap<String, Cid>,
    /// Dependency graph: output -> inputs
    dependencies: HashMap<String, Vec<String>>,
}

impl EinsumGraph {
    /// Create a new einsum graph
    pub fn new() -> Self {
        Self {
            expressions: Vec::new(),
            tensor_cids: HashMap::new(),
            dependencies: HashMap::new(),
        }
    }

    /// Add an einsum expression to the graph
    pub fn add_expression(&mut self, expr: EinsumExpression) {
        // Record dependencies
        self.dependencies
            .insert(expr.output.clone(), expr.inputs.clone());
        self.expressions.push(expr);
    }

    /// Register a tensor CID
    pub fn register_tensor(&mut self, name: impl Into<String>, cid: Cid) {
        self.tensor_cids.insert(name.into(), cid);
    }

    /// Get all tensor CIDs
    pub fn tensor_cids(&self) -> &HashMap<String, Cid> {
        &self.tensor_cids
    }

    /// Get dependencies for a tensor
    pub fn get_dependencies(&self, tensor_name: &str) -> Option<Vec<Cid>> {
        let dep_names = self.dependencies.get(tensor_name)?;
        let cids: Option<Vec<Cid>> = dep_names
            .iter()
            .map(|name| self.tensor_cids.get(name).copied())
            .collect();
        cids
    }

    /// Compute topological order for tensor fetching
    ///
    /// Returns tensors in dependency order (leaves first)
    pub fn topological_order(&self) -> Result<Vec<(String, Cid)>> {
        let mut visited = std::collections::HashSet::new();
        let mut order = Vec::new();

        // Helper for DFS
        fn visit(
            node: &str,
            dependencies: &HashMap<String, Vec<String>>,
            tensor_cids: &HashMap<String, Cid>,
            visited: &mut std::collections::HashSet<String>,
            order: &mut Vec<(String, Cid)>,
        ) -> Result<()> {
            if visited.contains(node) {
                return Ok(());
            }

            visited.insert(node.to_string());

            // Visit dependencies first
            if let Some(deps) = dependencies.get(node) {
                for dep in deps {
                    visit(dep, dependencies, tensor_cids, visited, order)?;
                }
            }

            // Add this node
            if let Some(cid) = tensor_cids.get(node) {
                order.push((node.to_string(), *cid));
            }

            Ok(())
        }

        // Visit all tensors
        for tensor_name in self.tensor_cids.keys() {
            visit(
                tensor_name,
                &self.dependencies,
                &self.tensor_cids,
                &mut visited,
                &mut order,
            )?;
        }

        Ok(order)
    }

    /// Get priority for a tensor based on its position in the computation graph
    ///
    /// Leaf tensors (no dependencies) get highest priority
    pub fn compute_priority(&self, tensor_name: &str) -> i32 {
        let depth = self.compute_depth(tensor_name);
        // Invert depth so leaves get higher priority
        1000 - (depth as i32 * 100)
    }

    /// Compute depth of a tensor in the dependency graph
    fn compute_depth(&self, tensor_name: &str) -> usize {
        if let Some(deps) = self.dependencies.get(tensor_name) {
            if deps.is_empty() {
                return 0;
            }
            let max_dep_depth = deps
                .iter()
                .map(|d| self.compute_depth(d))
                .max()
                .unwrap_or(0);
            max_dep_depth + 1
        } else {
            0 // Leaf node
        }
    }

    /// Generate TensorMetadata with dependencies
    pub fn generate_metadata(&self, tensor_name: &str) -> Option<TensorMetadata> {
        let cid = *self.tensor_cids.get(tensor_name)?;
        let deps = self.get_dependencies(tensor_name).unwrap_or_default();
        let priority = self.compute_priority(tensor_name);

        Some(
            TensorMetadata::new(cid)
                .with_dependencies(deps)
                .with_priority_hint(priority),
        )
    }
}

impl Default for EinsumGraph {
    fn default() -> Self {
        Self::new()
    }
}
