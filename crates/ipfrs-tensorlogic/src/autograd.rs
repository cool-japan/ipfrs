//! Reverse-mode automatic differentiation (autograd) for scalar-output
//! functions over f64 tensor operations.
//!
//! This module tracks computational graphs and computes gradients via
//! backpropagation.

use std::collections::HashMap;

/// Unique identifier for a node in the autograd graph.
pub type NodeId = u64;

/// Describes the operation that produced an [`AutogradNode`].
#[derive(Clone, Debug, PartialEq)]
pub enum AutogradOp {
    /// Leaf node — no parents (input or constant).
    Input,
    /// Element-wise addition: `z = x + y`.
    Add { lhs: NodeId, rhs: NodeId },
    /// Element-wise multiplication: `z = x * y`.
    Mul { lhs: NodeId, rhs: NodeId },
    /// Negation: `z = -x`.
    Neg { input: NodeId },
    /// Natural exponential: `z = e^x`.
    Exp { input: NodeId },
    /// Natural logarithm: `z = ln(x)`.
    Ln { input: NodeId },
    /// Power with constant exponent: `z = x^e`.
    Pow { base: NodeId, exponent: f64 },
}

/// A single node in the autograd computational graph.
#[derive(Clone, Debug)]
pub struct AutogradNode {
    /// Unique identifier.
    pub id: NodeId,
    /// Operation that produced this node.
    pub op: AutogradOp,
    /// Forward-pass value.
    pub value: f64,
    /// Accumulated gradient (populated after `backward`).
    pub grad: f64,
    /// Whether gradients flow through this node.
    pub requires_grad: bool,
}

/// Reverse-mode automatic differentiation graph.
///
/// Build a computational graph by calling [`AutogradGraph::input`],
/// [`AutogradGraph::add`], [`AutogradGraph::mul`], etc., then call
/// [`AutogradGraph::backward`] on the scalar output node to populate
/// `.grad` on every node that has `requires_grad = true`.
pub struct AutogradGraph {
    /// All nodes keyed by their [`NodeId`].
    pub nodes: HashMap<NodeId, AutogradNode>,
    next_id: NodeId,
}

impl AutogradGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn alloc_id(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn insert(&mut self, op: AutogradOp, value: f64, requires_grad: bool) -> NodeId {
        let id = self.alloc_id();
        self.nodes.insert(
            id,
            AutogradNode {
                id,
                op,
                value,
                grad: 0.0,
                requires_grad,
            },
        );
        id
    }

    // -----------------------------------------------------------------------
    // Node constructors
    // -----------------------------------------------------------------------

    /// Create a leaf input node with `requires_grad = true`.
    pub fn input(&mut self, value: f64) -> NodeId {
        self.insert(AutogradOp::Input, value, true)
    }

    /// Create a constant leaf node with `requires_grad = false`.
    /// Gradients do not flow through constants.
    pub fn constant(&mut self, value: f64) -> NodeId {
        self.insert(AutogradOp::Input, value, false)
    }

    /// `z = lhs + rhs`.
    pub fn add(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let value = self.nodes[&lhs].value + self.nodes[&rhs].value;
        self.insert(AutogradOp::Add { lhs, rhs }, value, true)
    }

    /// `z = lhs * rhs`.
    pub fn mul(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let value = self.nodes[&lhs].value * self.nodes[&rhs].value;
        self.insert(AutogradOp::Mul { lhs, rhs }, value, true)
    }

    /// `z = -input`.
    pub fn neg(&mut self, input: NodeId) -> NodeId {
        let value = -self.nodes[&input].value;
        self.insert(AutogradOp::Neg { input }, value, true)
    }

    /// `z = e^input`.
    pub fn exp(&mut self, input: NodeId) -> NodeId {
        let value = self.nodes[&input].value.exp();
        self.insert(AutogradOp::Exp { input }, value, true)
    }

    /// `z = ln(input)`.
    pub fn ln(&mut self, input: NodeId) -> NodeId {
        let value = self.nodes[&input].value.ln();
        self.insert(AutogradOp::Ln { input }, value, true)
    }

    /// `z = base^exponent` where `exponent` is a scalar constant.
    pub fn pow(&mut self, base: NodeId, exponent: f64) -> NodeId {
        let value = self.nodes[&base].value.powf(exponent);
        self.insert(AutogradOp::Pow { base, exponent }, value, true)
    }

    // -----------------------------------------------------------------------
    // Backpropagation
    // -----------------------------------------------------------------------

    /// Run reverse-mode backpropagation starting from `output_id`.
    ///
    /// Sets `nodes[output_id].grad = 1.0`, performs a topological sort
    /// (reverse post-order DFS), and propagates gradients backwards through
    /// the graph.  Only nodes with `requires_grad = true` accumulate gradients.
    pub fn backward(&mut self, output_id: NodeId) {
        // Set seed gradient on the output.
        if let Some(node) = self.nodes.get_mut(&output_id) {
            node.grad = 1.0;
        }

        // Topological order via iterative post-order DFS.
        let topo = self.topo_sort(output_id);

        // Propagate in reverse topological order (from output → inputs).
        for &nid in topo.iter().rev() {
            // Gather what we need without holding a borrow on `self.nodes`.
            let (op, grad_out, out_val, rg) = {
                let node = match self.nodes.get(&nid) {
                    Some(n) => n,
                    None => continue,
                };
                (node.op.clone(), node.grad, node.value, node.requires_grad)
            };

            if !rg {
                continue;
            }

            match op {
                AutogradOp::Input => {
                    // Leaf — nothing to propagate further.
                }
                AutogradOp::Add { lhs, rhs } => {
                    // dz/dlhs = 1, dz/drhs = 1
                    self.accumulate_grad(lhs, grad_out);
                    self.accumulate_grad(rhs, grad_out);
                }
                AutogradOp::Mul { lhs, rhs } => {
                    // dz/dlhs = rhs_val, dz/drhs = lhs_val
                    let lhs_val = self.nodes[&lhs].value;
                    let rhs_val = self.nodes[&rhs].value;
                    self.accumulate_grad(lhs, grad_out * rhs_val);
                    self.accumulate_grad(rhs, grad_out * lhs_val);
                }
                AutogradOp::Neg { input } => {
                    // dz/dinput = -1
                    self.accumulate_grad(input, -grad_out);
                }
                AutogradOp::Exp { input } => {
                    // dz/dinput = e^input = out_val
                    self.accumulate_grad(input, grad_out * out_val);
                }
                AutogradOp::Ln { input } => {
                    // dz/dinput = 1 / input_val
                    let input_val = self.nodes[&input].value;
                    self.accumulate_grad(input, grad_out / input_val);
                }
                AutogradOp::Pow { base, exponent } => {
                    // dz/dbase = exponent * base^(exponent - 1)
                    let base_val = self.nodes[&base].value;
                    let grad = grad_out * exponent * base_val.powf(exponent - 1.0);
                    self.accumulate_grad(base, grad);
                }
            }
        }
    }

    /// Accumulate `delta` into `node_id.grad` only if the node exists and
    /// has `requires_grad = true`.
    fn accumulate_grad(&mut self, node_id: NodeId, delta: f64) {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            if node.requires_grad {
                node.grad += delta;
            }
        }
    }

    /// Iterative post-order DFS to produce a topological ordering of nodes
    /// reachable from `start`.
    fn topo_sort(&self, start: NodeId) -> Vec<NodeId> {
        let mut order: Vec<NodeId> = Vec::new();
        let mut visited: HashMap<NodeId, bool> = HashMap::new(); // false = in-stack, true = done
        let mut stack: Vec<(NodeId, bool)> = vec![(start, false)];

        while let Some((nid, processed)) = stack.pop() {
            if processed {
                order.push(nid);
                continue;
            }
            if visited.contains_key(&nid) {
                continue; // already processed
            }
            // Mark as in-progress and push "return" frame.
            visited.insert(nid, false);
            stack.push((nid, true)); // post-process frame

            // Push children (parents in the graph sense).
            if let Some(node) = self.nodes.get(&nid) {
                match &node.op {
                    AutogradOp::Input => {}
                    AutogradOp::Add { lhs, rhs } => {
                        if !visited.contains_key(lhs) {
                            stack.push((*lhs, false));
                        }
                        if !visited.contains_key(rhs) {
                            stack.push((*rhs, false));
                        }
                    }
                    AutogradOp::Mul { lhs, rhs } => {
                        if !visited.contains_key(lhs) {
                            stack.push((*lhs, false));
                        }
                        if !visited.contains_key(rhs) {
                            stack.push((*rhs, false));
                        }
                    }
                    AutogradOp::Neg { input }
                    | AutogradOp::Exp { input }
                    | AutogradOp::Ln { input } => {
                        if !visited.contains_key(input) {
                            stack.push((*input, false));
                        }
                    }
                    AutogradOp::Pow { base, .. } => {
                        if !visited.contains_key(base) {
                            stack.push((*base, false));
                        }
                    }
                }
            }
        }
        order
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Return the gradient of `node_id` if it exists and has
    /// `requires_grad = true`.
    pub fn grad(&self, node_id: NodeId) -> Option<f64> {
        self.nodes
            .get(&node_id)
            .and_then(|n| if n.requires_grad { Some(n.grad) } else { None })
    }

    /// Return the forward-pass value of `node_id`, or `None` if not found.
    pub fn value(&self, node_id: NodeId) -> Option<f64> {
        self.nodes.get(&node_id).map(|n| n.value)
    }

    /// Reset all accumulated gradients to `0.0`.
    pub fn zero_grad(&mut self) {
        for node in self.nodes.values_mut() {
            node.grad = 0.0;
        }
    }
}

impl Default for AutogradGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    // ------------------------------------------------------------------
    // Forward pass
    // ------------------------------------------------------------------

    #[test]
    fn test_input_node_value() {
        let mut g = AutogradGraph::new();
        let x = g.input(std::f64::consts::PI);
        assert!(approx_eq(
            g.value(x).expect("test: should succeed"),
            std::f64::consts::PI
        ));
    }

    #[test]
    fn test_add_forward() {
        let mut g = AutogradGraph::new();
        let x = g.input(2.0);
        let y = g.input(3.0);
        let z = g.add(x, y);
        assert!(approx_eq(g.value(z).expect("test: should succeed"), 5.0));
    }

    #[test]
    fn test_mul_forward() {
        let mut g = AutogradGraph::new();
        let x = g.input(4.0);
        let y = g.input(5.0);
        let z = g.mul(x, y);
        assert!(approx_eq(g.value(z).expect("test: should succeed"), 20.0));
    }

    #[test]
    fn test_neg_forward() {
        let mut g = AutogradGraph::new();
        let x = g.input(7.0);
        let z = g.neg(x);
        assert!(approx_eq(g.value(z).expect("test: should succeed"), -7.0));
    }

    #[test]
    fn test_exp_forward() {
        let mut g = AutogradGraph::new();
        let x = g.input(1.0);
        let z = g.exp(x);
        assert!(approx_eq(
            g.value(z).expect("test: should succeed"),
            std::f64::consts::E
        ));
    }

    #[test]
    fn test_ln_forward() {
        let mut g = AutogradGraph::new();
        let x = g.input(std::f64::consts::E);
        let z = g.ln(x);
        assert!(approx_eq(g.value(z).expect("test: should succeed"), 1.0));
    }

    #[test]
    fn test_pow_forward() {
        let mut g = AutogradGraph::new();
        let x = g.input(3.0);
        let z = g.pow(x, 3.0);
        assert!(approx_eq(g.value(z).expect("test: should succeed"), 27.0));
    }

    // ------------------------------------------------------------------
    // Backward: individual ops
    // ------------------------------------------------------------------

    #[test]
    fn test_backward_add_grad_splits() {
        // z = x + y  =>  dz/dx = 1, dz/dy = 1
        let mut g = AutogradGraph::new();
        let x = g.input(2.0);
        let y = g.input(3.0);
        let z = g.add(x, y);
        g.backward(z);
        assert!(approx_eq(g.grad(x).expect("test: should succeed"), 1.0));
        assert!(approx_eq(g.grad(y).expect("test: should succeed"), 1.0));
    }

    #[test]
    fn test_backward_mul_chain_rule() {
        // z = x * y  =>  dz/dx = y, dz/dy = x
        let mut g = AutogradGraph::new();
        let x = g.input(3.0);
        let y = g.input(4.0);
        let z = g.mul(x, y);
        g.backward(z);
        assert!(approx_eq(g.grad(x).expect("test: should succeed"), 4.0));
        assert!(approx_eq(g.grad(y).expect("test: should succeed"), 3.0));
    }

    #[test]
    fn test_backward_neg_grad() {
        // z = -x  =>  dz/dx = -1
        let mut g = AutogradGraph::new();
        let x = g.input(5.0);
        let z = g.neg(x);
        g.backward(z);
        assert!(approx_eq(g.grad(x).expect("test: should succeed"), -1.0));
    }

    #[test]
    fn test_backward_exp_grad() {
        // z = e^x  =>  dz/dx = e^x = z_val
        let mut g = AutogradGraph::new();
        let x = g.input(2.0);
        let z = g.exp(x);
        g.backward(z);
        let expected = 2.0_f64.exp();
        assert!(approx_eq(
            g.grad(x).expect("test: should succeed"),
            expected
        ));
    }

    #[test]
    fn test_backward_ln_grad() {
        // z = ln(x)  =>  dz/dx = 1/x
        let mut g = AutogradGraph::new();
        let x = g.input(4.0);
        let z = g.ln(x);
        g.backward(z);
        assert!(approx_eq(g.grad(x).expect("test: should succeed"), 0.25));
    }

    #[test]
    fn test_backward_pow_grad() {
        // z = x^3  =>  dz/dx = 3x^2 = 3*4 = 12
        let mut g = AutogradGraph::new();
        let x = g.input(2.0);
        let z = g.pow(x, 3.0);
        g.backward(z);
        assert!(approx_eq(g.grad(x).expect("test: should succeed"), 12.0));
    }

    // ------------------------------------------------------------------
    // Chain rule tests
    // ------------------------------------------------------------------

    #[test]
    fn test_chain_rule_x_squared() {
        // z = x * x  =>  dz/dx = 2x = 2*3 = 6
        let mut g = AutogradGraph::new();
        let x = g.input(3.0);
        let z = g.mul(x, x);
        g.backward(z);
        assert!(approx_eq(g.grad(x).expect("test: should succeed"), 6.0));
    }

    #[test]
    fn test_chain_rule_x_plus_y_times_x() {
        // z = (x + y) * x
        // dz/dx = (x+y) + x = 2x+y; dz/dy = x
        // x=2, y=3 => dz/dx = 7, dz/dy = 2
        let mut g = AutogradGraph::new();
        let x = g.input(2.0);
        let y = g.input(3.0);
        let s = g.add(x, y);
        let z = g.mul(s, x);
        g.backward(z);
        assert!(approx_eq(g.grad(x).expect("test: should succeed"), 7.0));
        assert!(approx_eq(g.grad(y).expect("test: should succeed"), 2.0));
    }

    #[test]
    fn test_chain_rule_exp_of_mul() {
        // z = exp(x * y)
        // dz/dx = exp(x*y) * y
        // x=1, y=2 => z=e^2, dz/dx = 2*e^2
        let mut g = AutogradGraph::new();
        let x = g.input(1.0);
        let y = g.input(2.0);
        let p = g.mul(x, y);
        let z = g.exp(p);
        g.backward(z);
        let expected = 2.0 * 2.0_f64.exp();
        assert!(approx_eq(
            g.grad(x).expect("test: should succeed"),
            expected
        ));
    }

    #[test]
    fn test_chain_rule_ln_of_pow() {
        // z = ln(x^2), x=3 => z = ln(9), dz/dx = 2/x = 2/3
        let mut g = AutogradGraph::new();
        let x = g.input(3.0);
        let p = g.pow(x, 2.0);
        let z = g.ln(p);
        g.backward(z);
        assert!(approx_eq(
            g.grad(x).expect("test: should succeed"),
            2.0 / 3.0
        ));
    }

    #[test]
    fn test_chain_rule_neg_of_add() {
        // z = -(x + y), x=1, y=2 => dz/dx = -1, dz/dy = -1
        let mut g = AutogradGraph::new();
        let x = g.input(1.0);
        let y = g.input(2.0);
        let s = g.add(x, y);
        let z = g.neg(s);
        g.backward(z);
        assert!(approx_eq(g.grad(x).expect("test: should succeed"), -1.0));
        assert!(approx_eq(g.grad(y).expect("test: should succeed"), -1.0));
    }

    // ------------------------------------------------------------------
    // Constant nodes — no gradient
    // ------------------------------------------------------------------

    #[test]
    fn test_constant_node_no_grad() {
        let mut g = AutogradGraph::new();
        let c = g.constant(5.0);
        let x = g.input(3.0);
        let z = g.mul(x, c);
        g.backward(z);
        // constant node: grad should be None
        assert!(g.grad(c).is_none());
        // x should have gradient = c_val = 5.0
        assert!(approx_eq(g.grad(x).expect("test: should succeed"), 5.0));
    }

    #[test]
    fn test_constant_value_accessible() {
        let mut g = AutogradGraph::new();
        let c = g.constant(42.0);
        assert!(approx_eq(g.value(c).expect("test: should succeed"), 42.0));
    }

    // ------------------------------------------------------------------
    // zero_grad
    // ------------------------------------------------------------------

    #[test]
    fn test_zero_grad_resets() {
        let mut g = AutogradGraph::new();
        let x = g.input(2.0);
        let y = g.input(3.0);
        let z = g.add(x, y);
        g.backward(z);
        assert!(approx_eq(g.grad(x).expect("test: should succeed"), 1.0));
        g.zero_grad();
        assert!(approx_eq(g.grad(x).expect("test: should succeed"), 0.0));
        assert!(approx_eq(g.grad(y).expect("test: should succeed"), 0.0));
    }

    #[test]
    fn test_zero_grad_then_recompute() {
        let mut g = AutogradGraph::new();
        let x = g.input(3.0);
        let z = g.pow(x, 2.0);
        g.backward(z);
        g.zero_grad();
        // Run backward again — should produce correct grad
        g.backward(z);
        assert!(approx_eq(g.grad(x).expect("test: should succeed"), 6.0));
    }

    // ------------------------------------------------------------------
    // Edge cases
    // ------------------------------------------------------------------

    #[test]
    fn test_grad_none_for_nonexistent_node() {
        let g = AutogradGraph::new();
        assert!(g.grad(9999).is_none());
    }

    #[test]
    fn test_value_none_for_nonexistent_node() {
        let g = AutogradGraph::new();
        assert!(g.value(9999).is_none());
    }

    #[test]
    fn test_backward_single_input_node() {
        // backward on a raw input: grad should be 1.0
        let mut g = AutogradGraph::new();
        let x = g.input(5.0);
        g.backward(x);
        assert!(approx_eq(g.grad(x).expect("test: should succeed"), 1.0));
    }

    #[test]
    fn test_multiple_uses_accumulate_correctly() {
        // z = x * x + x  => dz/dx = 2x + 1 = 2*3+1 = 7
        let mut g = AutogradGraph::new();
        let x = g.input(3.0);
        let x2 = g.mul(x, x);
        let z = g.add(x2, x);
        g.backward(z);
        assert!(approx_eq(g.grad(x).expect("test: should succeed"), 7.0));
    }

    #[test]
    fn test_deep_chain_add() {
        // z = ((x + x) + x), x=1 => z=3, dz/dx = 3
        let mut g = AutogradGraph::new();
        let x = g.input(1.0);
        let a = g.add(x, x);
        let z = g.add(a, x);
        g.backward(z);
        assert!(approx_eq(g.grad(x).expect("test: should succeed"), 3.0));
    }

    #[test]
    fn test_pow_fractional_exponent() {
        // z = x^0.5 = sqrt(x), x=4 => dz/dx = 0.5 * x^(-0.5) = 0.25
        let mut g = AutogradGraph::new();
        let x = g.input(4.0);
        let z = g.pow(x, 0.5);
        g.backward(z);
        assert!(approx_eq(g.grad(x).expect("test: should succeed"), 0.25));
    }

    #[test]
    fn test_output_grad_is_one() {
        // The output node itself should have grad = 1.0 after backward.
        let mut g = AutogradGraph::new();
        let x = g.input(2.0);
        let z = g.exp(x);
        g.backward(z);
        assert!(approx_eq(g.nodes[&z].grad, 1.0));
    }

    #[test]
    fn test_default_graph_is_empty() {
        let g = AutogradGraph::default();
        assert!(g.nodes.is_empty());
    }

    #[test]
    fn test_autograd_op_clone_and_eq() {
        let op1 = AutogradOp::Add { lhs: 0, rhs: 1 };
        let op2 = op1.clone();
        assert_eq!(op1, op2);

        let op3 = AutogradOp::Pow {
            base: 2,
            exponent: 3.0,
        };
        assert_ne!(op1, op3);
    }

    #[test]
    fn test_ln_of_exp_identity() {
        // z = ln(exp(x)) = x => dz/dx = 1
        let mut g = AutogradGraph::new();
        let x = g.input(2.5);
        let e = g.exp(x);
        let z = g.ln(e);
        // forward: should be ≈ 2.5
        assert!((g.value(z).expect("test: should succeed") - 2.5).abs() < 1e-9);
        g.backward(z);
        assert!(approx_eq(g.grad(x).expect("test: should succeed"), 1.0));
    }
}
