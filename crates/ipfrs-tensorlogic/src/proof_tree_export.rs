//! Proof tree exporter — renders verified proof trees to multiple output formats.
//!
//! Supported formats:
//! - [`ExportFormat::Dot`] — Graphviz DOT language
//! - [`ExportFormat::Json`] — compact JSON array via serde_json
//! - [`ExportFormat::IndentedText`] — indented ASCII tree (2 spaces per depth)
//! - [`ExportFormat::EdgeList`] — `from_id -> to_id` one per line

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A single node in the exported proof tree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportNode {
    /// Unique identifier for this node.
    pub node_id: u64,
    /// Identifier of the rule applied at this node.
    pub rule_id: String,
    /// The goal (predicate / formula) proved at this node.
    pub goal: String,
    /// Depth of this node from the root (root = 0).
    pub depth: usize,
    /// Ordered list of child node IDs.
    pub children: Vec<u64>,
}

// ---------------------------------------------------------------------------
// Export format
// ---------------------------------------------------------------------------

/// Output format for [`ProofTreeExporter`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportFormat {
    /// Graphviz DOT language.
    Dot,
    /// Compact JSON array.
    Json,
    /// Indented ASCII tree (2 spaces per depth level).
    IndentedText,
    /// One `from_id -> to_id` edge per line.
    EdgeList,
}

// ---------------------------------------------------------------------------
// Export configuration
// ---------------------------------------------------------------------------

/// Configuration for [`ProofTreeExporter`].
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Target output format.
    pub format: ExportFormat,
    /// Maximum depth to include; deeper nodes are excluded when set.
    pub max_depth: Option<usize>,
    /// Whether to include the rule ID in rendered labels (default: `true`).
    pub include_rule_ids: bool,
    /// Maximum label length in characters; longer labels are truncated (default: 40).
    pub label_max_len: usize,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::IndentedText,
            max_depth: None,
            include_rule_ids: true,
            label_max_len: 40,
        }
    }
}

impl ExportConfig {
    /// Create a new config with all defaults plus the given format.
    pub fn new(format: ExportFormat) -> Self {
        Self {
            format,
            ..Self::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Exporter
// ---------------------------------------------------------------------------

/// Exports verified proof trees to multiple formats.
pub struct ProofTreeExporter {
    /// Export configuration.
    pub config: ExportConfig,
}

impl ProofTreeExporter {
    /// Create a new exporter with the given configuration.
    pub fn new(config: ExportConfig) -> Self {
        Self { config }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Export `nodes` to the string format described by `self.config`.
    ///
    /// Returns `"(empty)"` when `nodes` is empty.
    pub fn export(&self, nodes: &[ExportNode]) -> String {
        if nodes.is_empty() {
            return "(empty)".to_string();
        }

        // Check whether any nodes survive the depth filter before dispatching.
        let would_be_empty = match self.config.max_depth {
            None => false,
            Some(max) => !nodes.iter().any(|n| n.depth <= max),
        };
        if would_be_empty {
            return "(empty)".to_string();
        }

        // The format methods each apply the depth filter internally.
        match self.config.format {
            ExportFormat::Dot => self.to_dot(nodes),
            ExportFormat::Json => self.to_json(nodes),
            ExportFormat::IndentedText => self.to_indented_text(nodes),
            ExportFormat::EdgeList => self.to_edge_list(nodes),
        }
    }

    /// Render nodes as Graphviz DOT.
    pub fn to_dot(&self, nodes: &[ExportNode]) -> String {
        let filtered_owned = self.owned_filtered(nodes);
        let nodes: &[ExportNode] = &filtered_owned;

        let mut out = String::from("digraph proof {\n");

        for node in nodes {
            let label = self.make_label(node);
            // Escape quotes inside the label for DOT safety.
            let escaped = label.replace('\\', "\\\\").replace('"', "\\\"");
            out.push_str(&format!("    {} [label=\"{}\"];\n", node.node_id, escaped));
        }

        // Build a set of node ids present in the filtered slice so we only
        // emit edges between present nodes.
        let id_set: HashSet<u64> = nodes.iter().map(|n| n.node_id).collect();

        for node in nodes {
            for &child_id in &node.children {
                if id_set.contains(&child_id) {
                    out.push_str(&format!("    {} -> {};\n", node.node_id, child_id));
                }
            }
        }

        out.push('}');
        out
    }

    /// Render nodes as a compact JSON array.
    pub fn to_json(&self, nodes: &[ExportNode]) -> String {
        let filtered_owned = self.owned_filtered(nodes);
        // serde_json::to_string returns Result; fall back to a minimal
        // representation if serialization fails (should never happen here).
        serde_json::to_string(&filtered_owned).unwrap_or_else(|_| "[]".to_string())
    }

    /// Render nodes as an indented ASCII tree (DFS from root).
    pub fn to_indented_text(&self, nodes: &[ExportNode]) -> String {
        let filtered_owned = self.owned_filtered(nodes);
        let nodes: &[ExportNode] = &filtered_owned;

        if nodes.is_empty() {
            return String::new();
        }

        let root_id = Self::find_root(nodes);
        let id_map: HashMap<u64, &ExportNode> = nodes.iter().map(|n| (n.node_id, n)).collect();

        let mut out = String::new();
        let mut stack: Vec<u64> = vec![root_id];

        while let Some(id) = stack.pop() {
            if let Some(node) = id_map.get(&id) {
                let indent = " ".repeat(2 * node.depth);
                if self.config.include_rule_ids {
                    out.push_str(&format!(
                        "{}{} [rule: {}]\n",
                        indent,
                        self.truncate(&node.goal),
                        node.rule_id
                    ));
                } else {
                    out.push_str(&format!("{}{}\n", indent, self.truncate(&node.goal)));
                }
                // Push children in reverse order so they are popped in order.
                for &child_id in node.children.iter().rev() {
                    stack.push(child_id);
                }
            }
        }

        // Remove trailing newline for consistent output.
        if out.ends_with('\n') {
            out.pop();
        }
        out
    }

    /// Render nodes as a flat edge list (`parent_id -> child_id`).
    pub fn to_edge_list(&self, nodes: &[ExportNode]) -> String {
        let filtered_owned = self.owned_filtered(nodes);
        let nodes: &[ExportNode] = &filtered_owned;

        let id_set: HashSet<u64> = nodes.iter().map(|n| n.node_id).collect();

        let mut lines: Vec<String> = Vec::new();
        for node in nodes {
            for &child_id in &node.children {
                if id_set.contains(&child_id) {
                    lines.push(format!("{} -> {}", node.node_id, child_id));
                }
            }
        }

        lines.join("\n")
    }

    /// Number of nodes that pass the `max_depth` filter.
    pub fn node_count(&self, nodes: &[ExportNode]) -> usize {
        self.owned_filtered(nodes).len()
    }

    /// Total number of child edges among nodes that pass the `max_depth` filter.
    pub fn edge_count(&self, nodes: &[ExportNode]) -> usize {
        let filtered = self.owned_filtered(nodes);
        let id_set: HashSet<u64> = filtered.iter().map(|n| n.node_id).collect();
        filtered
            .iter()
            .flat_map(|n| n.children.iter())
            .filter(|&&child_id| id_set.contains(&child_id))
            .count()
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Return owned clones of nodes that pass the `max_depth` filter.
    fn owned_filtered(&self, nodes: &[ExportNode]) -> Vec<ExportNode> {
        match self.config.max_depth {
            None => nodes.to_vec(),
            Some(max) => nodes.iter().filter(|n| n.depth <= max).cloned().collect(),
        }
    }

    /// Build the display label for a node.
    fn make_label(&self, node: &ExportNode) -> String {
        if self.config.include_rule_ids {
            let combined = format!("{} ({})", node.goal, node.rule_id);
            self.truncate(&combined)
        } else {
            self.truncate(&node.goal)
        }
    }

    /// Truncate a string to at most `label_max_len` characters, appending `…`
    /// when truncation occurs.
    fn truncate(&self, s: &str) -> String {
        let max = self.config.label_max_len;
        if s.chars().count() <= max {
            s.to_string()
        } else {
            let truncated: String = s.chars().take(max.saturating_sub(1)).collect();
            format!("{}…", truncated)
        }
    }

    /// Find the root node: the node whose `node_id` does not appear in any
    /// other node's `children` list.  Falls back to `nodes[0]` when every
    /// node appears as a child (cyclic / degenerate tree).
    fn find_root(nodes: &[ExportNode]) -> u64 {
        let all_children: HashSet<u64> = nodes
            .iter()
            .flat_map(|n| n.children.iter().copied())
            .collect();

        nodes
            .iter()
            .find(|n| !all_children.contains(&n.node_id))
            .map(|n| n.node_id)
            .unwrap_or(nodes[0].node_id)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helpers for building test fixtures
    // ------------------------------------------------------------------

    fn leaf(id: u64, rule: &str, goal: &str, depth: usize) -> ExportNode {
        ExportNode {
            node_id: id,
            rule_id: rule.to_string(),
            goal: goal.to_string(),
            depth,
            children: vec![],
        }
    }

    fn parent_node(
        id: u64,
        rule: &str,
        goal: &str,
        depth: usize,
        children: Vec<u64>,
    ) -> ExportNode {
        ExportNode {
            node_id: id,
            rule_id: rule.to_string(),
            goal: goal.to_string(),
            depth,
            children,
        }
    }

    /// Single-node tree.
    fn single() -> Vec<ExportNode> {
        vec![leaf(1, "R1", "fact(a)", 0)]
    }

    /// Two-level tree: root → child.
    fn two_level() -> Vec<ExportNode> {
        vec![
            parent_node(1, "R1", "root_goal", 0, vec![2]),
            leaf(2, "R2", "child_goal", 1),
        ]
    }

    /// Three-level tree: root → child → grandchild.
    fn three_level() -> Vec<ExportNode> {
        vec![
            parent_node(1, "R1", "root_goal", 0, vec![2]),
            parent_node(2, "R2", "child_goal", 1, vec![3]),
            leaf(3, "R3", "grandchild_goal", 2),
        ]
    }

    // ------------------------------------------------------------------
    // 1. new() with config
    // ------------------------------------------------------------------
    #[test]
    fn test_new_with_config() {
        let cfg = ExportConfig::new(ExportFormat::Dot);
        let exporter = ProofTreeExporter::new(cfg);
        assert_eq!(exporter.config.format, ExportFormat::Dot);
        assert!(exporter.config.include_rule_ids);
        assert_eq!(exporter.config.label_max_len, 40);
        assert!(exporter.config.max_depth.is_none());
    }

    // ------------------------------------------------------------------
    // 2. export() returns "(empty)" for empty input
    // ------------------------------------------------------------------
    #[test]
    fn test_export_empty_returns_empty_sentinel() {
        let exporter = ProofTreeExporter::new(ExportConfig::new(ExportFormat::Dot));
        assert_eq!(exporter.export(&[]), "(empty)");
    }

    // ------------------------------------------------------------------
    // 3. to_dot: single node, no edges
    // ------------------------------------------------------------------
    #[test]
    fn test_to_dot_single_node_no_edges() {
        let exporter = ProofTreeExporter::new(ExportConfig::new(ExportFormat::Dot));
        let nodes = single();
        let dot = exporter.to_dot(&nodes);
        assert!(dot.contains("digraph proof {"));
        assert!(dot.contains("1 [label="));
        // No edge lines
        assert!(!dot.contains("->"));
    }

    // ------------------------------------------------------------------
    // 4. to_dot: parent → child edge present
    // ------------------------------------------------------------------
    #[test]
    fn test_to_dot_edge_present() {
        let exporter = ProofTreeExporter::new(ExportConfig::new(ExportFormat::Dot));
        let nodes = two_level();
        let dot = exporter.to_dot(&nodes);
        assert!(dot.contains("1 -> 2;"));
    }

    // ------------------------------------------------------------------
    // 5. to_dot: label truncated at label_max_len
    // ------------------------------------------------------------------
    #[test]
    fn test_to_dot_label_truncated() {
        let mut cfg = ExportConfig::new(ExportFormat::Dot);
        cfg.label_max_len = 10;
        cfg.include_rule_ids = false;
        let exporter = ProofTreeExporter::new(cfg);
        let nodes = vec![leaf(1, "R1", "a_very_long_goal_that_exceeds_limit", 0)];
        let dot = exporter.to_dot(&nodes);
        // Label should be truncated (≤ 10 chars + ellipsis marker inside quotes)
        // The raw goal is 35 chars; we expect truncation.
        assert!(dot.contains('…') || dot.contains("a_very_lo"));
        // Original full string should not appear
        assert!(!dot.contains("a_very_long_goal_that_exceeds_limit"));
    }

    // ------------------------------------------------------------------
    // 6. to_dot: rule_id omitted when include_rule_ids=false
    // ------------------------------------------------------------------
    #[test]
    fn test_to_dot_no_rule_ids() {
        let mut cfg = ExportConfig::new(ExportFormat::Dot);
        cfg.include_rule_ids = false;
        let exporter = ProofTreeExporter::new(cfg);
        let nodes = single();
        let dot = exporter.to_dot(&nodes);
        assert!(!dot.contains("R1"));
    }

    // ------------------------------------------------------------------
    // 7. to_json: round-trip via serde_json::from_str
    // ------------------------------------------------------------------
    #[test]
    fn test_to_json_round_trip() {
        let exporter = ProofTreeExporter::new(ExportConfig::new(ExportFormat::Json));
        let nodes = two_level();
        let json = exporter.to_json(&nodes);
        let recovered: Vec<ExportNode> = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].node_id, 1);
        assert_eq!(recovered[1].node_id, 2);
    }

    // ------------------------------------------------------------------
    // 8. to_json: node_id and goal present in output string
    // ------------------------------------------------------------------
    #[test]
    fn test_to_json_contains_fields() {
        let exporter = ProofTreeExporter::new(ExportConfig::new(ExportFormat::Json));
        let nodes = single();
        let json = exporter.to_json(&nodes);
        assert!(json.contains("node_id"));
        assert!(json.contains("goal"));
        assert!(json.contains("fact(a)"));
    }

    // ------------------------------------------------------------------
    // 9. to_indented_text: root at indent 0
    // ------------------------------------------------------------------
    #[test]
    fn test_to_indented_text_root_no_indent() {
        let exporter = ProofTreeExporter::new(ExportConfig::new(ExportFormat::IndentedText));
        let nodes = two_level();
        let text = exporter.to_indented_text(&nodes);
        // First line must start with the root goal (no leading spaces)
        let first_line = text.lines().next().expect("at least one line");
        assert!(!first_line.starts_with(' '));
        assert!(first_line.contains("root_goal"));
    }

    // ------------------------------------------------------------------
    // 10. to_indented_text: child at indent 2
    // ------------------------------------------------------------------
    #[test]
    fn test_to_indented_text_child_indent_2() {
        let exporter = ProofTreeExporter::new(ExportConfig::new(ExportFormat::IndentedText));
        let nodes = two_level();
        let text = exporter.to_indented_text(&nodes);
        let child_line = text
            .lines()
            .find(|l| l.contains("child_goal"))
            .expect("child line present");
        assert!(
            child_line.starts_with("  "),
            "expected 2-space indent, got: {:?}",
            child_line
        );
    }

    // ------------------------------------------------------------------
    // 11. to_indented_text: grandchild at indent 4
    // ------------------------------------------------------------------
    #[test]
    fn test_to_indented_text_grandchild_indent_4() {
        let exporter = ProofTreeExporter::new(ExportConfig::new(ExportFormat::IndentedText));
        let nodes = three_level();
        let text = exporter.to_indented_text(&nodes);
        let gc_line = text
            .lines()
            .find(|l| l.contains("grandchild_goal"))
            .expect("grandchild line present");
        assert!(
            gc_line.starts_with("    "),
            "expected 4-space indent, got: {:?}",
            gc_line
        );
    }

    // ------------------------------------------------------------------
    // 12. to_edge_list: "X -> Y" format
    // ------------------------------------------------------------------
    #[test]
    fn test_to_edge_list_format() {
        let exporter = ProofTreeExporter::new(ExportConfig::new(ExportFormat::EdgeList));
        let nodes = two_level();
        let list = exporter.to_edge_list(&nodes);
        assert!(list.contains("1 -> 2"), "output was: {}", list);
    }

    // ------------------------------------------------------------------
    // 13. to_edge_list: empty for no-child nodes
    // ------------------------------------------------------------------
    #[test]
    fn test_to_edge_list_empty_for_leaf_only() {
        let exporter = ProofTreeExporter::new(ExportConfig::new(ExportFormat::EdgeList));
        let nodes = single();
        let list = exporter.to_edge_list(&nodes);
        assert!(list.is_empty(), "expected empty edge list, got: {:?}", list);
    }

    // ------------------------------------------------------------------
    // 14. max_depth filters out deep nodes
    // ------------------------------------------------------------------
    #[test]
    fn test_max_depth_filters_deep_nodes() {
        let mut cfg = ExportConfig::new(ExportFormat::IndentedText);
        cfg.max_depth = Some(1);
        let exporter = ProofTreeExporter::new(cfg);
        let nodes = three_level(); // depths 0, 1, 2
        let text = exporter.to_indented_text(&nodes);
        assert!(
            !text.contains("grandchild_goal"),
            "depth-2 node should be filtered"
        );
        assert!(text.contains("child_goal"), "depth-1 node should remain");
    }

    // ------------------------------------------------------------------
    // 15. node_count respects max_depth
    // ------------------------------------------------------------------
    #[test]
    fn test_node_count_with_max_depth() {
        let mut cfg = ExportConfig::new(ExportFormat::Dot);
        cfg.max_depth = Some(1);
        let exporter = ProofTreeExporter::new(cfg);
        let nodes = three_level(); // 3 nodes at depths 0, 1, 2
        assert_eq!(exporter.node_count(&nodes), 2); // only depths 0 and 1
    }

    // ------------------------------------------------------------------
    // 16. edge_count correct
    // ------------------------------------------------------------------
    #[test]
    fn test_edge_count_correct() {
        let exporter = ProofTreeExporter::new(ExportConfig::new(ExportFormat::Dot));
        let nodes = three_level(); // 2 edges: 1→2 and 2→3
        assert_eq!(exporter.edge_count(&nodes), 2);
    }

    // ------------------------------------------------------------------
    // Bonus: edge_count with max_depth cuts the deeper edge
    // ------------------------------------------------------------------
    #[test]
    fn test_edge_count_with_max_depth() {
        let mut cfg = ExportConfig::new(ExportFormat::Dot);
        cfg.max_depth = Some(1);
        let exporter = ProofTreeExporter::new(cfg);
        let nodes = three_level();
        // Node 3 (depth 2) filtered out → only 1→2 edge remains
        assert_eq!(exporter.edge_count(&nodes), 1);
    }

    // ------------------------------------------------------------------
    // Bonus: export dispatches to edge_list format correctly
    // ------------------------------------------------------------------
    #[test]
    fn test_export_dispatches_edge_list() {
        let exporter = ProofTreeExporter::new(ExportConfig::new(ExportFormat::EdgeList));
        let result = exporter.export(&two_level());
        assert!(result.contains("->"));
    }

    // ------------------------------------------------------------------
    // Bonus: export dispatches to json format correctly
    // ------------------------------------------------------------------
    #[test]
    fn test_export_dispatches_json() {
        let exporter = ProofTreeExporter::new(ExportConfig::new(ExportFormat::Json));
        let result = exporter.export(&single());
        assert!(result.starts_with('['));
    }
}
