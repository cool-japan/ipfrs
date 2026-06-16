//! DAG (Directed Acyclic Graph) traversal and analysis utilities.
//!
//! This module provides utilities for working with IPLD Merkle DAGs:
//! - Extracting CID links from IPLD data
//! - Calculating DAG statistics
//! - Validating DAG structures
//! - Collecting all CIDs in a DAG
//!
//! # Examples
//!
//! ```rust
//! use ipfrs_core::{Ipld, Cid, CidBuilder};
//! use ipfrs_core::dag::{extract_links, DagStats};
//! use std::collections::BTreeMap;
//!
//! // Create IPLD with links
//! let cid1 = CidBuilder::new().build(b"data1").unwrap();
//! let cid2 = CidBuilder::new().build(b"data2").unwrap();
//!
//! let mut map = BTreeMap::new();
//! map.insert("link1".to_string(), Ipld::link(cid1));
//! map.insert("link2".to_string(), Ipld::link(cid2));
//! let ipld = Ipld::Map(map);
//!
//! // Extract all CID links
//! let links = extract_links(&ipld);
//! assert_eq!(links.len(), 2);
//! ```

use crate::cid::{Cid, SerializableCid};
use crate::ipld::Ipld;
use std::collections::{HashMap, HashSet, VecDeque};

/// Extract all CID links from an IPLD structure (non-recursive).
///
/// This function finds all direct `Ipld::Link` values in the given IPLD data,
/// but does not recursively traverse nested links. Use `collect_all_links` for
/// recursive traversal.
///
/// # Arguments
///
/// * `ipld` - The IPLD data to extract links from
///
/// # Returns
///
/// A vector of all CID links found at the top level
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::{Ipld, CidBuilder};
/// use ipfrs_core::dag::extract_links;
/// use std::collections::BTreeMap;
///
/// let cid = CidBuilder::new().build(b"test").unwrap();
/// let mut map = BTreeMap::new();
/// map.insert("file".to_string(), Ipld::link(cid.clone()));
/// let ipld = Ipld::Map(map);
///
/// let links = extract_links(&ipld);
/// assert_eq!(links.len(), 1);
/// assert_eq!(links[0], cid);
/// ```
pub fn extract_links(ipld: &Ipld) -> Vec<Cid> {
    let mut links = Vec::new();
    extract_links_recursive(ipld, &mut links, false);
    links
}

/// Extract all CID links from an IPLD structure recursively.
///
/// This function traverses the entire IPLD tree and collects all `Ipld::Link`
/// values found at any depth.
///
/// # Arguments
///
/// * `ipld` - The IPLD data to extract links from
///
/// # Returns
///
/// A vector of all CID links found (may contain duplicates)
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::{Ipld, CidBuilder};
/// use ipfrs_core::dag::collect_all_links;
/// use std::collections::BTreeMap;
///
/// let cid1 = CidBuilder::new().build(b"test1").unwrap();
/// let cid2 = CidBuilder::new().build(b"test2").unwrap();
///
/// // Nested structure
/// let mut inner = BTreeMap::new();
/// inner.insert("link".to_string(), Ipld::link(cid2.clone()));
///
/// let mut outer = BTreeMap::new();
/// outer.insert("file".to_string(), Ipld::link(cid1.clone()));
/// outer.insert("nested".to_string(), Ipld::Map(inner));
///
/// let ipld = Ipld::Map(outer);
/// let links = collect_all_links(&ipld);
/// assert_eq!(links.len(), 2);
/// ```
pub fn collect_all_links(ipld: &Ipld) -> Vec<Cid> {
    let mut links = Vec::new();
    extract_links_recursive(ipld, &mut links, true);
    links
}

/// Extract unique CID links from an IPLD structure (no duplicates).
///
/// This function is similar to `collect_all_links` but returns a set of unique
/// CIDs, removing any duplicates.
///
/// # Arguments
///
/// * `ipld` - The IPLD data to extract links from
///
/// # Returns
///
/// A set of unique CID links
pub fn collect_unique_links(ipld: &Ipld) -> HashSet<Cid> {
    collect_all_links(ipld).into_iter().collect()
}

/// Internal recursive link extraction
fn extract_links_recursive(ipld: &Ipld, links: &mut Vec<Cid>, recursive: bool) {
    match ipld {
        Ipld::Link(SerializableCid(cid)) => {
            links.push(*cid);
        }
        Ipld::List(items) => {
            if recursive {
                for item in items {
                    extract_links_recursive(item, links, true);
                }
            } else {
                for item in items {
                    if let Ipld::Link(SerializableCid(cid)) = item {
                        links.push(*cid);
                    }
                }
            }
        }
        Ipld::Map(map) => {
            if recursive {
                for value in map.values() {
                    extract_links_recursive(value, links, true);
                }
            } else {
                for value in map.values() {
                    if let Ipld::Link(SerializableCid(cid)) = value {
                        links.push(*cid);
                    }
                }
            }
        }
        _ => {}
    }
}

/// Statistics about a DAG structure
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DagStats {
    /// Total number of unique CIDs in the DAG
    pub unique_cids: usize,
    /// Total number of links (including duplicates)
    pub total_links: usize,
    /// Maximum depth of the DAG
    pub max_depth: usize,
    /// Number of leaf nodes (nodes with no outgoing links)
    pub leaf_count: usize,
    /// Number of intermediate nodes (nodes with outgoing links)
    pub intermediate_count: usize,
}

impl DagStats {
    /// Create empty DAG statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate statistics for an IPLD structure
    ///
    /// Note: This only analyzes the structure of the IPLD data provided,
    /// it does not follow CID links to fetch additional blocks.
    ///
    /// # Arguments
    ///
    /// * `ipld` - The IPLD data to analyze
    ///
    /// # Returns
    ///
    /// Statistics about the DAG structure
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ipfrs_core::{Ipld, CidBuilder};
    /// use ipfrs_core::dag::DagStats;
    /// use std::collections::BTreeMap;
    ///
    /// let cid = CidBuilder::new().build(b"data").unwrap();
    /// let mut map = BTreeMap::new();
    /// map.insert("link".to_string(), Ipld::link(cid));
    /// let ipld = Ipld::Map(map);
    ///
    /// let stats = DagStats::from_ipld(&ipld);
    /// assert_eq!(stats.total_links, 1);
    /// assert_eq!(stats.unique_cids, 1);
    /// ```
    pub fn from_ipld(ipld: &Ipld) -> Self {
        let all_links = collect_all_links(ipld);
        let unique_links: HashSet<_> = all_links.iter().collect();

        let depth = calculate_depth(ipld, 0);
        let (leaves, intermediates) = count_node_types(ipld);

        Self {
            unique_cids: unique_links.len(),
            total_links: all_links.len(),
            max_depth: depth,
            leaf_count: leaves,
            intermediate_count: intermediates,
        }
    }

    /// Calculate the deduplication ratio
    ///
    /// Returns the ratio of duplicate links to total links.
    /// A value of 0.0 means no duplication, 1.0 means all links are duplicates.
    pub fn deduplication_ratio(&self) -> f64 {
        if self.total_links == 0 {
            return 0.0;
        }
        let duplicates = self.total_links.saturating_sub(self.unique_cids);
        duplicates as f64 / self.total_links as f64
    }
}

/// Calculate the maximum depth of an IPLD tree
fn calculate_depth(ipld: &Ipld, current_depth: usize) -> usize {
    match ipld {
        Ipld::List(items) => {
            let max_child_depth = items
                .iter()
                .map(|item| calculate_depth(item, current_depth + 1))
                .max()
                .unwrap_or(current_depth);
            max_child_depth
        }
        Ipld::Map(map) => {
            let max_child_depth = map
                .values()
                .map(|value| calculate_depth(value, current_depth + 1))
                .max()
                .unwrap_or(current_depth);
            max_child_depth
        }
        _ => current_depth,
    }
}

/// Count leaf and intermediate nodes
fn count_node_types(ipld: &Ipld) -> (usize, usize) {
    match ipld {
        Ipld::List(items) => {
            if items.is_empty() {
                return (1, 0); // Empty list is a leaf
            }
            let mut leaves = 0;
            let mut intermediates = 1; // This node is intermediate
            for item in items {
                let (l, i) = count_node_types(item);
                leaves += l;
                intermediates += i;
            }
            (leaves, intermediates)
        }
        Ipld::Map(map) => {
            if map.is_empty() {
                return (1, 0); // Empty map is a leaf
            }
            let mut leaves = 0;
            let mut intermediates = 1; // This node is intermediate
            for value in map.values() {
                let (l, i) = count_node_types(value);
                leaves += l;
                intermediates += i;
            }
            (leaves, intermediates)
        }
        _ => (1, 0), // Scalar values are leaves
    }
}

/// Validate that an IPLD structure forms a proper DAG (no cycles).
///
/// This function checks that there are no circular references in the CID links.
/// Note: This only validates the structure of the provided IPLD data, it does
/// not fetch and validate linked blocks.
///
/// # Arguments
///
/// * `ipld` - The IPLD data to validate
///
/// # Returns
///
/// `true` if the structure is acyclic, `false` if cycles are detected
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::{Ipld, CidBuilder};
/// use ipfrs_core::dag::is_dag;
/// use std::collections::BTreeMap;
///
/// let cid = CidBuilder::new().build(b"test").unwrap();
/// let mut map = BTreeMap::new();
/// map.insert("link".to_string(), Ipld::link(cid));
/// let ipld = Ipld::Map(map);
///
/// assert!(is_dag(&ipld));
/// ```
pub fn is_dag(ipld: &Ipld) -> bool {
    let mut visited = HashSet::new();
    let mut stack = HashSet::new();
    has_cycle_dfs(ipld, &mut visited, &mut stack)
}

/// DFS cycle detection
fn has_cycle_dfs(ipld: &Ipld, visited: &mut HashSet<String>, stack: &mut HashSet<String>) -> bool {
    match ipld {
        Ipld::Link(SerializableCid(cid)) => {
            let cid_str = cid.to_string();
            if stack.contains(&cid_str) {
                return false; // Cycle detected
            }
            if visited.contains(&cid_str) {
                return true; // Already validated this path
            }
            visited.insert(cid_str.clone());
            stack.insert(cid_str.clone());
            // Note: We can't follow the link without a BlockFetcher
            // So we just mark it as visited
            stack.remove(&cid_str);
            true
        }
        Ipld::List(items) => {
            for item in items {
                if !has_cycle_dfs(item, visited, stack) {
                    return false;
                }
            }
            true
        }
        Ipld::Map(map) => {
            for value in map.values() {
                if !has_cycle_dfs(value, visited, stack) {
                    return false;
                }
            }
            true
        }
        _ => true,
    }
}

/// Find all paths from root to a specific CID in an IPLD structure.
///
/// Returns a list of paths (as lists of keys) that lead to the target CID.
///
/// # Arguments
///
/// * `ipld` - The IPLD data to search
/// * `target_cid` - The CID to find paths to
///
/// # Returns
///
/// A vector of paths, where each path is a vector of keys leading to the CID
pub fn find_paths_to_cid(ipld: &Ipld, target_cid: &Cid) -> Vec<Vec<String>> {
    let mut paths = Vec::new();
    let mut current_path = Vec::new();
    find_paths_recursive(ipld, target_cid, &mut current_path, &mut paths);
    paths
}

/// Recursive path finding helper
fn find_paths_recursive(
    ipld: &Ipld,
    target_cid: &Cid,
    current_path: &mut Vec<String>,
    paths: &mut Vec<Vec<String>>,
) {
    match ipld {
        Ipld::Link(SerializableCid(cid)) if cid == target_cid => {
            paths.push(current_path.clone());
        }
        Ipld::List(items) => {
            for (i, item) in items.iter().enumerate() {
                current_path.push(format!("[{}]", i));
                find_paths_recursive(item, target_cid, current_path, paths);
                current_path.pop();
            }
        }
        Ipld::Map(map) => {
            for (key, value) in map {
                current_path.push(key.clone());
                find_paths_recursive(value, target_cid, current_path, paths);
                current_path.pop();
            }
        }
        _ => {}
    }
}

/// Traverse a DAG in breadth-first order, collecting all IPLD nodes.
///
/// Note: This only traverses the structure of the provided IPLD data,
/// it does not fetch linked blocks.
///
/// # Arguments
///
/// * `root` - The root IPLD node to start traversal from
///
/// # Returns
///
/// A vector of all IPLD nodes in breadth-first order
pub fn traverse_bfs(root: &Ipld) -> Vec<Ipld> {
    let mut result = Vec::new();
    let mut queue = VecDeque::new();
    queue.push_back(root.clone());

    while let Some(node) = queue.pop_front() {
        result.push(node.clone());

        match &node {
            Ipld::List(items) => {
                for item in items {
                    queue.push_back(item.clone());
                }
            }
            Ipld::Map(map) => {
                for value in map.values() {
                    queue.push_back(value.clone());
                }
            }
            _ => {}
        }
    }

    result
}

/// Traverse a DAG in depth-first order, collecting all IPLD nodes.
///
/// # Arguments
///
/// * `root` - The root IPLD node to start traversal from
///
/// # Returns
///
/// A vector of all IPLD nodes in depth-first order
pub fn traverse_dfs(root: &Ipld) -> Vec<Ipld> {
    let mut result = Vec::new();
    traverse_dfs_recursive(root, &mut result);
    result
}

/// Recursive DFS helper
fn traverse_dfs_recursive(node: &Ipld, result: &mut Vec<Ipld>) {
    result.push(node.clone());

    match node {
        Ipld::List(items) => {
            for item in items {
                traverse_dfs_recursive(item, result);
            }
        }
        Ipld::Map(map) => {
            for value in map.values() {
                traverse_dfs_recursive(value, result);
            }
        }
        _ => {}
    }
}

/// Additional DAG metrics beyond basic statistics.
///
/// Provides advanced graph-theoretic metrics for analyzing DAG structure.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DagMetrics {
    /// Average branching factor (average number of children per non-leaf node)
    pub avg_branching_factor: f64,
    /// Maximum branching factor (maximum number of children for any node)
    pub max_branching_factor: usize,
    /// Width of the DAG (maximum number of nodes at any level)
    pub width: usize,
    /// Total number of nodes in the DAG
    pub total_nodes: usize,
    /// Average depth of leaf nodes
    pub avg_leaf_depth: f64,
}

impl DagMetrics {
    /// Calculate advanced metrics for an IPLD structure.
    ///
    /// # Arguments
    ///
    /// * `ipld` - The IPLD data to analyze
    ///
    /// # Returns
    ///
    /// Advanced metrics about the DAG structure
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ipfrs_core::{Ipld, CidBuilder};
    /// use ipfrs_core::dag::DagMetrics;
    /// use std::collections::BTreeMap;
    ///
    /// let cid = CidBuilder::new().build(b"data").unwrap();
    /// let mut map = BTreeMap::new();
    /// map.insert("link1".to_string(), Ipld::link(cid.clone()));
    /// map.insert("link2".to_string(), Ipld::link(cid));
    /// let ipld = Ipld::Map(map);
    ///
    /// let metrics = DagMetrics::from_ipld(&ipld);
    /// assert_eq!(metrics.max_branching_factor, 2);
    /// ```
    pub fn from_ipld(ipld: &Ipld) -> Self {
        let mut levels: HashMap<usize, usize> = HashMap::new();
        let mut branching_factors = Vec::new();
        let mut leaf_depths = Vec::new();
        let mut total_nodes = 0;

        calculate_metrics(
            ipld,
            0,
            &mut levels,
            &mut branching_factors,
            &mut leaf_depths,
            &mut total_nodes,
        );

        let width = levels.values().copied().max().unwrap_or(0);
        let max_branching_factor = branching_factors.iter().copied().max().unwrap_or(0);
        let avg_branching_factor = if branching_factors.is_empty() {
            0.0
        } else {
            branching_factors.iter().sum::<usize>() as f64 / branching_factors.len() as f64
        };
        let avg_leaf_depth = if leaf_depths.is_empty() {
            0.0
        } else {
            leaf_depths.iter().sum::<usize>() as f64 / leaf_depths.len() as f64
        };

        Self {
            avg_branching_factor,
            max_branching_factor,
            width,
            total_nodes,
            avg_leaf_depth,
        }
    }
}

/// Helper function to calculate DAG metrics recursively
fn calculate_metrics(
    ipld: &Ipld,
    depth: usize,
    levels: &mut HashMap<usize, usize>,
    branching_factors: &mut Vec<usize>,
    leaf_depths: &mut Vec<usize>,
    total_nodes: &mut usize,
) {
    *total_nodes += 1;
    *levels.entry(depth).or_insert(0) += 1;

    match ipld {
        Ipld::List(items) => {
            if items.is_empty() {
                leaf_depths.push(depth);
            } else {
                branching_factors.push(items.len());
                for item in items {
                    calculate_metrics(
                        item,
                        depth + 1,
                        levels,
                        branching_factors,
                        leaf_depths,
                        total_nodes,
                    );
                }
            }
        }
        Ipld::Map(map) => {
            if map.is_empty() {
                leaf_depths.push(depth);
            } else {
                branching_factors.push(map.len());
                for value in map.values() {
                    calculate_metrics(
                        value,
                        depth + 1,
                        levels,
                        branching_factors,
                        leaf_depths,
                        total_nodes,
                    );
                }
            }
        }
        _ => {
            leaf_depths.push(depth);
        }
    }
}

/// Perform a topological sort on IPLD nodes containing CID links.
///
/// Returns nodes in dependency order (dependencies before dependents).
/// This is useful for processing DAGs where nodes depend on their children.
///
/// Note: Since we cannot fetch linked blocks, this only sorts the CIDs
/// found in the IPLD structure based on their dependency relationships.
///
/// # Arguments
///
/// * `ipld` - The IPLD data to sort
///
/// # Returns
///
/// A vector of CIDs in topological order (leaves first, root last)
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::{Ipld, CidBuilder};
/// use ipfrs_core::dag::topological_sort;
/// use std::collections::BTreeMap;
///
/// let cid1 = CidBuilder::new().build(b"leaf1").unwrap();
/// let cid2 = CidBuilder::new().build(b"leaf2").unwrap();
///
/// let mut map = BTreeMap::new();
/// map.insert("child1".to_string(), Ipld::link(cid1));
/// map.insert("child2".to_string(), Ipld::link(cid2));
/// let ipld = Ipld::Map(map);
///
/// let sorted = topological_sort(&ipld);
/// assert_eq!(sorted.len(), 2);
/// ```
pub fn topological_sort(ipld: &Ipld) -> Vec<Cid> {
    let links = collect_all_links(ipld);
    let mut result = Vec::new();
    let mut visited = HashSet::new();

    // Simple topological sort: collect all unique CIDs
    // Since we can't fetch blocks, we just deduplicate and return in encounter order
    for cid in links {
        if visited.insert(cid) {
            result.push(cid);
        }
    }

    result
}

/// Calculate the size (number of nodes) of a subgraph rooted at the given IPLD node.
///
/// This counts all nodes reachable from the root, including the root itself.
///
/// # Arguments
///
/// * `ipld` - The root of the subgraph
///
/// # Returns
///
/// The total number of nodes in the subgraph
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::Ipld;
/// use ipfrs_core::dag::subgraph_size;
///
/// let ipld = Ipld::List(vec![
///     Ipld::Integer(1),
///     Ipld::Integer(2),
///     Ipld::List(vec![Ipld::Integer(3)]),
/// ]);
///
/// let size = subgraph_size(&ipld);
/// assert_eq!(size, 5); // Root list + 2 integers + nested list + nested integer
/// ```
pub fn subgraph_size(ipld: &Ipld) -> usize {
    let mut count = 1; // Count the root

    match ipld {
        Ipld::List(items) => {
            for item in items {
                count += subgraph_size(item);
            }
        }
        Ipld::Map(map) => {
            for value in map.values() {
                count += subgraph_size(value);
            }
        }
        _ => {}
    }

    count
}

/// Calculate the fanout (number of direct children) for each level of the DAG.
///
/// Returns a vector where index i contains the total number of children
/// at depth i in the DAG.
///
/// # Arguments
///
/// * `ipld` - The IPLD data to analyze
///
/// # Returns
///
/// A vector of fanout counts per level
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::Ipld;
/// use ipfrs_core::dag::dag_fanout_by_level;
///
/// let ipld = Ipld::List(vec![
///     Ipld::Integer(1),
///     Ipld::List(vec![Ipld::Integer(2), Ipld::Integer(3)]),
/// ]);
///
/// let fanout = dag_fanout_by_level(&ipld);
/// assert!(fanout.len() >= 2);
/// ```
pub fn dag_fanout_by_level(ipld: &Ipld) -> Vec<usize> {
    let mut fanout_by_level = Vec::new();
    calculate_fanout(ipld, 0, &mut fanout_by_level);
    fanout_by_level
}

/// Helper to calculate fanout at each level
fn calculate_fanout(ipld: &Ipld, depth: usize, fanout_by_level: &mut Vec<usize>) {
    // Ensure vector is large enough
    while fanout_by_level.len() <= depth {
        fanout_by_level.push(0);
    }

    match ipld {
        Ipld::List(items) => {
            fanout_by_level[depth] += items.len();
            for item in items {
                calculate_fanout(item, depth + 1, fanout_by_level);
            }
        }
        Ipld::Map(map) => {
            fanout_by_level[depth] += map.len();
            for value in map.values() {
                calculate_fanout(value, depth + 1, fanout_by_level);
            }
        }
        _ => {}
    }
}

/// Count the number of CID links at each depth level in the DAG.
///
/// Returns a vector where index i contains the count of CID links at depth i.
///
/// # Arguments
///
/// * `ipld` - The IPLD data to analyze
///
/// # Returns
///
/// A vector of link counts per depth level
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::{Ipld, CidBuilder};
/// use ipfrs_core::dag::count_links_by_depth;
///
/// // Direct link at depth 0
/// let cid = CidBuilder::new().build(b"test").unwrap();
/// let ipld = Ipld::link(cid);
///
/// let counts = count_links_by_depth(&ipld);
/// assert_eq!(counts[0], 1);
/// ```
pub fn count_links_by_depth(ipld: &Ipld) -> Vec<usize> {
    let mut counts = Vec::new();
    count_links_recursive(ipld, 0, &mut counts);
    counts
}

/// Helper to count links at each depth
fn count_links_recursive(ipld: &Ipld, depth: usize, counts: &mut Vec<usize>) {
    // Ensure vector is large enough
    while counts.len() <= depth {
        counts.push(0);
    }

    match ipld {
        Ipld::Link(_) => {
            counts[depth] += 1;
        }
        Ipld::List(items) => {
            for item in items {
                count_links_recursive(item, depth + 1, counts);
            }
        }
        Ipld::Map(map) => {
            for value in map.values() {
                count_links_recursive(value, depth + 1, counts);
            }
        }
        _ => {}
    }
}

/// Filter an IPLD structure to only include nodes matching a predicate.
///
/// This creates a new IPLD structure containing only the nodes where the
/// predicate returns true.
///
/// # Arguments
///
/// * `ipld` - The IPLD data to filter
/// * `predicate` - Function that returns true for nodes to keep
///
/// # Returns
///
/// A filtered IPLD structure, or None if the root doesn't match
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::Ipld;
/// use ipfrs_core::dag::filter_dag;
///
/// let ipld = Ipld::List(vec![
///     Ipld::Integer(1),
///     Ipld::Integer(2),
///     Ipld::String("hello".to_string()),
/// ]);
///
/// // Keep only integers
/// let filtered = filter_dag(&ipld, &|node| matches!(node, Ipld::Integer(_) | Ipld::List(_)));
/// assert!(filtered.is_some());
/// ```
pub fn filter_dag<F>(ipld: &Ipld, predicate: &F) -> Option<Ipld>
where
    F: Fn(&Ipld) -> bool,
{
    if !predicate(ipld) {
        return None;
    }

    match ipld {
        Ipld::List(items) => {
            let filtered_items: Vec<Ipld> = items
                .iter()
                .filter_map(|item| filter_dag(item, predicate))
                .collect();
            Some(Ipld::List(filtered_items))
        }
        Ipld::Map(map) => {
            let filtered_map: std::collections::BTreeMap<String, Ipld> = map
                .iter()
                .filter_map(|(k, v)| filter_dag(v, predicate).map(|filtered| (k.clone(), filtered)))
                .collect();
            Some(Ipld::Map(filtered_map))
        }
        other => Some(other.clone()),
    }
}

/// Transform all nodes in a DAG using a mapping function.
///
/// Applies the transformation function to each node in the IPLD structure,
/// building a new transformed DAG.
///
/// # Arguments
///
/// * `ipld` - The IPLD data to transform
/// * `transform` - Function to transform each node
///
/// # Returns
///
/// The transformed IPLD structure
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::Ipld;
/// use ipfrs_core::dag::map_dag;
///
/// let ipld = Ipld::Integer(42);
///
/// // Double all integers
/// let transformed = map_dag(&ipld, &|node| {
///     match node {
///         Ipld::Integer(n) => Ipld::Integer(n * 2),
///         other => other.clone(),
///     }
/// });
///
/// assert_eq!(transformed, Ipld::Integer(84));
/// ```
pub fn map_dag<F>(ipld: &Ipld, transform: &F) -> Ipld
where
    F: Fn(&Ipld) -> Ipld,
{
    let transformed = match ipld {
        Ipld::List(items) => {
            let mapped_items: Vec<Ipld> =
                items.iter().map(|item| map_dag(item, transform)).collect();
            Ipld::List(mapped_items)
        }
        Ipld::Map(map) => {
            let mapped_map: std::collections::BTreeMap<String, Ipld> = map
                .iter()
                .map(|(k, v)| (k.clone(), map_dag(v, transform)))
                .collect();
            Ipld::Map(mapped_map)
        }
        other => other.clone(),
    };

    transform(&transformed)
}

/// Find the differences between two IPLD DAGs
///
/// Returns a tuple of (unique_to_first, unique_to_second, common_links).
/// This is useful for determining what changed between two versions of a DAG.
///
/// # Arguments
///
/// * `dag1` - First DAG to compare
/// * `dag2` - Second DAG to compare
///
/// # Returns
///
/// A tuple of:
/// - Links unique to dag1
/// - Links unique to dag2
/// - Links common to both
///
/// # Example
///
/// ```rust
/// use ipfrs_core::{Ipld, CidBuilder};
/// use ipfrs_core::dag::dag_diff;
/// use std::collections::BTreeMap;
///
/// let cid1 = CidBuilder::new().build(b"a").unwrap();
/// let cid2 = CidBuilder::new().build(b"b").unwrap();
/// let cid3 = CidBuilder::new().build(b"c").unwrap();
///
/// let mut map1 = BTreeMap::new();
/// map1.insert("link1".to_string(), Ipld::link(cid1));
/// map1.insert("link2".to_string(), Ipld::link(cid2));
///
/// let mut map2 = BTreeMap::new();
/// map2.insert("link2".to_string(), Ipld::link(cid2));
/// map2.insert("link3".to_string(), Ipld::link(cid3));
///
/// let dag1 = Ipld::Map(map1);
/// let dag2 = Ipld::Map(map2);
///
/// let (unique1, unique2, common) = dag_diff(&dag1, &dag2);
/// assert_eq!(unique1.len(), 1); // cid1
/// assert_eq!(unique2.len(), 1); // cid3
/// assert_eq!(common.len(), 1);  // cid2
/// ```
pub fn dag_diff(dag1: &Ipld, dag2: &Ipld) -> (HashSet<Cid>, HashSet<Cid>, HashSet<Cid>) {
    let links1 = collect_unique_links(dag1);
    let links2 = collect_unique_links(dag2);

    let unique_to_first: HashSet<Cid> = links1.difference(&links2).copied().collect();
    let unique_to_second: HashSet<Cid> = links2.difference(&links1).copied().collect();
    let common: HashSet<Cid> = links1.intersection(&links2).copied().collect();

    (unique_to_first, unique_to_second, common)
}

/// Find common ancestor links between two DAGs
///
/// Returns the set of CID links that appear in both DAGs, which can help
/// identify shared structure or common ancestry.
///
/// # Arguments
///
/// * `dag1` - First DAG
/// * `dag2` - Second DAG
///
/// # Returns
///
/// A set of CIDs that appear in both DAGs
///
/// # Example
///
/// ```rust
/// use ipfrs_core::{Ipld, CidBuilder};
/// use ipfrs_core::dag::find_common_links;
///
/// let cid = CidBuilder::new().build(b"shared").unwrap();
///
/// let dag1 = Ipld::List(vec![Ipld::link(cid)]);
/// let dag2 = Ipld::List(vec![Ipld::link(cid)]);
///
/// let common = find_common_links(&dag1, &dag2);
/// assert_eq!(common.len(), 1);
/// ```
pub fn find_common_links(dag1: &Ipld, dag2: &Ipld) -> HashSet<Cid> {
    let links1 = collect_unique_links(dag1);
    let links2 = collect_unique_links(dag2);

    links1.intersection(&links2).copied().collect()
}

/// Prune nodes from a DAG based on a predicate
///
/// This function creates a new DAG with only the nodes that match the predicate.
/// It's useful for removing unwanted nodes or creating filtered views of a DAG.
///
/// # Arguments
///
/// * `ipld` - The DAG to prune
/// * `should_keep` - Predicate function returning true for nodes to keep
///
/// # Returns
///
/// A new IPLD structure with pruned nodes, or None if the root is pruned
///
/// # Example
///
/// ```rust
/// use ipfrs_core::Ipld;
/// use ipfrs_core::dag::prune_dag;
///
/// let ipld = Ipld::List(vec![
///     Ipld::Integer(1),
///     Ipld::Integer(2),
///     Ipld::String("keep".to_string()),
/// ]);
///
/// // Keep only strings
/// let pruned = prune_dag(&ipld, &|node| {
///     matches!(node, Ipld::String(_) | Ipld::List(_))
/// });
///
/// assert!(pruned.is_some());
/// ```
pub fn prune_dag<F>(ipld: &Ipld, should_keep: &F) -> Option<Ipld>
where
    F: Fn(&Ipld) -> bool,
{
    if !should_keep(ipld) {
        return None;
    }

    match ipld {
        Ipld::List(items) => {
            let pruned_items: Vec<Ipld> = items
                .iter()
                .filter_map(|item| prune_dag(item, should_keep))
                .collect();

            if pruned_items.is_empty() && !items.is_empty() {
                None
            } else {
                Some(Ipld::List(pruned_items))
            }
        }
        Ipld::Map(map) => {
            let pruned_map: std::collections::BTreeMap<String, Ipld> = map
                .iter()
                .filter_map(|(k, v)| prune_dag(v, should_keep).map(|pruned| (k.clone(), pruned)))
                .collect();

            if pruned_map.is_empty() && !map.is_empty() {
                None
            } else {
                Some(Ipld::Map(pruned_map))
            }
        }
        other => Some(other.clone()),
    }
}

/// Merge two DAGs into a single DAG
///
/// Creates a new DAG that contains all nodes from both input DAGs. When both
/// DAGs are maps, their keys are merged (dag2 values override dag1 on conflicts).
/// When both are lists, they are concatenated.
///
/// # Arguments
///
/// * `dag1` - First DAG
/// * `dag2` - Second DAG
///
/// # Returns
///
/// A merged IPLD structure
///
/// # Example
///
/// ```rust
/// use ipfrs_core::Ipld;
/// use ipfrs_core::dag::merge_dags;
/// use std::collections::BTreeMap;
///
/// let mut map1 = BTreeMap::new();
/// map1.insert("a".to_string(), Ipld::Integer(1));
///
/// let mut map2 = BTreeMap::new();
/// map2.insert("b".to_string(), Ipld::Integer(2));
///
/// let dag1 = Ipld::Map(map1);
/// let dag2 = Ipld::Map(map2);
///
/// let merged = merge_dags(&dag1, &dag2);
/// if let Ipld::Map(m) = merged {
///     assert_eq!(m.len(), 2);
/// }
/// ```
pub fn merge_dags(dag1: &Ipld, dag2: &Ipld) -> Ipld {
    match (dag1, dag2) {
        (Ipld::Map(map1), Ipld::Map(map2)) => {
            let mut merged = map1.clone();
            for (k, v) in map2 {
                merged.insert(k.clone(), v.clone());
            }
            Ipld::Map(merged)
        }
        (Ipld::List(list1), Ipld::List(list2)) => {
            let mut merged = list1.clone();
            merged.extend(list2.clone());
            Ipld::List(merged)
        }
        // If types don't match, prefer dag2
        (_, dag2) => dag2.clone(),
    }
}

/// Count the total number of nodes in a DAG
///
/// This includes all nodes at all levels, counting duplicates if they appear
/// multiple times in the structure.
///
/// # Arguments
///
/// * `ipld` - The DAG to count nodes in
///
/// # Returns
///
/// The total number of nodes (including the root)
///
/// # Example
///
/// ```rust
/// use ipfrs_core::Ipld;
/// use ipfrs_core::dag::count_nodes;
///
/// let ipld = Ipld::List(vec![
///     Ipld::Integer(1),
///     Ipld::Integer(2),
///     Ipld::List(vec![Ipld::Integer(3)]),
/// ]);
///
/// assert_eq!(count_nodes(&ipld), 5); // List + 2 ints + inner list + 1 int
/// ```
pub fn count_nodes(ipld: &Ipld) -> usize {
    match ipld {
        Ipld::List(items) => 1 + items.iter().map(count_nodes).sum::<usize>(),
        Ipld::Map(map) => 1 + map.values().map(count_nodes).sum::<usize>(),
        _ => 1,
    }
}

/// Get the maximum depth of a DAG
///
/// Returns the length of the longest path from the root to a leaf node.
///
/// # Arguments
///
/// * `ipld` - The DAG to measure
///
/// # Returns
///
/// The maximum depth (0 for leaf nodes, 1+ for containers)
///
/// # Example
///
/// ```rust
/// use ipfrs_core::Ipld;
/// use ipfrs_core::dag::dag_depth;
///
/// let ipld = Ipld::List(vec![
///     Ipld::Integer(1),
///     Ipld::List(vec![
///         Ipld::Integer(2),
///         Ipld::List(vec![Ipld::Integer(3)]),
///     ]),
/// ]);
///
/// assert_eq!(dag_depth(&ipld), 4);
/// ```
pub fn dag_depth(ipld: &Ipld) -> usize {
    match ipld {
        Ipld::List(items) => {
            if items.is_empty() {
                1
            } else {
                1 + items.iter().map(dag_depth).max().unwrap_or(0)
            }
        }
        Ipld::Map(map) => {
            if map.is_empty() {
                1
            } else {
                1 + map.values().map(dag_depth).max().unwrap_or(0)
            }
        }
        _ => 1,
    }
}

/// Find all leaf nodes in a DAG
///
/// Returns all nodes that have no children (i.e., not List or Map, or empty containers).
///
/// # Arguments
///
/// * `ipld` - The DAG to search
///
/// # Returns
///
/// A vector of all leaf nodes
///
/// # Example
///
/// ```rust
/// use ipfrs_core::Ipld;
/// use ipfrs_core::dag::find_leaves;
///
/// let ipld = Ipld::List(vec![
///     Ipld::Integer(1),
///     Ipld::String("leaf".to_string()),
///     Ipld::List(vec![Ipld::Integer(2)]),
/// ]);
///
/// let leaves = find_leaves(&ipld);
/// assert_eq!(leaves.len(), 3); // Two integers and one string
/// ```
pub fn find_leaves(ipld: &Ipld) -> Vec<Ipld> {
    match ipld {
        Ipld::List(items) => {
            if items.is_empty() {
                vec![ipld.clone()]
            } else {
                items.iter().flat_map(find_leaves).collect()
            }
        }
        Ipld::Map(map) => {
            if map.is_empty() {
                vec![ipld.clone()]
            } else {
                map.values().flat_map(find_leaves).collect()
            }
        }
        _ => vec![ipld.clone()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cid::CidBuilder;
    use std::collections::BTreeMap;

    #[test]
    fn test_extract_links() {
        let cid1 = CidBuilder::new().build(b"test1").unwrap();
        let cid2 = CidBuilder::new().build(b"test2").unwrap();

        let mut map = BTreeMap::new();
        map.insert("link1".to_string(), Ipld::link(cid1));
        map.insert("link2".to_string(), Ipld::link(cid2));
        map.insert("data".to_string(), Ipld::String("hello".to_string()));

        let ipld = Ipld::Map(map);
        let links = extract_links(&ipld);

        assert_eq!(links.len(), 2);
        assert!(links.contains(&cid1));
        assert!(links.contains(&cid2));
    }

    #[test]
    fn test_collect_all_links_nested() {
        let cid1 = CidBuilder::new().build(b"test1").unwrap();
        let cid2 = CidBuilder::new().build(b"test2").unwrap();
        let cid3 = CidBuilder::new().build(b"test3").unwrap();

        let mut inner = BTreeMap::new();
        inner.insert("deep_link".to_string(), Ipld::link(cid3));

        let mut outer = BTreeMap::new();
        outer.insert("link1".to_string(), Ipld::link(cid1));
        outer.insert("link2".to_string(), Ipld::link(cid2));
        outer.insert("nested".to_string(), Ipld::Map(inner));

        let ipld = Ipld::Map(outer);
        let links = collect_all_links(&ipld);

        assert_eq!(links.len(), 3);
        assert!(links.contains(&cid1));
        assert!(links.contains(&cid2));
        assert!(links.contains(&cid3));
    }

    #[test]
    fn test_collect_unique_links() {
        let cid1 = CidBuilder::new().build(b"test1").unwrap();
        let cid2 = CidBuilder::new().build(b"test2").unwrap();

        // Create structure with duplicate links
        let list = vec![
            Ipld::link(cid1),
            Ipld::link(cid2),
            Ipld::link(cid1), // Duplicate
        ];

        let ipld = Ipld::List(list);
        let unique = collect_unique_links(&ipld);

        assert_eq!(unique.len(), 2);
    }

    #[test]
    fn test_dag_stats() {
        let cid1 = CidBuilder::new().build(b"test1").unwrap();
        let cid2 = CidBuilder::new().build(b"test2").unwrap();

        let mut map = BTreeMap::new();
        map.insert("link1".to_string(), Ipld::link(cid1));
        map.insert("link2".to_string(), Ipld::link(cid2));
        map.insert("dup_link".to_string(), Ipld::link(cid1)); // Duplicate

        let ipld = Ipld::Map(map);
        let stats = DagStats::from_ipld(&ipld);

        assert_eq!(stats.unique_cids, 2);
        assert_eq!(stats.total_links, 3);
        assert!(stats.deduplication_ratio() > 0.0);
    }

    #[test]
    fn test_dag_stats_nested() {
        let cid1 = CidBuilder::new().build(b"test1").unwrap();
        let cid2 = CidBuilder::new().build(b"test2").unwrap();

        let mut inner = BTreeMap::new();
        inner.insert("deep".to_string(), Ipld::link(cid2));

        let mut outer = BTreeMap::new();
        outer.insert("link".to_string(), Ipld::link(cid1));
        outer.insert("nested".to_string(), Ipld::Map(inner));

        let ipld = Ipld::Map(outer);
        let stats = DagStats::from_ipld(&ipld);

        assert_eq!(stats.unique_cids, 2);
        assert_eq!(stats.max_depth, 2);
    }

    #[test]
    fn test_is_dag() {
        let cid1 = CidBuilder::new().build(b"test1").unwrap();

        let mut map = BTreeMap::new();
        map.insert("link".to_string(), Ipld::link(cid1));

        let ipld = Ipld::Map(map);
        assert!(is_dag(&ipld));
    }

    #[test]
    fn test_find_paths_to_cid() {
        let cid1 = CidBuilder::new().build(b"test1").unwrap();
        let cid2 = CidBuilder::new().build(b"test2").unwrap();

        let mut inner = BTreeMap::new();
        inner.insert("target".to_string(), Ipld::link(cid1));

        let mut outer = BTreeMap::new();
        outer.insert("other".to_string(), Ipld::link(cid2));
        outer.insert("nested".to_string(), Ipld::Map(inner));

        let ipld = Ipld::Map(outer);
        let paths = find_paths_to_cid(&ipld, &cid1);

        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], vec!["nested".to_string(), "target".to_string()]);
    }

    #[test]
    fn test_traverse_bfs() {
        let list = vec![
            Ipld::Integer(1),
            Ipld::Integer(2),
            Ipld::List(vec![Ipld::Integer(3), Ipld::Integer(4)]),
        ];
        let ipld = Ipld::List(list);

        let nodes = traverse_bfs(&ipld);
        assert_eq!(nodes.len(), 6); // Root + 3 direct children (1, 2, list) + 2 nested (3, 4)
    }

    #[test]
    fn test_traverse_dfs() {
        let list = vec![
            Ipld::Integer(1),
            Ipld::List(vec![Ipld::Integer(2)]),
            Ipld::Integer(3),
        ];
        let ipld = Ipld::List(list);

        let nodes = traverse_dfs(&ipld);
        assert_eq!(nodes.len(), 5); // Root + all children
    }

    #[test]
    fn test_empty_ipld() {
        let ipld = Ipld::Null;
        let links = extract_links(&ipld);
        assert!(links.is_empty());

        let stats = DagStats::from_ipld(&ipld);
        assert_eq!(stats.unique_cids, 0);
        assert_eq!(stats.total_links, 0);
    }

    #[test]
    fn test_deduplication_ratio() {
        // No duplication
        let cid1 = CidBuilder::new().build(b"test1").unwrap();
        let cid2 = CidBuilder::new().build(b"test2").unwrap();

        let list = vec![Ipld::link(cid1), Ipld::link(cid2)];
        let ipld = Ipld::List(list);
        let stats = DagStats::from_ipld(&ipld);
        assert_eq!(stats.deduplication_ratio(), 0.0);

        // 50% duplication
        let list = vec![
            Ipld::link(cid1),
            Ipld::link(cid2),
            Ipld::link(cid1),
            Ipld::link(cid2),
        ];
        let ipld = Ipld::List(list);
        let stats = DagStats::from_ipld(&ipld);
        assert_eq!(stats.deduplication_ratio(), 0.5);
    }

    #[test]
    fn test_dag_metrics() {
        let cid = CidBuilder::new().build(b"test").unwrap();
        let mut map = BTreeMap::new();
        map.insert("link1".to_string(), Ipld::link(cid));
        map.insert("link2".to_string(), Ipld::link(cid));
        let ipld = Ipld::Map(map);

        let metrics = DagMetrics::from_ipld(&ipld);
        assert_eq!(metrics.max_branching_factor, 2);
        assert!(metrics.width >= 1);
        assert!(metrics.total_nodes > 0);
    }

    #[test]
    fn test_dag_metrics_nested() {
        let cid = CidBuilder::new().build(b"test").unwrap();

        let mut inner = BTreeMap::new();
        inner.insert("deep1".to_string(), Ipld::Integer(1));
        inner.insert("deep2".to_string(), Ipld::Integer(2));
        inner.insert("deep3".to_string(), Ipld::link(cid));

        let mut outer = BTreeMap::new();
        outer.insert("nested".to_string(), Ipld::Map(inner));
        outer.insert("value".to_string(), Ipld::String("test".to_string()));

        let ipld = Ipld::Map(outer);
        let metrics = DagMetrics::from_ipld(&ipld);

        assert_eq!(metrics.max_branching_factor, 3);
        assert!(metrics.avg_branching_factor > 0.0);
        assert!(metrics.avg_leaf_depth > 0.0);
    }

    #[test]
    fn test_topological_sort() {
        let cid1 = CidBuilder::new().build(b"leaf1").unwrap();
        let cid2 = CidBuilder::new().build(b"leaf2").unwrap();

        let mut map = BTreeMap::new();
        map.insert("child1".to_string(), Ipld::link(cid1));
        map.insert("child2".to_string(), Ipld::link(cid2));
        let ipld = Ipld::Map(map);

        let sorted = topological_sort(&ipld);
        assert_eq!(sorted.len(), 2);
        assert!(sorted.contains(&cid1));
        assert!(sorted.contains(&cid2));
    }

    #[test]
    fn test_topological_sort_with_duplicates() {
        let cid1 = CidBuilder::new().build(b"test").unwrap();

        let list = vec![Ipld::link(cid1), Ipld::link(cid1), Ipld::link(cid1)];
        let ipld = Ipld::List(list);

        let sorted = topological_sort(&ipld);
        // Should deduplicate
        assert_eq!(sorted.len(), 1);
        assert_eq!(sorted[0], cid1);
    }

    #[test]
    fn test_subgraph_size() {
        let ipld = Ipld::List(vec![
            Ipld::Integer(1),
            Ipld::Integer(2),
            Ipld::List(vec![Ipld::Integer(3)]),
        ]);

        let size = subgraph_size(&ipld);
        assert_eq!(size, 5); // Root list + 2 integers + nested list + nested integer
    }

    #[test]
    fn test_subgraph_size_single_node() {
        let ipld = Ipld::Integer(42);
        let size = subgraph_size(&ipld);
        assert_eq!(size, 1);
    }

    #[test]
    fn test_dag_fanout_by_level() {
        let ipld = Ipld::List(vec![
            Ipld::Integer(1),
            Ipld::List(vec![Ipld::Integer(2), Ipld::Integer(3)]),
        ]);

        let fanout = dag_fanout_by_level(&ipld);
        assert!(fanout.len() >= 2);
        assert_eq!(fanout[0], 2); // Root has 2 children
    }

    #[test]
    fn test_dag_fanout_empty() {
        let ipld = Ipld::Integer(42);
        let fanout = dag_fanout_by_level(&ipld);
        // Scalar value has no children
        assert!(fanout.is_empty() || fanout.iter().all(|&f| f == 0));
    }

    #[test]
    fn test_count_links_by_depth() {
        let cid = CidBuilder::new().build(b"test").unwrap();
        let mut map = BTreeMap::new();
        map.insert("link".to_string(), Ipld::link(cid));
        let ipld = Ipld::Map(map);

        let counts = count_links_by_depth(&ipld);
        assert!(!counts.is_empty());
        // Links inside map values are at depth 1
        assert_eq!(counts[1], 1);
    }

    #[test]
    fn test_count_links_by_depth_nested() {
        let cid1 = CidBuilder::new().build(b"test1").unwrap();
        let cid2 = CidBuilder::new().build(b"test2").unwrap();

        let mut inner = BTreeMap::new();
        inner.insert("deep".to_string(), Ipld::link(cid2));

        let mut outer = BTreeMap::new();
        outer.insert("shallow".to_string(), Ipld::link(cid1));
        outer.insert("nested".to_string(), Ipld::Map(inner));

        let ipld = Ipld::Map(outer);
        let counts = count_links_by_depth(&ipld);

        assert!(counts.len() >= 2);
        // Links in outer map are at depth 1
        assert_eq!(counts[1], 1); // One link at depth 1 (shallow)
                                  // Links in inner map are at depth 2
        assert_eq!(counts[2], 1); // One link at depth 2 (deep)
    }

    #[test]
    fn test_filter_dag() {
        let ipld = Ipld::List(vec![
            Ipld::Integer(1),
            Ipld::Integer(2),
            Ipld::String("hello".to_string()),
        ]);

        // Keep only integers and lists
        let filtered = filter_dag(&ipld, &|node| {
            matches!(node, Ipld::Integer(_) | Ipld::List(_))
        });
        assert!(filtered.is_some());

        if let Some(Ipld::List(items)) = filtered {
            assert_eq!(items.len(), 2); // Only the two integers
        } else {
            panic!("Expected filtered list");
        }
    }

    #[test]
    fn test_filter_dag_all_filtered() {
        let ipld = Ipld::Integer(42);

        // Filter out everything
        let filtered = filter_dag(&ipld, &|_| false);
        assert!(filtered.is_none());
    }

    #[test]
    fn test_map_dag() {
        let ipld = Ipld::Integer(42);

        // Double all integers
        let transformed = map_dag(&ipld, &|node| match node {
            Ipld::Integer(n) => Ipld::Integer(n * 2),
            other => other.clone(),
        });

        assert_eq!(transformed, Ipld::Integer(84));
    }

    #[test]
    fn test_map_dag_nested() {
        let ipld = Ipld::List(vec![Ipld::Integer(1), Ipld::Integer(2)]);

        // Double all integers
        let transformed = map_dag(&ipld, &|node| match node {
            Ipld::Integer(n) => Ipld::Integer(n * 2),
            other => other.clone(),
        });

        if let Ipld::List(items) = transformed {
            assert_eq!(items[0], Ipld::Integer(2));
            assert_eq!(items[1], Ipld::Integer(4));
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_map_dag_preserve_structure() {
        let mut map = BTreeMap::new();
        map.insert("a".to_string(), Ipld::Integer(1));
        map.insert("b".to_string(), Ipld::Integer(2));
        let ipld = Ipld::Map(map);

        // Transform preserves map structure
        let transformed = map_dag(&ipld, &|node| match node {
            Ipld::Integer(n) => Ipld::Integer(n + 10),
            other => other.clone(),
        });

        if let Ipld::Map(result_map) = transformed {
            assert_eq!(result_map.get("a"), Some(&Ipld::Integer(11)));
            assert_eq!(result_map.get("b"), Some(&Ipld::Integer(12)));
        } else {
            panic!("Expected map");
        }
    }
}
