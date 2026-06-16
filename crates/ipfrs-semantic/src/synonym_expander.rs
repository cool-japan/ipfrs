//! # Semantic Synonym Expander
//!
//! Builds and queries a synonym graph for vocabulary expansion in semantic search.
//! Supports weighted synonymy relationships and multi-hop BFS expansion.

use std::collections::{HashMap, HashSet, VecDeque};

/// Describes the semantic relationship type between two terms in the synonym graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SynonymRelation {
    /// Identical meaning (e.g., "automobile" ↔ "car").
    Exact,
    /// Close but not identical meaning (e.g., "happy" ↔ "joyful").
    Near,
    /// Source is more specific than target (e.g., "spaniel" → "dog").
    Broader,
    /// Source is more general than target (e.g., "dog" → "spaniel").
    Narrower,
}

/// A single directed synonym edge in the graph.
#[derive(Clone, Debug)]
pub struct SynonymEdge {
    /// Target term this edge points to.
    pub target: String,
    /// Semantic relationship from the source to the target.
    pub relation: SynonymRelation,
    /// Confidence/strength of the synonym relationship in [0.0, 1.0].
    pub weight: f32,
}

/// Configuration for the [`SemanticSynonymExpander`].
#[derive(Clone, Debug)]
pub struct ExpanderConfig {
    /// Maximum BFS depth when expanding from a query term.
    pub max_hops: usize,
    /// Minimum edge weight to follow during expansion.
    pub min_weight: f32,
    /// Maximum number of expanded terms to return per query.
    pub max_expansions: usize,
}

impl Default for ExpanderConfig {
    fn default() -> Self {
        Self {
            max_hops: 2,
            min_weight: 0.5,
            max_expansions: 20,
        }
    }
}

/// A single term produced by [`SemanticSynonymExpander::expand`].
#[derive(Clone, Debug)]
pub struct ExpandedTerm {
    /// The expanded synonym term.
    pub term: String,
    /// Relation from the immediate predecessor on the expansion path.
    pub relation: SynonymRelation,
    /// Product of edge weights along the path from the query term.
    pub cumulative_weight: f32,
    /// Number of hops from the query term.
    pub hops: usize,
}

/// Aggregate statistics tracked by the expander.
#[derive(Clone, Debug, Default)]
pub struct SynonymExpanderStats {
    /// Number of unique terms currently in the graph.
    pub total_terms: usize,
    /// Total synonym edges currently in the graph.
    pub total_edges: usize,
    /// Total number of calls to [`SemanticSynonymExpander::expand`].
    pub total_expand_calls: u64,
    /// Total number of terms returned across all expand calls.
    pub total_terms_expanded: u64,
}

/// BFS frontier entry used internally during expansion.
struct FrontierEntry {
    term: String,
    hops: usize,
    cumulative_weight: f32,
    relation: SynonymRelation,
}

/// Synonym graph for vocabulary expansion in semantic search.
///
/// Maintains a weighted directed adjacency list; `add_synonym` automatically
/// inserts bidirectional edges with correctly mirrored relationship types.
pub struct SemanticSynonymExpander {
    /// Adjacency list: term → list of outgoing synonym edges.
    pub graph: HashMap<String, Vec<SynonymEdge>>,
    /// Expander configuration.
    pub config: ExpanderConfig,
    /// Runtime statistics.
    pub stats: SynonymExpanderStats,
}

impl SemanticSynonymExpander {
    /// Create a new expander with the given configuration.
    pub fn new(config: ExpanderConfig) -> Self {
        Self {
            graph: HashMap::new(),
            config,
            stats: SynonymExpanderStats::default(),
        }
    }

    /// Add a bidirectional synonym edge between `from` and `to`.
    ///
    /// The reverse edge uses the mirrored relation:
    /// - `Broader` ↔ `Narrower`
    /// - `Exact` ↔ `Exact`
    /// - `Near` ↔ `Near`
    ///
    /// Duplicate edges (same target + same relation) are silently skipped;
    /// `stats.total_edges` is incremented only for edges actually inserted.
    pub fn add_synonym(&mut self, from: &str, to: &str, relation: SynonymRelation, weight: f32) {
        let reverse_relation = match relation {
            SynonymRelation::Broader => SynonymRelation::Narrower,
            SynonymRelation::Narrower => SynonymRelation::Broader,
            other => other,
        };

        // Forward edge: from → to
        let forward_added = Self::insert_edge(
            &mut self.graph,
            from,
            SynonymEdge {
                target: to.to_owned(),
                relation,
                weight,
            },
        );

        // Reverse edge: to → from
        let reverse_added = Self::insert_edge(
            &mut self.graph,
            to,
            SynonymEdge {
                target: from.to_owned(),
                relation: reverse_relation,
                weight,
            },
        );

        self.stats.total_edges += forward_added as usize + reverse_added as usize;
        self.stats.total_terms = self.graph.len();
    }

    /// Insert `edge` into `graph[node]`, skipping if a duplicate (same target + relation) exists.
    /// Returns `true` if the edge was actually inserted.
    fn insert_edge(
        graph: &mut HashMap<String, Vec<SynonymEdge>>,
        node: &str,
        edge: SynonymEdge,
    ) -> bool {
        let edges = graph.entry(node.to_owned()).or_default();
        let is_dup = edges
            .iter()
            .any(|e| e.target == edge.target && e.relation == edge.relation);
        if is_dup {
            return false;
        }
        edges.push(edge);
        true
    }

    /// BFS expansion from `term` up to `config.max_hops` hops.
    ///
    /// Only follows edges whose weight >= `config.min_weight` and whose relation
    /// is in `allowed_relations` (if `Some`).  Results are sorted by
    /// `cumulative_weight` descending, then alphabetically by term, and
    /// truncated to `config.max_expansions`.
    pub fn expand(
        &mut self,
        term: &str,
        allowed_relations: Option<&[SynonymRelation]>,
    ) -> Vec<ExpandedTerm> {
        self.stats.total_expand_calls += 1;

        if !self.graph.contains_key(term) {
            return Vec::new();
        }

        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(term.to_owned());

        let mut queue: VecDeque<FrontierEntry> = VecDeque::new();

        // Seed the BFS with neighbours of the query term
        if let Some(edges) = self.graph.get(term) {
            for edge in edges {
                if edge.weight < self.config.min_weight {
                    continue;
                }
                if let Some(allowed) = allowed_relations {
                    if !allowed.contains(&edge.relation) {
                        continue;
                    }
                }
                if visited.insert(edge.target.clone()) {
                    queue.push_back(FrontierEntry {
                        term: edge.target.clone(),
                        hops: 1,
                        cumulative_weight: edge.weight,
                        relation: edge.relation,
                    });
                }
            }
        }

        let mut results: Vec<ExpandedTerm> = Vec::new();

        while let Some(entry) = queue.pop_front() {
            results.push(ExpandedTerm {
                term: entry.term.clone(),
                relation: entry.relation,
                cumulative_weight: entry.cumulative_weight,
                hops: entry.hops,
            });

            if entry.hops >= self.config.max_hops {
                continue;
            }

            // Expand further from this node
            if let Some(edges) = self.graph.get(&entry.term) {
                // Collect to avoid borrow-checker issues
                let candidates: Vec<_> = edges
                    .iter()
                    .filter(|e| {
                        e.weight >= self.config.min_weight
                            && allowed_relations
                                .map(|a| a.contains(&e.relation))
                                .unwrap_or(true)
                            && !visited.contains(&e.target)
                    })
                    .map(|e| (e.target.clone(), e.relation, e.weight))
                    .collect();

                for (target, relation, weight) in candidates {
                    if visited.insert(target.clone()) {
                        queue.push_back(FrontierEntry {
                            term: target,
                            hops: entry.hops + 1,
                            cumulative_weight: entry.cumulative_weight * weight,
                            relation,
                        });
                    }
                }
            }
        }

        // Sort: cumulative_weight descending, then term alphabetically
        results.sort_by(|a, b| {
            b.cumulative_weight
                .partial_cmp(&a.cumulative_weight)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.term.cmp(&b.term))
        });

        results.truncate(self.config.max_expansions);

        self.stats.total_terms_expanded += results.len() as u64;
        results
    }

    /// Remove `term` from the graph and all incoming edges from other nodes.
    pub fn remove_term(&mut self, term: &str) {
        self.graph.remove(term);
        for edges in self.graph.values_mut() {
            edges.retain(|e| e.target != term);
        }
        self.stats.total_terms = self.graph.len();
        self.stats.total_edges = self.graph.values().map(|v| v.len()).sum();
    }

    /// Return a reference to the current statistics.
    pub fn stats(&self) -> &SynonymExpanderStats {
        &self.stats
    }

    /// Return the number of unique terms currently in the graph.
    pub fn term_count(&self) -> usize {
        self.graph.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_expander() -> SemanticSynonymExpander {
        SemanticSynonymExpander::new(ExpanderConfig::default())
    }

    // -----------------------------------------------------------------------
    // add_synonym — basic bidirectionality
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_synonym_bidirectional_exact() {
        let mut exp = default_expander();
        exp.add_synonym("car", "automobile", SynonymRelation::Exact, 0.9);
        // car → automobile
        assert!(exp.graph["car"].iter().any(|e| e.target == "automobile"));
        // automobile → car
        assert!(exp.graph["automobile"].iter().any(|e| e.target == "car"));
    }

    #[test]
    fn test_add_synonym_bidirectional_near() {
        let mut exp = default_expander();
        exp.add_synonym("happy", "joyful", SynonymRelation::Near, 0.8);
        let fwd = exp.graph["happy"]
            .iter()
            .find(|e| e.target == "joyful")
            .expect("forward edge missing");
        let rev = exp.graph["joyful"]
            .iter()
            .find(|e| e.target == "happy")
            .expect("reverse edge missing");
        assert_eq!(fwd.relation, SynonymRelation::Near);
        assert_eq!(rev.relation, SynonymRelation::Near);
    }

    #[test]
    fn test_add_synonym_broader_reverse_is_narrower() {
        let mut exp = default_expander();
        exp.add_synonym("spaniel", "dog", SynonymRelation::Broader, 0.9);
        let rev = exp.graph["dog"]
            .iter()
            .find(|e| e.target == "spaniel")
            .expect("reverse edge missing");
        assert_eq!(rev.relation, SynonymRelation::Narrower);
    }

    #[test]
    fn test_add_synonym_narrower_reverse_is_broader() {
        let mut exp = default_expander();
        exp.add_synonym("dog", "spaniel", SynonymRelation::Narrower, 0.9);
        let rev = exp.graph["spaniel"]
            .iter()
            .find(|e| e.target == "dog")
            .expect("reverse edge missing");
        assert_eq!(rev.relation, SynonymRelation::Broader);
    }

    #[test]
    fn test_add_synonym_weight_preserved() {
        let mut exp = default_expander();
        exp.add_synonym("a", "b", SynonymRelation::Exact, 0.75);
        assert!((exp.graph["a"][0].weight - 0.75).abs() < f32::EPSILON);
        assert!((exp.graph["b"][0].weight - 0.75).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // duplicate edge skipping
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_synonym_no_duplicate_edges() {
        let mut exp = default_expander();
        exp.add_synonym("car", "automobile", SynonymRelation::Exact, 0.9);
        exp.add_synonym("car", "automobile", SynonymRelation::Exact, 0.7);
        // Each direction should still have exactly one edge
        assert_eq!(exp.graph["car"].len(), 1);
        assert_eq!(exp.graph["automobile"].len(), 1);
    }

    #[test]
    fn test_add_synonym_duplicate_does_not_increment_stats() {
        let mut exp = default_expander();
        exp.add_synonym("car", "automobile", SynonymRelation::Exact, 0.9);
        let edges_after_first = exp.stats.total_edges;
        exp.add_synonym("car", "automobile", SynonymRelation::Exact, 0.7);
        assert_eq!(exp.stats.total_edges, edges_after_first);
    }

    #[test]
    fn test_add_synonym_different_relation_is_not_duplicate() {
        let mut exp = default_expander();
        exp.add_synonym("word", "synonym", SynonymRelation::Exact, 0.9);
        exp.add_synonym("word", "synonym", SynonymRelation::Near, 0.7);
        // Two distinct relations → two edges in each direction
        assert_eq!(exp.graph["word"].len(), 2);
    }

    // -----------------------------------------------------------------------
    // stats: total_terms and total_edges
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_total_terms_and_edges() {
        let mut exp = default_expander();
        exp.add_synonym("a", "b", SynonymRelation::Exact, 0.9);
        assert_eq!(exp.stats.total_terms, 2);
        assert_eq!(exp.stats.total_edges, 2); // one forward + one reverse
    }

    #[test]
    fn test_stats_multiple_synonyms() {
        let mut exp = default_expander();
        exp.add_synonym("a", "b", SynonymRelation::Exact, 0.9);
        exp.add_synonym("a", "c", SynonymRelation::Near, 0.8);
        assert_eq!(exp.stats.total_terms, 3);
        assert_eq!(exp.stats.total_edges, 4); // 2 forward + 2 reverse
    }

    // -----------------------------------------------------------------------
    // expand — basic
    // -----------------------------------------------------------------------

    #[test]
    fn test_expand_empty_for_unknown_term() {
        let mut exp = default_expander();
        let results = exp.expand("ghost", None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_expand_single_hop() {
        let mut exp = default_expander();
        exp.add_synonym("car", "automobile", SynonymRelation::Exact, 0.9);
        let results = exp.expand("car", None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].term, "automobile");
        assert_eq!(results[0].hops, 1);
        assert!((results[0].cumulative_weight - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_expand_does_not_include_query_term() {
        let mut exp = default_expander();
        exp.add_synonym("car", "automobile", SynonymRelation::Exact, 0.9);
        let results = exp.expand("car", None);
        assert!(!results.iter().any(|r| r.term == "car"));
    }

    // -----------------------------------------------------------------------
    // expand — BFS depth and max_hops
    // -----------------------------------------------------------------------

    #[test]
    fn test_expand_bfs_up_to_max_hops() {
        let config = ExpanderConfig {
            max_hops: 2,
            ..ExpanderConfig::default()
        };
        let mut exp = SemanticSynonymExpander::new(config);
        // chain: a → b → c → d
        exp.add_synonym("a", "b", SynonymRelation::Exact, 0.9);
        exp.add_synonym("b", "c", SynonymRelation::Exact, 0.9);
        exp.add_synonym("c", "d", SynonymRelation::Exact, 0.9);
        let results = exp.expand("a", None);
        let terms: Vec<&str> = results.iter().map(|r| r.term.as_str()).collect();
        assert!(terms.contains(&"b"), "hop-1 term b must be present");
        assert!(terms.contains(&"c"), "hop-2 term c must be present");
        // d is 3 hops from a → should NOT appear with max_hops=2
        assert!(
            !terms.contains(&"d"),
            "hop-3 term d must be absent with max_hops=2"
        );
    }

    #[test]
    fn test_expand_max_hops_one() {
        let config = ExpanderConfig {
            max_hops: 1,
            ..ExpanderConfig::default()
        };
        let mut exp = SemanticSynonymExpander::new(config);
        exp.add_synonym("a", "b", SynonymRelation::Exact, 0.9);
        exp.add_synonym("b", "c", SynonymRelation::Exact, 0.9);
        let results = exp.expand("a", None);
        let terms: Vec<&str> = results.iter().map(|r| r.term.as_str()).collect();
        assert!(terms.contains(&"b"));
        assert!(!terms.contains(&"c"));
    }

    // -----------------------------------------------------------------------
    // expand — min_weight filtering
    // -----------------------------------------------------------------------

    #[test]
    fn test_expand_filters_by_min_weight() {
        let config = ExpanderConfig {
            min_weight: 0.7,
            ..ExpanderConfig::default()
        };
        let mut exp = SemanticSynonymExpander::new(config);
        exp.add_synonym("a", "b", SynonymRelation::Exact, 0.9); // passes
        exp.add_synonym("a", "c", SynonymRelation::Near, 0.4); // below threshold
        let results = exp.expand("a", None);
        let terms: Vec<&str> = results.iter().map(|r| r.term.as_str()).collect();
        assert!(terms.contains(&"b"));
        assert!(!terms.contains(&"c"));
    }

    #[test]
    fn test_expand_min_weight_exact_boundary() {
        let config = ExpanderConfig {
            min_weight: 0.5,
            ..ExpanderConfig::default()
        };
        let mut exp = SemanticSynonymExpander::new(config);
        exp.add_synonym("x", "y", SynonymRelation::Exact, 0.5); // exactly at boundary
        let results = exp.expand("x", None);
        assert!(
            !results.is_empty(),
            "edge at exactly min_weight should be followed"
        );
    }

    // -----------------------------------------------------------------------
    // expand — allowed_relations filtering
    // -----------------------------------------------------------------------

    #[test]
    fn test_expand_filters_by_allowed_relations() {
        let mut exp = default_expander();
        exp.add_synonym("fruit", "apple", SynonymRelation::Narrower, 0.9);
        exp.add_synonym("fruit", "food", SynonymRelation::Broader, 0.9);
        // Only follow Narrower edges
        let results = exp.expand("fruit", Some(&[SynonymRelation::Narrower]));
        let terms: Vec<&str> = results.iter().map(|r| r.term.as_str()).collect();
        assert!(terms.contains(&"apple"));
        assert!(!terms.contains(&"food"));
    }

    #[test]
    fn test_expand_allowed_relations_none_means_all() {
        let mut exp = default_expander();
        exp.add_synonym("fruit", "apple", SynonymRelation::Narrower, 0.9);
        exp.add_synonym("fruit", "food", SynonymRelation::Broader, 0.9);
        let results = exp.expand("fruit", None);
        // Should include both directions
        assert!(results.len() >= 2);
    }

    // -----------------------------------------------------------------------
    // expand — cumulative_weight
    // -----------------------------------------------------------------------

    #[test]
    fn test_expand_cumulative_weight_product() {
        let config = ExpanderConfig {
            max_hops: 3,
            ..ExpanderConfig::default()
        };
        let mut exp = SemanticSynonymExpander::new(config);
        exp.add_synonym("a", "b", SynonymRelation::Exact, 0.8);
        exp.add_synonym("b", "c", SynonymRelation::Exact, 0.9);
        let results = exp.expand("a", None);
        let c = results
            .iter()
            .find(|r| r.term == "c")
            .expect("c should be found");
        let expected = 0.8_f32 * 0.9;
        assert!(
            (c.cumulative_weight - expected).abs() < 1e-5,
            "expected cumulative_weight ~ {expected}, got {}",
            c.cumulative_weight
        );
        assert_eq!(c.hops, 2);
    }

    // -----------------------------------------------------------------------
    // expand — max_expansions truncation
    // -----------------------------------------------------------------------

    #[test]
    fn test_expand_truncates_at_max_expansions() {
        let config = ExpanderConfig {
            max_expansions: 3,
            ..ExpanderConfig::default()
        };
        let mut exp = SemanticSynonymExpander::new(config);
        for i in 0..10 {
            exp.add_synonym("root", &format!("term{i}"), SynonymRelation::Near, 0.9);
        }
        let results = exp.expand("root", None);
        assert!(results.len() <= 3, "results exceeded max_expansions");
    }

    // -----------------------------------------------------------------------
    // expand — sorting
    // -----------------------------------------------------------------------

    #[test]
    fn test_expand_sorted_by_weight_descending() {
        let mut exp = default_expander();
        exp.add_synonym("root", "high", SynonymRelation::Exact, 0.95);
        exp.add_synonym("root", "low", SynonymRelation::Near, 0.6);
        let results = exp.expand("root", None);
        // High-weight term should come first
        assert_eq!(results[0].term, "high");
    }

    #[test]
    fn test_expand_sorted_alphabetically_on_weight_tie() {
        let mut exp = default_expander();
        exp.add_synonym("root", "beta", SynonymRelation::Exact, 0.8);
        exp.add_synonym("root", "alpha", SynonymRelation::Exact, 0.8);
        let results = exp.expand("root", None);
        // Same weight → alphabetical
        assert_eq!(results[0].term, "alpha");
        assert_eq!(results[1].term, "beta");
    }

    // -----------------------------------------------------------------------
    // expand — stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_expand_increments_total_expand_calls() {
        let mut exp = default_expander();
        exp.expand("unknown", None);
        exp.expand("unknown", None);
        assert_eq!(exp.stats.total_expand_calls, 2);
    }

    #[test]
    fn test_expand_accumulates_total_terms_expanded() {
        let mut exp = default_expander();
        exp.add_synonym("a", "b", SynonymRelation::Exact, 0.9);
        exp.add_synonym("a", "c", SynonymRelation::Exact, 0.9);
        let r = exp.expand("a", None);
        assert_eq!(exp.stats.total_terms_expanded, r.len() as u64);
        exp.expand("a", None);
        assert_eq!(exp.stats.total_terms_expanded, 2 * r.len() as u64);
    }

    // -----------------------------------------------------------------------
    // remove_term
    // -----------------------------------------------------------------------

    #[test]
    fn test_remove_term_removes_node() {
        let mut exp = default_expander();
        exp.add_synonym("a", "b", SynonymRelation::Exact, 0.9);
        exp.remove_term("b");
        assert!(!exp.graph.contains_key("b"));
    }

    #[test]
    fn test_remove_term_removes_incoming_edges() {
        let mut exp = default_expander();
        exp.add_synonym("a", "b", SynonymRelation::Exact, 0.9);
        exp.remove_term("b");
        // a should no longer have an edge to b
        assert!(!exp.graph["a"].iter().any(|e| e.target == "b"));
    }

    #[test]
    fn test_remove_term_updates_stats() {
        let mut exp = default_expander();
        exp.add_synonym("a", "b", SynonymRelation::Exact, 0.9);
        exp.remove_term("b");
        assert_eq!(exp.stats.total_terms, 1); // only "a" remains
        assert_eq!(exp.stats.total_edges, 0); // no more edges
    }

    #[test]
    fn test_remove_term_unknown_is_noop() {
        let mut exp = default_expander();
        exp.add_synonym("a", "b", SynonymRelation::Exact, 0.9);
        let terms_before = exp.stats.total_terms;
        let edges_before = exp.stats.total_edges;
        exp.remove_term("ghost");
        assert_eq!(exp.stats.total_terms, terms_before);
        assert_eq!(exp.stats.total_edges, edges_before);
    }

    // -----------------------------------------------------------------------
    // term_count
    // -----------------------------------------------------------------------

    #[test]
    fn test_term_count() {
        let mut exp = default_expander();
        assert_eq!(exp.term_count(), 0);
        exp.add_synonym("a", "b", SynonymRelation::Exact, 0.9);
        assert_eq!(exp.term_count(), 2);
    }

    // -----------------------------------------------------------------------
    // stats accessor
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_accessor_returns_ref() {
        let exp = default_expander();
        let s = exp.stats();
        assert_eq!(s.total_expand_calls, 0);
    }

    // -----------------------------------------------------------------------
    // ExpanderConfig defaults
    // -----------------------------------------------------------------------

    #[test]
    fn test_expander_config_defaults() {
        let cfg = ExpanderConfig::default();
        assert_eq!(cfg.max_hops, 2);
        assert!((cfg.min_weight - 0.5).abs() < f32::EPSILON);
        assert_eq!(cfg.max_expansions, 20);
    }

    // -----------------------------------------------------------------------
    // Multi-hop path not revisited
    // -----------------------------------------------------------------------

    #[test]
    fn test_expand_no_revisit_cycle() {
        let mut exp = default_expander();
        // a ↔ b ↔ a is already prevented by visited set
        // Triangle: a - b - c - a
        exp.add_synonym("a", "b", SynonymRelation::Exact, 0.9);
        exp.add_synonym("b", "c", SynonymRelation::Exact, 0.9);
        exp.add_synonym("c", "a", SynonymRelation::Exact, 0.9);
        let results = exp.expand("a", None);
        // Should not contain "a" and should not loop infinitely
        assert!(!results.iter().any(|r| r.term == "a"));
    }
}
