//! # Semantic Concept Hierarchy
//!
//! Models a concept ontology as a directed acyclic graph (DAG) where concepts
//! are linked by "is-a" (hypernym/hyponym) and "related-to" relationships,
//! enabling hierarchical search expansion.

use std::collections::{HashMap, HashSet, VecDeque};

/// The kind of directed relationship between two concepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConceptRelation {
    /// The `from` concept is a subtype of the `to` concept (hypernym relationship).
    IsA,
    /// Bidirectional semantic similarity between the two concepts.
    RelatedTo,
    /// Antonym relationship between the two concepts.
    OppositeOf,
}

/// A directed, weighted edge between two concepts in the ontology.
#[derive(Clone, Debug)]
pub struct ConceptEdge {
    /// Source concept name.
    pub from: String,
    /// Target concept name.
    pub to: String,
    /// The semantic relationship type.
    pub relation: ConceptRelation,
    /// Strength of the relationship, clamped to [0.0, 1.0].
    pub weight: f32,
}

/// A concept node in the ontology graph.
#[derive(Clone, Debug)]
pub struct ConceptNode {
    /// Unique name / identifier of the concept.
    pub name: String,
    /// Optional prototype embedding vector for the concept.
    pub embedding: Option<Vec<f32>>,
    /// Free-form human-readable description.
    pub metadata: String,
}

/// Aggregate statistics about the hierarchy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HierarchyStats {
    /// Total number of concept nodes.
    pub total_concepts: usize,
    /// Total number of edges (all relation types).
    pub total_edges: usize,
    /// Length of the longest IsA path from any root concept.
    pub max_depth: usize,
    /// Number of concepts that have no incoming IsA edges (i.e., root concepts).
    pub root_count: usize,
}

/// A concept ontology modelled as a DAG supporting hierarchical search expansion.
///
/// Concepts are linked by [`ConceptRelation::IsA`], [`ConceptRelation::RelatedTo`],
/// and [`ConceptRelation::OppositeOf`] edges.  The IsA relation induces a partial
/// order from which roots, parents, children, and transitive ancestors are derived.
pub struct SemanticConceptHierarchy {
    /// All known concepts, keyed by name.
    pub concepts: HashMap<String, ConceptNode>,
    /// All edges in the ontology.
    pub edges: Vec<ConceptEdge>,
}

// ── construction ────────────────────────────────────────────────────────────

impl SemanticConceptHierarchy {
    /// Create an empty hierarchy.
    pub fn new() -> Self {
        Self {
            concepts: HashMap::new(),
            edges: Vec::new(),
        }
    }

    /// Add a concept node.  If a concept with the same name already exists it is
    /// left unchanged (idempotent on name).
    pub fn add_concept(&mut self, node: ConceptNode) {
        self.concepts.entry(node.name.clone()).or_insert(node);
    }

    /// Add an edge.  Both the `from` and `to` concepts are automatically
    /// registered with empty metadata if they are not already present.
    pub fn add_edge(&mut self, edge: ConceptEdge) {
        // Auto-register missing endpoints.
        for name in [edge.from.clone(), edge.to.clone()] {
            self.concepts
                .entry(name.clone())
                .or_insert_with(|| ConceptNode {
                    name,
                    embedding: None,
                    metadata: String::new(),
                });
        }
        self.edges.push(edge);
    }
}

// ── Default ─────────────────────────────────────────────────────────────────

impl Default for SemanticConceptHierarchy {
    fn default() -> Self {
        Self::new()
    }
}

// ── traversal helpers ────────────────────────────────────────────────────────

impl SemanticConceptHierarchy {
    /// Direct IsA parents of `concept` (edges where `from == concept` and
    /// relation is [`ConceptRelation::IsA`]), sorted alphabetically.
    pub fn parents_of<'a>(&'a self, concept: &str) -> Vec<&'a str> {
        let mut parents: Vec<&str> = self
            .edges
            .iter()
            .filter(|e| e.relation == ConceptRelation::IsA && e.from == concept)
            .map(|e| e.to.as_str())
            .collect();
        parents.sort_unstable();
        parents.dedup();
        parents
    }

    /// Direct IsA children of `concept` (edges where `to == concept` and
    /// relation is [`ConceptRelation::IsA`]), sorted alphabetically.
    pub fn children_of<'a>(&'a self, concept: &str) -> Vec<&'a str> {
        let mut children: Vec<&str> = self
            .edges
            .iter()
            .filter(|e| e.relation == ConceptRelation::IsA && e.to == concept)
            .map(|e| e.from.as_str())
            .collect();
        children.sort_unstable();
        children.dedup();
        children
    }

    /// All transitive IsA ancestors of `concept` via BFS.
    ///
    /// The result is sorted alphabetically, contains no duplicates, and does
    /// not include `concept` itself.  Safe against DAGs with multiple paths
    /// to the same ancestor (no infinite loop).
    pub fn ancestors_of(&self, concept: &str) -> Vec<String> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();

        // Seed queue with direct parents.
        for parent in self.parents_of(concept) {
            if visited.insert(parent.to_string()) {
                queue.push_back(parent.to_string());
            }
        }

        while let Some(current) = queue.pop_front() {
            for parent in self.parents_of(&current) {
                if visited.insert(parent.to_string()) {
                    queue.push_back(parent.to_string());
                }
            }
        }

        let mut result: Vec<String> = visited.into_iter().collect();
        result.sort_unstable();
        result
    }

    /// Direct [`ConceptRelation::RelatedTo`] neighbors of `concept` in either
    /// direction, sorted alphabetically and deduplicated.
    pub fn related_to<'a>(&'a self, concept: &str) -> Vec<&'a str> {
        let mut neighbors: Vec<&str> = self
            .edges
            .iter()
            .filter(|e| e.relation == ConceptRelation::RelatedTo)
            .filter_map(|e| {
                if e.from == concept {
                    Some(e.to.as_str())
                } else if e.to == concept {
                    Some(e.from.as_str())
                } else {
                    None
                }
            })
            .collect();
        neighbors.sort_unstable();
        neighbors.dedup();
        neighbors
    }

    /// Expand a query concept into a set of related concepts for recall improvement.
    ///
    /// Performs a BFS over IsA edges in **both** directions (ancestors and
    /// descendants) up to `depth` hops, then adds all direct
    /// [`ConceptRelation::RelatedTo`] neighbours.  The concept itself is always
    /// included.  The result is deduplicated and sorted alphabetically.
    ///
    /// When `depth == 0` only the concept itself is returned.
    pub fn expand_query(&self, concept: &str, depth: usize) -> Vec<String> {
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(concept.to_string());

        if depth > 0 {
            // BFS over IsA edges in both directions (up and down).
            let mut queue: VecDeque<(String, usize)> = VecDeque::new();
            queue.push_back((concept.to_string(), 0));

            while let Some((current, current_depth)) = queue.pop_front() {
                if current_depth >= depth {
                    continue;
                }

                // Upward (ancestors)
                for parent in self.parents_of(&current) {
                    if visited.insert(parent.to_string()) {
                        queue.push_back((parent.to_string(), current_depth + 1));
                    }
                }

                // Downward (descendants)
                for child in self.children_of(&current) {
                    if visited.insert(child.to_string()) {
                        queue.push_back((child.to_string(), current_depth + 1));
                    }
                }
            }
        }

        // Include direct RelatedTo neighbours regardless of depth.
        for neighbor in self.related_to(concept) {
            visited.insert(neighbor.to_string());
        }

        let mut result: Vec<String> = visited.into_iter().collect();
        result.sort_unstable();
        result
    }

    /// Compute aggregate statistics for the hierarchy.
    pub fn stats(&self) -> HierarchyStats {
        let total_concepts = self.concepts.len();
        let total_edges = self.edges.len();

        // A root is a concept that has no IsA parent, i.e. it never appears as
        // the `from` end of an IsA edge.  In the convention "from IsA to",
        // `from` is the child/subtype and `to` is the parent/supertype, so a
        // root (top of the hierarchy) is a concept that is never a child.
        let has_isa_parent: HashSet<&str> = self
            .edges
            .iter()
            .filter(|e| e.relation == ConceptRelation::IsA)
            .map(|e| e.from.as_str())
            .collect();

        let roots: Vec<&str> = self
            .concepts
            .keys()
            .map(|k| k.as_str())
            .filter(|k| !has_isa_parent.contains(*k))
            .collect();

        let root_count = roots.len();

        // BFS from every root, tracking the maximum path length reached.
        let max_depth = if total_concepts == 0 {
            0
        } else {
            let mut global_max = 0usize;
            for root in roots {
                // (concept_name, depth_from_root)
                let mut queue: VecDeque<(&str, usize)> = VecDeque::new();
                let mut local_visited: HashSet<&str> = HashSet::new();
                queue.push_back((root, 0));
                local_visited.insert(root);

                while let Some((current, depth)) = queue.pop_front() {
                    if depth > global_max {
                        global_max = depth;
                    }
                    for child in self.children_of(current) {
                        if local_visited.insert(child) {
                            queue.push_back((child, depth + 1));
                        }
                    }
                }
            }
            global_max
        };

        HierarchyStats {
            total_concepts,
            total_edges,
            max_depth,
            root_count,
        }
    }

    /// Remove a concept and all edges that connect to or from it.
    ///
    /// Returns `true` if the concept existed and was removed, `false` if it
    /// was not found.
    pub fn remove_concept(&mut self, name: &str) -> bool {
        if self.concepts.remove(name).is_none() {
            return false;
        }
        self.edges.retain(|e| e.from != name && e.to != name);
        true
    }
}

// ── tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────

    fn node(name: &str) -> ConceptNode {
        ConceptNode {
            name: name.to_string(),
            embedding: None,
            metadata: name.to_string(),
        }
    }

    fn isa(from: &str, to: &str) -> ConceptEdge {
        ConceptEdge {
            from: from.to_string(),
            to: to.to_string(),
            relation: ConceptRelation::IsA,
            weight: 1.0,
        }
    }

    fn related(a: &str, b: &str) -> ConceptEdge {
        ConceptEdge {
            from: a.to_string(),
            to: b.to_string(),
            relation: ConceptRelation::RelatedTo,
            weight: 0.8,
        }
    }

    // ── test 1: new() starts empty ────────────────────────────────────────
    #[test]
    fn test_new_starts_empty() {
        let h = SemanticConceptHierarchy::new();
        assert!(h.concepts.is_empty());
        assert!(h.edges.is_empty());
    }

    // ── test 2: add_concept idempotent ───────────────────────────────────
    #[test]
    fn test_add_concept_idempotent() {
        let mut h = SemanticConceptHierarchy::new();
        h.add_concept(ConceptNode {
            name: "animal".to_string(),
            embedding: Some(vec![1.0, 2.0]),
            metadata: "original".to_string(),
        });
        // Second add must not overwrite.
        h.add_concept(ConceptNode {
            name: "animal".to_string(),
            embedding: None,
            metadata: "overwrite-attempt".to_string(),
        });
        assert_eq!(h.concepts.len(), 1);
        assert_eq!(h.concepts["animal"].metadata, "original");
    }

    // ── test 3: add_edge auto-registers missing concepts ─────────────────
    #[test]
    fn test_add_edge_auto_registers_concepts() {
        let mut h = SemanticConceptHierarchy::new();
        h.add_edge(isa("dog", "animal"));
        assert!(h.concepts.contains_key("dog"));
        assert!(h.concepts.contains_key("animal"));
        assert_eq!(h.edges.len(), 1);
    }

    // ── test 4: parents_of returns direct IsA parents ────────────────────
    #[test]
    fn test_parents_of_direct() {
        let mut h = SemanticConceptHierarchy::new();
        h.add_edge(isa("dog", "animal"));
        h.add_edge(isa("dog", "mammal"));
        let parents = h.parents_of("dog");
        assert_eq!(parents, vec!["animal", "mammal"]);
    }

    // ── test 5: parents_of empty for root ────────────────────────────────
    #[test]
    fn test_parents_of_empty_for_root() {
        let mut h = SemanticConceptHierarchy::new();
        h.add_concept(node("entity"));
        assert!(h.parents_of("entity").is_empty());
    }

    // ── test 6: children_of returns direct IsA children ──────────────────
    #[test]
    fn test_children_of_direct() {
        let mut h = SemanticConceptHierarchy::new();
        h.add_edge(isa("dog", "animal"));
        h.add_edge(isa("cat", "animal"));
        let mut children = h.children_of("animal");
        children.sort_unstable();
        assert_eq!(children, vec!["cat", "dog"]);
    }

    // ── test 7: ancestors_of transitively includes grandparents ──────────
    #[test]
    fn test_ancestors_of_transitive() {
        let mut h = SemanticConceptHierarchy::new();
        // poodle -> dog -> animal -> entity
        h.add_edge(isa("poodle", "dog"));
        h.add_edge(isa("dog", "animal"));
        h.add_edge(isa("animal", "entity"));

        let ancestors = h.ancestors_of("poodle");
        assert!(ancestors.contains(&"dog".to_string()));
        assert!(ancestors.contains(&"animal".to_string()));
        assert!(ancestors.contains(&"entity".to_string()));
        assert!(!ancestors.contains(&"poodle".to_string()));
    }

    // ── test 8: ancestors_of returns empty for root ───────────────────────
    #[test]
    fn test_ancestors_of_empty_for_root() {
        let mut h = SemanticConceptHierarchy::new();
        h.add_concept(node("entity"));
        assert!(h.ancestors_of("entity").is_empty());
    }

    // ── test 9: related_to both directions ───────────────────────────────
    #[test]
    fn test_related_to_both_directions() {
        let mut h = SemanticConceptHierarchy::new();
        h.add_edge(related("cat", "tiger")); // forward
        h.add_edge(related("lion", "cat")); // backward

        let related_to_cat = h.related_to("cat");
        assert!(related_to_cat.contains(&"tiger"));
        assert!(related_to_cat.contains(&"lion"));
    }

    // ── test 10: related_to no duplicates ────────────────────────────────
    #[test]
    fn test_related_to_no_duplicates() {
        let mut h = SemanticConceptHierarchy::new();
        // Two RelatedTo edges in both directions between the same pair.
        h.add_edge(related("a", "b"));
        h.add_edge(related("b", "a"));

        let rel_a = h.related_to("a");
        let count_b = rel_a.iter().filter(|&&x| x == "b").count();
        assert_eq!(count_b, 1, "should deduplicate 'b'");
    }

    // ── test 11: expand_query includes self ───────────────────────────────
    #[test]
    fn test_expand_query_includes_self() {
        let mut h = SemanticConceptHierarchy::new();
        h.add_concept(node("alpha"));
        let expanded = h.expand_query("alpha", 2);
        assert!(expanded.contains(&"alpha".to_string()));
    }

    // ── test 12: expand_query depth=0 returns only self ──────────────────
    #[test]
    fn test_expand_query_depth_zero() {
        let mut h = SemanticConceptHierarchy::new();
        h.add_edge(isa("dog", "animal"));
        // No RelatedTo edges, so depth 0 must give only "dog".
        let expanded = h.expand_query("dog", 0);
        assert_eq!(expanded, vec!["dog".to_string()]);
    }

    // ── test 13: expand_query depth=1 includes parents and children ───────
    #[test]
    fn test_expand_query_depth_one_parents_children() {
        let mut h = SemanticConceptHierarchy::new();
        h.add_edge(isa("dog", "animal"));
        h.add_edge(isa("poodle", "dog"));

        let expanded = h.expand_query("dog", 1);
        assert!(expanded.contains(&"dog".to_string()));
        assert!(expanded.contains(&"animal".to_string())); // parent
        assert!(expanded.contains(&"poodle".to_string())); // child
    }

    // ── test 14: expand_query includes RelatedTo neighbors ────────────────
    #[test]
    fn test_expand_query_includes_related_to() {
        let mut h = SemanticConceptHierarchy::new();
        h.add_concept(node("cat"));
        h.add_edge(related("cat", "tiger"));

        let expanded = h.expand_query("cat", 0);
        assert!(expanded.contains(&"cat".to_string()));
        assert!(expanded.contains(&"tiger".to_string()));
    }

    // ── test 15: expand_query sorted and deduped ──────────────────────────
    #[test]
    fn test_expand_query_sorted_deduped() {
        let mut h = SemanticConceptHierarchy::new();
        h.add_edge(isa("dog", "animal"));
        h.add_edge(isa("dog", "mammal"));
        h.add_edge(related("dog", "wolf"));

        let expanded = h.expand_query("dog", 2);
        let mut sorted = expanded.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            expanded, sorted,
            "expand_query must return sorted, deduped results"
        );
    }

    // ── test 16: stats root_count correct ────────────────────────────────
    #[test]
    fn test_stats_root_count() {
        let mut h = SemanticConceptHierarchy::new();
        h.add_concept(node("entity")); // root
        h.add_concept(node("thing")); // root
        h.add_edge(isa("dog", "entity"));
        let stats = h.stats();
        assert_eq!(stats.root_count, 2);
    }

    // ── test 17: stats max_depth for linear chain ─────────────────────────
    #[test]
    fn test_stats_max_depth_linear_chain() {
        let mut h = SemanticConceptHierarchy::new();
        // entity -> animal -> mammal -> dog -> poodle  (depth 4 from root)
        h.add_edge(isa("animal", "entity"));
        h.add_edge(isa("mammal", "animal"));
        h.add_edge(isa("dog", "mammal"));
        h.add_edge(isa("poodle", "dog"));

        let stats = h.stats();
        assert_eq!(stats.max_depth, 4);
    }

    // ── test 18: stats total_concepts and total_edges ─────────────────────
    #[test]
    fn test_stats_totals() {
        let mut h = SemanticConceptHierarchy::new();
        h.add_concept(node("a"));
        h.add_concept(node("b"));
        h.add_edge(isa("a", "b"));
        h.add_edge(related("a", "b"));

        let stats = h.stats();
        assert_eq!(stats.total_concepts, 2);
        assert_eq!(stats.total_edges, 2);
    }

    // ── test 19: remove_concept removes node ─────────────────────────────
    #[test]
    fn test_remove_concept_removes_node() {
        let mut h = SemanticConceptHierarchy::new();
        h.add_concept(node("dog"));
        assert!(h.remove_concept("dog"));
        assert!(!h.concepts.contains_key("dog"));
    }

    // ── test 20: remove_concept removes connected edges ───────────────────
    #[test]
    fn test_remove_concept_removes_edges() {
        let mut h = SemanticConceptHierarchy::new();
        h.add_edge(isa("dog", "animal"));
        h.add_edge(related("dog", "wolf"));
        h.add_edge(isa("poodle", "dog"));

        assert!(h.remove_concept("dog"));
        assert!(
            h.edges.is_empty(),
            "all edges touching 'dog' must be removed"
        );
    }

    // ── test 21: remove_concept returns false for unknown ─────────────────
    #[test]
    fn test_remove_concept_false_for_unknown() {
        let mut h = SemanticConceptHierarchy::new();
        assert!(!h.remove_concept("nonexistent"));
    }

    // ── test 22: ancestors_of no infinite loop in DAG with multiple paths ─
    #[test]
    fn test_ancestors_no_infinite_loop_multi_path() {
        // Diamond: poodle -> dog -> animal -> entity
        //                  mammal -> entity
        //          dog -> mammal
        let mut h = SemanticConceptHierarchy::new();
        h.add_edge(isa("poodle", "dog"));
        h.add_edge(isa("dog", "animal"));
        h.add_edge(isa("dog", "mammal"));
        h.add_edge(isa("animal", "entity"));
        h.add_edge(isa("mammal", "entity"));

        let ancestors = h.ancestors_of("poodle");
        // Must contain all four ancestors exactly once.
        let count_entity = ancestors.iter().filter(|x| x.as_str() == "entity").count();
        assert_eq!(count_entity, 1, "'entity' must appear exactly once");
        assert_eq!(ancestors.len(), 4, "dog, animal, mammal, entity");
    }

    // ── test 23: expand_query depth>1 traverses multiple hops ────────────
    #[test]
    fn test_expand_query_multi_hop() {
        let mut h = SemanticConceptHierarchy::new();
        // poodle -> dog -> animal -> entity
        h.add_edge(isa("poodle", "dog"));
        h.add_edge(isa("dog", "animal"));
        h.add_edge(isa("animal", "entity"));

        // depth 3 from "dog": should reach poodle (down 1), animal (up 1), entity (up 2)
        let expanded = h.expand_query("dog", 3);
        assert!(expanded.contains(&"poodle".to_string()));
        assert!(expanded.contains(&"animal".to_string()));
        assert!(expanded.contains(&"entity".to_string()));
    }

    // ── test 24: HierarchyStats equality ─────────────────────────────────
    #[test]
    fn test_hierarchy_stats_eq() {
        let s1 = HierarchyStats {
            total_concepts: 3,
            total_edges: 2,
            max_depth: 2,
            root_count: 1,
        };
        let s2 = s1.clone();
        assert_eq!(s1, s2);
    }

    // ── test 25: ConceptRelation derives ─────────────────────────────────
    #[test]
    fn test_concept_relation_derives() {
        let r = ConceptRelation::IsA;
        let r2 = r; // Copy
        assert_eq!(r, r2);
        assert_ne!(ConceptRelation::IsA, ConceptRelation::RelatedTo);
        assert_ne!(ConceptRelation::RelatedTo, ConceptRelation::OppositeOf);
    }
}
