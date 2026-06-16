//! # Semantic Knowledge Graph
//!
//! Links concepts, entities, and their embeddings into a queryable graph
//! for multi-hop semantic reasoning.

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// EntityKind
// ---------------------------------------------------------------------------

/// The category of a graph entity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EntityKind {
    Concept,
    Document,
    Person,
    Organization,
    Event,
}

// ---------------------------------------------------------------------------
// GraphEntity
// ---------------------------------------------------------------------------

/// A node in the knowledge graph.
#[derive(Clone, Debug)]
pub struct GraphEntity {
    /// Unique identifier.
    pub entity_id: u64,
    /// Human-readable name.
    pub name: String,
    /// Category of the entity.
    pub kind: EntityKind,
    /// Optional dense embedding vector.
    pub embedding: Option<Vec<f32>>,
    /// Arbitrary key-value properties.
    pub properties: Vec<(String, String)>,
}

// ---------------------------------------------------------------------------
// GraphEdge
// ---------------------------------------------------------------------------

/// A directed, weighted relationship between two entities.
#[derive(Clone, Debug)]
pub struct GraphEdge {
    /// Unique identifier.
    pub edge_id: u64,
    /// Source entity id.
    pub from_id: u64,
    /// Destination entity id.
    pub to_id: u64,
    /// Relation label, e.g. "is_about", "authored_by", "mentions".
    pub relation: String,
    /// Edge weight (higher = stronger association).
    pub weight: f32,
}

// ---------------------------------------------------------------------------
// GraphQuery
// ---------------------------------------------------------------------------

/// Parameters for a multi-hop BFS traversal.
#[derive(Clone, Debug)]
pub struct GraphQuery {
    /// Entity from which traversal starts.
    pub start_entity_id: u64,
    /// If `Some`, only traverse edges whose relation equals this string.
    pub relation_filter: Option<String>,
    /// Maximum number of hops from the start entity.
    pub max_hops: usize,
    /// If `Some`, only include entities of this kind in the results.
    pub entity_kind_filter: Option<EntityKind>,
}

// ---------------------------------------------------------------------------
// KnowledgeGraphStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for a [`SemanticKnowledgeGraph`].
#[derive(Clone, Debug)]
pub struct KnowledgeGraphStats {
    /// Number of entities stored in the graph.
    pub total_entities: usize,
    /// Number of directed edges stored.
    pub total_edges: usize,
    /// Number of entities that carry an embedding vector.
    pub entities_with_embeddings: usize,
    /// Undirected-approximation average degree: `total_edges * 2 / total_entities`.
    /// Returns `0.0` when the graph has no entities.
    pub avg_degree: f64,
}

// ---------------------------------------------------------------------------
// Cosine similarity helper
// ---------------------------------------------------------------------------

/// Compute the cosine similarity between two equal-length slices.
///
/// Returns `0.0` when either vector has zero magnitude.
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let mut dot = 0.0_f32;
    let mut mag_a = 0.0_f32;
    let mut mag_b = 0.0_f32;

    for i in 0..len {
        dot += a[i] * b[i];
        mag_a += a[i] * a[i];
        mag_b += b[i] * b[i];
    }

    let denom = mag_a.sqrt() * mag_b.sqrt();
    if denom < f32::EPSILON {
        0.0
    } else {
        dot / denom
    }
}

// ---------------------------------------------------------------------------
// SemanticKnowledgeGraph
// ---------------------------------------------------------------------------

/// A queryable semantic knowledge graph that links entities via typed, weighted edges.
///
/// Supports:
/// - Adding/removing entities and edges.
/// - Neighbour lookup with optional relation filtering (sorted by weight desc).
/// - BFS traversal with relation and entity-kind filters.
/// - Cosine-similarity–based entity retrieval for entities that carry embeddings.
pub struct SemanticKnowledgeGraph {
    /// All entities, keyed by their `entity_id`.
    pub entities: HashMap<u64, GraphEntity>,
    /// All directed edges in insertion order.
    pub edges: Vec<GraphEdge>,
    /// Monotonically increasing counter for entity ids.
    pub next_entity_id: u64,
    /// Monotonically increasing counter for edge ids.
    pub next_edge_id: u64,
}

impl SemanticKnowledgeGraph {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Create a new, empty knowledge graph.
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            edges: Vec::new(),
            next_entity_id: 0,
            next_edge_id: 0,
        }
    }

    // ------------------------------------------------------------------
    // Mutation
    // ------------------------------------------------------------------

    /// Add a new entity and return its freshly assigned `entity_id`.
    pub fn add_entity(&mut self, name: &str, kind: EntityKind, embedding: Option<Vec<f32>>) -> u64 {
        let id = self.next_entity_id;
        self.next_entity_id += 1;

        self.entities.insert(
            id,
            GraphEntity {
                entity_id: id,
                name: name.to_owned(),
                kind,
                embedding,
                properties: Vec::new(),
            },
        );

        id
    }

    /// Add a directed edge between two entities and return its `edge_id`.
    ///
    /// The entities referenced by `from_id` and `to_id` need not exist yet;
    /// no validation is performed here so that callers can build the graph in
    /// any order.
    pub fn add_edge(&mut self, from_id: u64, to_id: u64, relation: &str, weight: f32) -> u64 {
        let id = self.next_edge_id;
        self.next_edge_id += 1;

        self.edges.push(GraphEdge {
            edge_id: id,
            from_id,
            to_id,
            relation: relation.to_owned(),
            weight,
        });

        id
    }

    /// Look up an entity by id.
    pub fn get_entity(&self, id: u64) -> Option<&GraphEntity> {
        self.entities.get(&id)
    }

    /// Remove an entity and all edges that touch it.
    ///
    /// Returns `true` if the entity existed, `false` otherwise.
    pub fn remove_entity(&mut self, entity_id: u64) -> bool {
        if self.entities.remove(&entity_id).is_none() {
            return false;
        }
        self.edges
            .retain(|e| e.from_id != entity_id && e.to_id != entity_id);
        true
    }

    // ------------------------------------------------------------------
    // Query
    // ------------------------------------------------------------------

    /// Return all entities reachable via a single outgoing edge from `entity_id`.
    ///
    /// - When `relation` is `Some(r)`, only edges whose `relation == r` are
    ///   considered.
    /// - The result list is sorted by edge weight descending.
    pub fn neighbors(&self, entity_id: u64, relation: Option<&str>) -> Vec<&GraphEntity> {
        // Collect (weight, entity) pairs.
        let mut candidates: Vec<(f32, &GraphEntity)> = self
            .edges
            .iter()
            .filter(|e| e.from_id == entity_id && relation.is_none_or(|r| e.relation == r))
            .filter_map(|e| self.entities.get(&e.to_id).map(|entity| (e.weight, entity)))
            .collect();

        // Sort by weight descending; use entity_id as tiebreaker for determinism.
        candidates.sort_by(|(wa, ea), (wb, eb)| {
            wb.partial_cmp(wa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| ea.entity_id.cmp(&eb.entity_id))
        });

        candidates.into_iter().map(|(_, e)| e).collect()
    }

    /// BFS traversal from `query.start_entity_id` up to `query.max_hops` hops.
    ///
    /// At every hop the relation filter and entity-kind filter (from the
    /// [`GraphQuery`]) are applied to the *destination* entity.  The start
    /// entity itself is excluded from the result.  Results are sorted by
    /// `entity_id` ascending.
    pub fn traverse(&self, query: &GraphQuery) -> Vec<&GraphEntity> {
        let mut visited: HashSet<u64> = HashSet::new();
        let mut result_ids: Vec<u64> = Vec::new();

        visited.insert(query.start_entity_id);

        // Queue stores (entity_id, remaining_hops).
        let mut queue: VecDeque<(u64, usize)> = VecDeque::new();
        queue.push_back((query.start_entity_id, query.max_hops));

        while let Some((current_id, hops_left)) = queue.pop_front() {
            if hops_left == 0 {
                continue;
            }

            for edge in self.edges.iter().filter(|e| e.from_id == current_id) {
                // Apply relation filter.
                if let Some(ref rel) = query.relation_filter {
                    if &edge.relation != rel {
                        continue;
                    }
                }

                let dest_id = edge.to_id;

                if visited.contains(&dest_id) {
                    continue;
                }
                visited.insert(dest_id);

                // Apply entity-kind filter: the destination must exist and
                // match the requested kind (if any).
                if let Some(entity) = self.entities.get(&dest_id) {
                    let kind_ok = query.entity_kind_filter.is_none_or(|k| entity.kind == k);
                    if kind_ok {
                        result_ids.push(dest_id);
                    }
                    // Continue BFS regardless of the kind filter so that we
                    // can discover matching entities further in the graph.
                    queue.push_back((dest_id, hops_left - 1));
                }
            }
        }

        result_ids.sort_unstable();
        result_ids
            .iter()
            .filter_map(|id| self.entities.get(id))
            .collect()
    }

    /// Find all entities (excluding `entity_id` itself) whose embedding has a
    /// cosine similarity ≥ `threshold` with the embedding of `entity_id`.
    ///
    /// Returns an empty vec if the query entity does not exist or has no
    /// embedding.  Results are sorted by similarity descending.
    pub fn similar_entities(&self, entity_id: u64, threshold: f32) -> Vec<(&GraphEntity, f32)> {
        let query_emb = match self
            .entities
            .get(&entity_id)
            .and_then(|e| e.embedding.as_ref())
        {
            Some(emb) => emb,
            None => return Vec::new(),
        };

        let mut results: Vec<(&GraphEntity, f32)> = self
            .entities
            .values()
            .filter(|e| e.entity_id != entity_id)
            .filter_map(|e| {
                e.embedding
                    .as_ref()
                    .map(|emb| (e, cosine_sim(query_emb, emb)))
            })
            .filter(|(_, sim)| *sim >= threshold)
            .collect();

        results.sort_by(|(_, sa), (_, sb)| sb.partial_cmp(sa).unwrap_or(std::cmp::Ordering::Equal));

        results
    }

    // ------------------------------------------------------------------
    // Statistics
    // ------------------------------------------------------------------

    /// Return aggregate statistics for this graph.
    pub fn stats(&self) -> KnowledgeGraphStats {
        let total_entities = self.entities.len();
        let total_edges = self.edges.len();
        let entities_with_embeddings = self
            .entities
            .values()
            .filter(|e| e.embedding.is_some())
            .count();

        let avg_degree = if total_entities == 0 {
            0.0
        } else {
            (total_edges as f64 * 2.0) / total_entities as f64
        };

        KnowledgeGraphStats {
            total_entities,
            total_edges,
            entities_with_embeddings,
            avg_degree,
        }
    }
}

impl Default for SemanticKnowledgeGraph {
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

    // ------------------------------------------------------------------
    // Helper builders
    // ------------------------------------------------------------------

    fn small_graph() -> SemanticKnowledgeGraph {
        let mut g = SemanticKnowledgeGraph::new();
        // 0: Concept "AI"
        let ai = g.add_entity("AI", EntityKind::Concept, Some(vec![1.0, 0.0]));
        // 1: Document "ML paper"
        let paper = g.add_entity("ML paper", EntityKind::Document, Some(vec![0.9, 0.1]));
        // 2: Person "Alice"
        let alice = g.add_entity("Alice", EntityKind::Person, Some(vec![0.0, 1.0]));
        // 3: Organization "OpenAI"
        let openai = g.add_entity("OpenAI", EntityKind::Organization, None);
        // Edges
        g.add_edge(ai, paper, "is_about", 0.9);
        g.add_edge(paper, alice, "authored_by", 0.8);
        g.add_edge(ai, openai, "mentions", 0.5);
        g
    }

    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    #[test]
    fn test_new_starts_empty() {
        let g = SemanticKnowledgeGraph::new();
        assert!(g.entities.is_empty());
        assert!(g.edges.is_empty());
        assert_eq!(g.next_entity_id, 0);
        assert_eq!(g.next_edge_id, 0);
    }

    // ------------------------------------------------------------------
    // add_entity
    // ------------------------------------------------------------------

    #[test]
    fn test_add_entity_returns_incrementing_ids() {
        let mut g = SemanticKnowledgeGraph::new();
        let id0 = g.add_entity("A", EntityKind::Concept, None);
        let id1 = g.add_entity("B", EntityKind::Document, None);
        let id2 = g.add_entity("C", EntityKind::Person, None);
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[test]
    fn test_add_entity_stores_properties() {
        let mut g = SemanticKnowledgeGraph::new();
        let id = g.add_entity("E", EntityKind::Event, Some(vec![0.1, 0.2]));
        let e = g.get_entity(id).expect("entity must exist");
        assert_eq!(e.name, "E");
        assert_eq!(e.kind, EntityKind::Event);
        assert_eq!(e.embedding, Some(vec![0.1, 0.2]));
        assert!(e.properties.is_empty());
    }

    // ------------------------------------------------------------------
    // add_edge
    // ------------------------------------------------------------------

    #[test]
    fn test_add_edge_stores_edge() {
        let mut g = SemanticKnowledgeGraph::new();
        let a = g.add_entity("A", EntityKind::Concept, None);
        let b = g.add_entity("B", EntityKind::Concept, None);
        let eid = g.add_edge(a, b, "related", 0.7);
        assert_eq!(eid, 0);
        assert_eq!(g.edges.len(), 1);
        let edge = &g.edges[0];
        assert_eq!(edge.from_id, a);
        assert_eq!(edge.to_id, b);
        assert_eq!(edge.relation, "related");
        assert!((edge.weight - 0.7).abs() < f32::EPSILON);
    }

    // ------------------------------------------------------------------
    // get_entity
    // ------------------------------------------------------------------

    #[test]
    fn test_get_entity_some() {
        let g = small_graph();
        assert!(g.get_entity(0).is_some());
        assert!(g.get_entity(1).is_some());
    }

    #[test]
    fn test_get_entity_none() {
        let g = small_graph();
        assert!(g.get_entity(999).is_none());
    }

    // ------------------------------------------------------------------
    // neighbors
    // ------------------------------------------------------------------

    #[test]
    fn test_neighbors_returns_correct_entities() {
        let g = small_graph();
        // Entity 0 (AI) has outgoing edges to 1 (paper) and 3 (openai).
        let nbrs = g.neighbors(0, None);
        let ids: Vec<u64> = nbrs.iter().map(|e| e.entity_id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
    }

    #[test]
    fn test_neighbors_with_relation_filter() {
        let g = small_graph();
        let nbrs = g.neighbors(0, Some("is_about"));
        assert_eq!(nbrs.len(), 1);
        assert_eq!(nbrs[0].entity_id, 1);
    }

    #[test]
    fn test_neighbors_sorted_by_weight_desc() {
        let mut g = SemanticKnowledgeGraph::new();
        let src = g.add_entity("src", EntityKind::Concept, None);
        let a = g.add_entity("a", EntityKind::Concept, None);
        let b = g.add_entity("b", EntityKind::Concept, None);
        let c = g.add_entity("c", EntityKind::Concept, None);
        g.add_edge(src, a, "r", 0.3);
        g.add_edge(src, b, "r", 0.9);
        g.add_edge(src, c, "r", 0.6);
        let nbrs = g.neighbors(src, None);
        let weights: Vec<f32> = nbrs
            .iter()
            .map(|e| {
                g.edges
                    .iter()
                    .find(|edge| edge.from_id == src && edge.to_id == e.entity_id)
                    .map(|edge| edge.weight)
                    .unwrap_or(0.0)
            })
            .collect();
        assert!(weights[0] >= weights[1]);
        assert!(weights[1] >= weights[2]);
    }

    // ------------------------------------------------------------------
    // traverse
    // ------------------------------------------------------------------

    #[test]
    fn test_traverse_single_hop() {
        let g = small_graph();
        let query = GraphQuery {
            start_entity_id: 0,
            relation_filter: None,
            max_hops: 1,
            entity_kind_filter: None,
        };
        let result = g.traverse(&query);
        let ids: Vec<u64> = result.iter().map(|e| e.entity_id).collect();
        // Direct neighbours of 0 are 1 and 3.
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
        // 2 is two hops away; must not appear.
        assert!(!ids.contains(&2));
    }

    #[test]
    fn test_traverse_multiple_hops() {
        let g = small_graph();
        let query = GraphQuery {
            start_entity_id: 0,
            relation_filter: None,
            max_hops: 2,
            entity_kind_filter: None,
        };
        let result = g.traverse(&query);
        let ids: Vec<u64> = result.iter().map(|e| e.entity_id).collect();
        // Two hops: 0 -> 1 -> 2
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
    }

    #[test]
    fn test_traverse_with_relation_filter() {
        let g = small_graph();
        let query = GraphQuery {
            start_entity_id: 0,
            relation_filter: Some("is_about".to_owned()),
            max_hops: 2,
            entity_kind_filter: None,
        };
        let result = g.traverse(&query);
        let ids: Vec<u64> = result.iter().map(|e| e.entity_id).collect();
        // Only "is_about" edges: 0 -> 1.
        // From 1, "authored_by" edge to 2 is filtered out.
        assert!(ids.contains(&1));
        assert!(!ids.contains(&2));
        assert!(!ids.contains(&3));
    }

    #[test]
    fn test_traverse_with_entity_kind_filter() {
        let g = small_graph();
        let query = GraphQuery {
            start_entity_id: 0,
            relation_filter: None,
            max_hops: 2,
            entity_kind_filter: Some(EntityKind::Person),
        };
        let result = g.traverse(&query);
        let ids: Vec<u64> = result.iter().map(|e| e.entity_id).collect();
        // Only Person entities: Alice (2).
        assert_eq!(ids, vec![2]);
    }

    #[test]
    fn test_traverse_excludes_start_entity() {
        let g = small_graph();
        let query = GraphQuery {
            start_entity_id: 0,
            relation_filter: None,
            max_hops: 5,
            entity_kind_filter: None,
        };
        let result = g.traverse(&query);
        assert!(!result.iter().any(|e| e.entity_id == 0));
    }

    #[test]
    fn test_traverse_sorted_by_entity_id() {
        let g = small_graph();
        let query = GraphQuery {
            start_entity_id: 0,
            relation_filter: None,
            max_hops: 5,
            entity_kind_filter: None,
        };
        let result = g.traverse(&query);
        let ids: Vec<u64> = result.iter().map(|e| e.entity_id).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted);
    }

    #[test]
    fn test_traverse_zero_hops_returns_empty() {
        let g = small_graph();
        let query = GraphQuery {
            start_entity_id: 0,
            relation_filter: None,
            max_hops: 0,
            entity_kind_filter: None,
        };
        let result = g.traverse(&query);
        assert!(result.is_empty());
    }

    // ------------------------------------------------------------------
    // similar_entities
    // ------------------------------------------------------------------

    #[test]
    fn test_similar_entities_above_threshold() {
        let g = small_graph();
        // Entity 0 embedding = [1, 0]; entity 1 embedding = [0.9, 0.1]
        // cosine_sim should be high.
        let sims = g.similar_entities(0, 0.9);
        assert!(!sims.is_empty());
        let ids: Vec<u64> = sims.iter().map(|(e, _)| e.entity_id).collect();
        assert!(ids.contains(&1));
    }

    #[test]
    fn test_similar_entities_excludes_self() {
        let g = small_graph();
        let sims = g.similar_entities(0, 0.0);
        assert!(sims.iter().all(|(e, _)| e.entity_id != 0));
    }

    #[test]
    fn test_similar_entities_sorted_by_sim_desc() {
        let g = small_graph();
        let sims = g.similar_entities(0, 0.0);
        if sims.len() > 1 {
            for i in 0..sims.len() - 1 {
                assert!(sims[i].1 >= sims[i + 1].1);
            }
        }
    }

    #[test]
    fn test_similar_entities_empty_when_no_embeddings() {
        let mut g = SemanticKnowledgeGraph::new();
        let a = g.add_entity("A", EntityKind::Concept, None);
        let _b = g.add_entity("B", EntityKind::Concept, None);
        // Neither entity has an embedding.
        let sims = g.similar_entities(a, 0.0);
        assert!(sims.is_empty());
    }

    #[test]
    fn test_similar_entities_returns_empty_for_missing_entity() {
        let g = small_graph();
        let sims = g.similar_entities(9999, 0.0);
        assert!(sims.is_empty());
    }

    // ------------------------------------------------------------------
    // remove_entity
    // ------------------------------------------------------------------

    #[test]
    fn test_remove_entity_removes_node() {
        let mut g = small_graph();
        let removed = g.remove_entity(0);
        assert!(removed);
        assert!(g.get_entity(0).is_none());
    }

    #[test]
    fn test_remove_entity_removes_connected_edges() {
        let mut g = small_graph();
        let edges_before = g.edges.len();
        g.remove_entity(0);
        // Entity 0 has two outgoing edges (to 1 and 3); both must be gone.
        assert!(g.edges.len() < edges_before);
        assert!(g.edges.iter().all(|e| e.from_id != 0 && e.to_id != 0));
    }

    #[test]
    fn test_remove_entity_false_for_unknown() {
        let mut g = small_graph();
        assert!(!g.remove_entity(999));
    }

    // ------------------------------------------------------------------
    // stats
    // ------------------------------------------------------------------

    #[test]
    fn test_stats_total_entities_and_edges() {
        let g = small_graph();
        let s = g.stats();
        // small_graph adds 4 entities and 3 edges.
        assert_eq!(s.total_entities, 4);
        assert_eq!(s.total_edges, 3);
    }

    #[test]
    fn test_stats_entities_with_embeddings() {
        let g = small_graph();
        let s = g.stats();
        // AI, ML paper, Alice have embeddings; OpenAI does not.
        assert_eq!(s.entities_with_embeddings, 3);
    }

    #[test]
    fn test_stats_avg_degree() {
        let g = small_graph();
        let s = g.stats();
        let expected = (3_f64 * 2.0) / 4.0;
        assert!((s.avg_degree - expected).abs() < 1e-9);
    }

    #[test]
    fn test_stats_avg_degree_empty_graph() {
        let g = SemanticKnowledgeGraph::new();
        let s = g.stats();
        assert_eq!(s.avg_degree, 0.0);
    }

    // ------------------------------------------------------------------
    // cosine_sim
    // ------------------------------------------------------------------

    #[test]
    fn test_cosine_sim_identical_vectors() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let sim = cosine_sim(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_sim_orthogonal_vectors() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let sim = cosine_sim(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_sim_zero_vector() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![1.0_f32, 0.0];
        let sim = cosine_sim(&a, &b);
        assert_eq!(sim, 0.0);
    }
}
