//! # Knowledge Base Builder
//!
//! Incrementally builds and maintains a semantic knowledge base from documents,
//! supporting semantic triples, entity relationships, and concept graphs.
//!
//! ## Features
//!
//! - Entity CRUD with alias indexing and optional embedding vectors
//! - Relation triples with FNV-1a-based IDs and confidence scores
//! - Concept co-occurrence graphs derived from document ingestion
//! - BFS path-finding between entities over the relation graph
//! - Rich statistics for monitoring and inspection

use std::collections::{HashMap, HashSet, VecDeque};

/// Error types for `KnowledgeBaseBuilder` operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum KbError {
    /// An entity with the given ID already exists.
    #[error("entity already exists: {0}")]
    EntityAlreadyExists(String),
    /// No entity found for the given ID.
    #[error("entity not found: {0}")]
    EntityNotFound(String),
    /// A document with the given ID already exists.
    #[error("document already exists: {0}")]
    DocumentAlreadyExists(String),
    /// A relation with the given ID already exists.
    #[error("relation already exists: {0}")]
    RelationAlreadyExists(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Core data types
// ─────────────────────────────────────────────────────────────────────────────

/// A named entity in the knowledge base.
#[derive(Debug, Clone)]
pub struct KbBuilderEntity {
    /// Unique string identifier.
    pub id: String,
    /// Human-readable primary name.
    pub name: String,
    /// Semantic type, e.g. "person", "organization", "concept".
    pub entity_type: String,
    /// Alternative surface forms for this entity.
    pub aliases: Vec<String>,
    /// Optional embedding vector.
    pub embedding: Option<Vec<f64>>,
    /// Unix epoch seconds at creation time.
    pub created_at: u64,
    /// Unix epoch seconds at last update time.
    pub updated_at: u64,
}

impl KbBuilderEntity {
    /// Create a minimal entity with the required fields.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        entity_type: impl Into<String>,
        now: u64,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            entity_type: entity_type.into(),
            aliases: Vec::new(),
            embedding: None,
            created_at: now,
            updated_at: now,
        }
    }
}

/// A directed semantic relation between two entities.
#[derive(Debug, Clone)]
pub struct KbRelation {
    /// FNV-1a hash of `subject_id + predicate + object_id`.
    pub id: String,
    /// Subject entity ID.
    pub subject_id: String,
    /// Predicate / relationship type.
    pub predicate: String,
    /// Object entity ID.
    pub object_id: String,
    /// Confidence in [0.0, 1.0].
    pub confidence: f64,
    /// Provenance source identifier.
    pub source: String,
    /// Unix epoch seconds at creation time.
    pub created_at: u64,
}

/// Convenience input type for a semantic triple.
#[derive(Debug, Clone)]
pub struct KbTriple {
    /// Subject entity ID.
    pub subject: String,
    /// Predicate string.
    pub predicate: String,
    /// Object entity ID.
    pub object: String,
}

impl KbTriple {
    /// Construct a triple.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }
}

/// A node in the concept co-occurrence graph.
#[derive(Debug, Clone)]
pub struct KbConceptNode {
    /// The concept string.
    pub concept: String,
    /// How many documents mention this concept.
    pub frequency: u32,
    /// Concepts that appear in documents alongside this one.
    pub related_concepts: Vec<String>,
    /// IDs of documents that mention this concept.
    pub documents: Vec<String>,
}

impl KbConceptNode {
    fn new(concept: impl Into<String>) -> Self {
        Self {
            concept: concept.into(),
            frequency: 0,
            related_concepts: Vec::new(),
            documents: Vec::new(),
        }
    }
}

/// A document ingested into the knowledge base.
#[derive(Debug, Clone)]
pub struct KbDocument {
    /// Unique document identifier.
    pub doc_id: String,
    /// Raw text content.
    pub content: String,
    /// IDs of entities mentioned in this document.
    pub entities_mentioned: Vec<String>,
    /// Concepts extracted from this document.
    pub concepts: Vec<String>,
    /// Unix epoch seconds when the document was added.
    pub added_at: u64,
}

impl KbDocument {
    /// Construct a document record.
    pub fn new(
        doc_id: impl Into<String>,
        content: impl Into<String>,
        entities_mentioned: Vec<String>,
        concepts: Vec<String>,
        added_at: u64,
    ) -> Self {
        Self {
            doc_id: doc_id.into(),
            content: content.into(),
            entities_mentioned,
            concepts,
            added_at,
        }
    }
}

/// Aggregate statistics for the knowledge base.
#[derive(Debug, Clone)]
pub struct KbStats {
    /// Total entity count.
    pub entity_count: usize,
    /// Total relation count.
    pub relation_count: usize,
    /// Total document count.
    pub document_count: usize,
    /// Total distinct concept count.
    pub concept_count: usize,
    /// Average number of relations per entity (incoming + outgoing).
    pub avg_relations_per_entity: f64,
    /// Average number of concepts per document.
    pub avg_concepts_per_doc: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// FNV-1a helper
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a 16-hex-char FNV-1a hash of the given string.
fn fnv1a_str(s: &str) -> String {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    format!("{:016x}", h)
}

// ─────────────────────────────────────────────────────────────────────────────
// KnowledgeBaseBuilder
// ─────────────────────────────────────────────────────────────────────────────

/// Incrementally builds and maintains a semantic knowledge base.
///
/// Supports entities, directed semantic relations, concept co-occurrence
/// graphs, and document ingestion.
#[derive(Debug, Default)]
pub struct KnowledgeBaseBuilder {
    /// All entities, keyed by entity ID.
    entities: HashMap<String, KbBuilderEntity>,
    /// All relations in insertion order.
    relations: Vec<KbRelation>,
    /// Concept co-occurrence graph, keyed by concept string.
    concept_graph: HashMap<String, KbConceptNode>,
    /// All ingested documents, keyed by doc_id.
    documents: HashMap<String, KbDocument>,
    /// Alias → canonical entity ID index (lower-cased alias).
    entity_alias_index: HashMap<String, String>,
}

impl KnowledgeBaseBuilder {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Create an empty `KnowledgeBaseBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    // ── Entity management ─────────────────────────────────────────────────────

    /// Add a new entity to the knowledge base.
    ///
    /// Returns [`KbError::EntityAlreadyExists`] if an entity with the same ID
    /// was previously registered.  Also registers all aliases in the alias index.
    pub fn add_entity(&mut self, entity: KbBuilderEntity) -> Result<(), KbError> {
        if self.entities.contains_key(&entity.id) {
            return Err(KbError::EntityAlreadyExists(entity.id.clone()));
        }

        // Index the primary name as an alias.
        let name_key = entity.name.to_lowercase();
        self.entity_alias_index
            .entry(name_key)
            .or_insert_with(|| entity.id.clone());

        // Index all explicit aliases.
        for alias in &entity.aliases {
            let alias_key = alias.to_lowercase();
            self.entity_alias_index
                .entry(alias_key)
                .or_insert_with(|| entity.id.clone());
        }

        self.entities.insert(entity.id.clone(), entity);
        Ok(())
    }

    /// Apply an in-place mutation to an existing entity.
    ///
    /// Returns [`KbError::EntityNotFound`] if the ID is not registered.
    /// After mutation, re-indexes all aliases (including the primary name).
    pub fn update_entity(
        &mut self,
        entity_id: &str,
        update_fn: impl FnOnce(&mut KbBuilderEntity),
    ) -> Result<(), KbError> {
        let entity = self
            .entities
            .get_mut(entity_id)
            .ok_or_else(|| KbError::EntityNotFound(entity_id.to_owned()))?;

        // Remove stale alias entries for this entity before mutation.
        let old_name = entity.name.to_lowercase();
        let old_aliases: Vec<String> = entity.aliases.iter().map(|a| a.to_lowercase()).collect();

        update_fn(entity);

        // Update updated_at is caller responsibility via the closure; mark here too.
        // Re-index aliases post mutation.
        let new_name = entity.name.to_lowercase();
        let new_aliases: Vec<String> = entity.aliases.iter().map(|a| a.to_lowercase()).collect();
        let entity_id_owned = entity.id.clone();

        // Remove old name/aliases.
        if self
            .entity_alias_index
            .get(&old_name)
            .map(|v| v == &entity_id_owned)
            .unwrap_or(false)
        {
            self.entity_alias_index.remove(&old_name);
        }
        for old_alias in &old_aliases {
            if self
                .entity_alias_index
                .get(old_alias)
                .map(|v| v == &entity_id_owned)
                .unwrap_or(false)
            {
                self.entity_alias_index.remove(old_alias);
            }
        }

        // Insert new name/aliases.
        self.entity_alias_index
            .entry(new_name)
            .or_insert(entity_id_owned.clone());
        for new_alias in new_aliases {
            self.entity_alias_index
                .entry(new_alias)
                .or_insert(entity_id_owned.clone());
        }

        Ok(())
    }

    /// Remove an entity and all relations that involve it.
    ///
    /// Returns `true` if the entity existed.
    pub fn remove_entity(&mut self, entity_id: &str) -> bool {
        let Some(entity) = self.entities.remove(entity_id) else {
            return false;
        };

        // Remove alias entries that point to this entity.
        let name_key = entity.name.to_lowercase();
        if self
            .entity_alias_index
            .get(&name_key)
            .map(|v| v == entity_id)
            .unwrap_or(false)
        {
            self.entity_alias_index.remove(&name_key);
        }
        for alias in &entity.aliases {
            let alias_key = alias.to_lowercase();
            if self
                .entity_alias_index
                .get(&alias_key)
                .map(|v| v == entity_id)
                .unwrap_or(false)
            {
                self.entity_alias_index.remove(&alias_key);
            }
        }

        // Remove relations involving this entity.
        self.relations
            .retain(|r| r.subject_id != entity_id && r.object_id != entity_id);

        true
    }

    // ── Relation management ───────────────────────────────────────────────────

    /// Add a semantic relation from a triple.
    ///
    /// Validates that both subject and object entities exist.  The relation ID
    /// is the FNV-1a hash of `subject_id + predicate + object_id`.
    ///
    /// Returns the relation ID on success.  Returns
    /// [`KbError::RelationAlreadyExists`] if the same triple was already added.
    pub fn add_relation(
        &mut self,
        triple: KbTriple,
        confidence: f64,
        source: String,
        now: u64,
    ) -> Result<String, KbError> {
        if !self.entities.contains_key(&triple.subject) {
            return Err(KbError::EntityNotFound(triple.subject.clone()));
        }
        if !self.entities.contains_key(&triple.object) {
            return Err(KbError::EntityNotFound(triple.object.clone()));
        }

        let relation_key = format!("{}{}{}", triple.subject, triple.predicate, triple.object);
        let relation_id = fnv1a_str(&relation_key);

        if self.relations.iter().any(|r| r.id == relation_id) {
            return Err(KbError::RelationAlreadyExists(relation_id));
        }

        self.relations.push(KbRelation {
            id: relation_id.clone(),
            subject_id: triple.subject,
            predicate: triple.predicate,
            object_id: triple.object,
            confidence,
            source,
            created_at: now,
        });

        Ok(relation_id)
    }

    /// Remove the relation with the given ID.
    ///
    /// Returns `true` if it existed.
    pub fn remove_relation(&mut self, relation_id: &str) -> bool {
        let before = self.relations.len();
        self.relations.retain(|r| r.id != relation_id);
        self.relations.len() < before
    }

    // ── Document management ───────────────────────────────────────────────────

    /// Ingest a document into the knowledge base.
    ///
    /// Updates the concept graph for each concept in the document:
    /// increments frequency, records the document ID, and adds
    /// co-occurrence edges to every other concept in the same document.
    ///
    /// Returns [`KbError::DocumentAlreadyExists`] if the doc_id is taken.
    pub fn add_document(&mut self, doc: KbDocument) -> Result<(), KbError> {
        if self.documents.contains_key(&doc.doc_id) {
            return Err(KbError::DocumentAlreadyExists(doc.doc_id.clone()));
        }

        let doc_id = doc.doc_id.clone();
        let concepts = doc.concepts.clone();

        // Ensure all concept nodes exist.
        for concept in &concepts {
            self.concept_graph
                .entry(concept.clone())
                .or_insert_with(|| KbConceptNode::new(concept.clone()));
        }

        // Update each concept node.
        for (i, concept) in concepts.iter().enumerate() {
            let node = self
                .concept_graph
                .get_mut(concept)
                .expect("node was just inserted");
            node.frequency += 1;
            node.documents.push(doc_id.clone());

            // Co-occurrence: add all other concepts in this document.
            for (j, other) in concepts.iter().enumerate() {
                if i != j && !node.related_concepts.contains(other) {
                    node.related_concepts.push(other.clone());
                }
            }
        }

        self.documents.insert(doc_id, doc);
        Ok(())
    }

    // ── Lookup helpers ────────────────────────────────────────────────────────

    /// Look up an entity by its primary name or any registered alias.
    ///
    /// The comparison is case-insensitive.
    pub fn find_entity_by_name(&self, name: &str) -> Option<&KbBuilderEntity> {
        let key = name.to_lowercase();
        // Check alias index first (covers primary name too).
        if let Some(entity_id) = self.entity_alias_index.get(&key) {
            return self.entities.get(entity_id);
        }
        // Fall back to linear scan over entity names.
        self.entities
            .values()
            .find(|e| e.name.to_lowercase() == key)
    }

    /// Look up an entity via the alias index (case-insensitive).
    pub fn find_entity_by_alias(&self, alias: &str) -> Option<&KbBuilderEntity> {
        let key = alias.to_lowercase();
        let entity_id = self.entity_alias_index.get(&key)?;
        self.entities.get(entity_id)
    }

    /// All relations where the entity is the subject **or** the object.
    pub fn relations_for_entity(&self, entity_id: &str) -> Vec<&KbRelation> {
        self.relations
            .iter()
            .filter(|r| r.subject_id == entity_id || r.object_id == entity_id)
            .collect()
    }

    /// Relations where `entity_id` is the subject (outgoing).
    pub fn outgoing_relations(&self, entity_id: &str) -> Vec<&KbRelation> {
        self.relations
            .iter()
            .filter(|r| r.subject_id == entity_id)
            .collect()
    }

    /// Relations where `entity_id` is the object (incoming).
    pub fn incoming_relations(&self, entity_id: &str) -> Vec<&KbRelation> {
        self.relations
            .iter()
            .filter(|r| r.object_id == entity_id)
            .collect()
    }

    /// Entities directly connected to `entity_id` via any relation (neighbours).
    pub fn entity_neighbors(&self, entity_id: &str) -> Vec<&KbBuilderEntity> {
        let mut seen: HashSet<&str> = HashSet::new();
        let mut result = Vec::new();

        for r in &self.relations {
            let neighbor_id: Option<&str> = if r.subject_id == entity_id {
                Some(&r.object_id)
            } else if r.object_id == entity_id {
                Some(&r.subject_id)
            } else {
                None
            };

            if let Some(nid) = neighbor_id {
                if seen.insert(nid) {
                    if let Some(e) = self.entities.get(nid) {
                        result.push(e);
                    }
                }
            }
        }

        result
    }

    /// Find the shortest path between two entities via BFS over relations.
    ///
    /// Returns a sequence of entity IDs (inclusive of `from_id` and `to_id`),
    /// or `None` if no path exists within `max_hops`.
    pub fn path_between(&self, from_id: &str, to_id: &str, max_hops: usize) -> Option<Vec<String>> {
        if from_id == to_id {
            return Some(vec![from_id.to_owned()]);
        }
        if max_hops == 0 {
            return None;
        }

        // BFS: (current_node, path_so_far)
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<Vec<String>> = VecDeque::new();

        visited.insert(from_id.to_owned());
        queue.push_back(vec![from_id.to_owned()]);

        while let Some(path) = queue.pop_front() {
            let current = path.last().expect("path is non-empty");
            let hops = path.len() - 1;

            if hops >= max_hops {
                continue;
            }

            for r in &self.relations {
                let neighbor: Option<&str> = if r.subject_id == *current {
                    Some(&r.object_id)
                } else if r.object_id == *current {
                    Some(&r.subject_id)
                } else {
                    None
                };

                if let Some(nid) = neighbor {
                    if nid == to_id {
                        let mut found = path.clone();
                        found.push(nid.to_owned());
                        return Some(found);
                    }
                    if !visited.contains(nid) {
                        visited.insert(nid.to_owned());
                        let mut next_path = path.clone();
                        next_path.push(nid.to_owned());
                        queue.push_back(next_path);
                    }
                }
            }
        }

        None
    }

    // ── Concept graph ─────────────────────────────────────────────────────────

    /// Return the top `n` concept nodes by frequency, descending.
    pub fn top_concepts(&self, n: usize) -> Vec<&KbConceptNode> {
        let mut nodes: Vec<&KbConceptNode> = self.concept_graph.values().collect();
        nodes.sort_by(|a, b| {
            b.frequency
                .cmp(&a.frequency)
                .then(a.concept.cmp(&b.concept))
        });
        nodes.truncate(n);
        nodes
    }

    /// Count how many documents mention both `concept_a` and `concept_b`.
    pub fn concept_cooccurrence(&self, concept_a: &str, concept_b: &str) -> usize {
        let docs_a: HashSet<&str> = self
            .concept_graph
            .get(concept_a)
            .map(|n| n.documents.iter().map(|d| d.as_str()).collect())
            .unwrap_or_default();

        let docs_b: HashSet<&str> = self
            .concept_graph
            .get(concept_b)
            .map(|n| n.documents.iter().map(|d| d.as_str()).collect())
            .unwrap_or_default();

        docs_a.intersection(&docs_b).count()
    }

    // ── Counts ────────────────────────────────────────────────────────────────

    /// Number of entities in the knowledge base.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Number of relations in the knowledge base.
    pub fn relation_count(&self) -> usize {
        self.relations.len()
    }

    /// Number of documents ingested.
    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    /// Compute aggregate statistics for the knowledge base.
    pub fn stats(&self) -> KbStats {
        let entity_count = self.entities.len();
        let relation_count = self.relations.len();
        let document_count = self.documents.len();
        let concept_count = self.concept_graph.len();

        let avg_relations_per_entity = if entity_count == 0 {
            0.0
        } else {
            // Count total relation endpoints (each relation touches two entities).
            (relation_count as f64 * 2.0) / entity_count as f64
        };

        let total_concepts: usize = self.documents.values().map(|d| d.concepts.len()).sum();
        let avg_concepts_per_doc = if document_count == 0 {
            0.0
        } else {
            total_concepts as f64 / document_count as f64
        };

        KbStats {
            entity_count,
            relation_count,
            document_count,
            concept_count,
            avg_relations_per_entity,
            avg_concepts_per_doc,
        }
    }

    // ── Read-only access to internal collections ──────────────────────────────

    /// Direct read access to the entity map.
    pub fn entities(&self) -> &HashMap<String, KbBuilderEntity> {
        &self.entities
    }

    /// Direct read access to the relation list.
    pub fn relations(&self) -> &[KbRelation] {
        &self.relations
    }

    /// Direct read access to the concept graph.
    pub fn concept_graph(&self) -> &HashMap<String, KbConceptNode> {
        &self.concept_graph
    }

    /// Direct read access to the document map.
    pub fn documents(&self) -> &HashMap<String, KbDocument> {
        &self.documents
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::knowledge_base_builder::{
        fnv1a_str, KbBuilderEntity, KbDocument, KbError, KbTriple, KnowledgeBaseBuilder,
    };

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_entity(id: &str, name: &str, now: u64) -> KbBuilderEntity {
        KbBuilderEntity::new(id, name, "concept", now)
    }

    fn make_entity_with_aliases(
        id: &str,
        name: &str,
        aliases: Vec<&str>,
        now: u64,
    ) -> KbBuilderEntity {
        let mut e = make_entity(id, name, now);
        e.aliases = aliases.iter().map(|s| s.to_string()).collect();
        e
    }

    fn make_doc(doc_id: &str, concepts: Vec<&str>, entities: Vec<&str>) -> KbDocument {
        KbDocument::new(
            doc_id,
            "content",
            entities.iter().map(|s| s.to_string()).collect(),
            concepts.iter().map(|s| s.to_string()).collect(),
            1_000_000,
        )
    }

    // ── FNV-1a ────────────────────────────────────────────────────────────────

    #[test]
    fn test_fnv1a_empty_string() {
        let h = fnv1a_str("");
        assert_eq!(h.len(), 16);
    }

    #[test]
    fn test_fnv1a_deterministic() {
        let h1 = fnv1a_str("hello");
        let h2 = fnv1a_str("hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fnv1a_distinct() {
        let h1 = fnv1a_str("abcpredxyz");
        let h2 = fnv1a_str("xyzpredabc");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_fnv1a_hex_format() {
        let h = fnv1a_str("test");
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(h.len(), 16);
    }

    // ── KnowledgeBaseBuilder::new ─────────────────────────────────────────────

    #[test]
    fn test_new_is_empty() {
        let kb = KnowledgeBaseBuilder::new();
        assert_eq!(kb.entity_count(), 0);
        assert_eq!(kb.relation_count(), 0);
        assert_eq!(kb.document_count(), 0);
    }

    // ── add_entity ────────────────────────────────────────────────────────────

    #[test]
    fn test_add_entity_success() {
        let mut kb = KnowledgeBaseBuilder::new();
        let e = make_entity("e1", "Alpha", 0);
        assert!(kb.add_entity(e).is_ok());
        assert_eq!(kb.entity_count(), 1);
    }

    #[test]
    fn test_add_entity_duplicate_returns_error() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_entity(make_entity("e1", "Alpha", 0))
            .expect("test: add entity e1 for duplicate check");
        let err = kb.add_entity(make_entity("e1", "Alpha2", 0)).unwrap_err();
        assert_eq!(err, KbError::EntityAlreadyExists("e1".to_owned()));
    }

    #[test]
    fn test_add_entity_alias_indexed() {
        let mut kb = KnowledgeBaseBuilder::new();
        let e = make_entity_with_aliases("e1", "Alpha", vec!["A", "al"], 0);
        kb.add_entity(e).expect("test: add entity with aliases");
        assert!(kb.find_entity_by_alias("a").is_some());
        assert!(kb.find_entity_by_alias("al").is_some());
        assert!(kb.find_entity_by_alias("ALPHA").is_some());
    }

    // ── update_entity ─────────────────────────────────────────────────────────

    #[test]
    fn test_update_entity_success() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_entity(make_entity("e1", "Alpha", 0))
            .expect("test: add entity e1 before update");
        kb.update_entity("e1", |e| {
            e.name = "Beta".to_owned();
        })
        .expect("test: update entity e1 name to Beta");
        let e = kb
            .find_entity_by_name("Beta")
            .expect("test: find entity by name Beta after update");
        assert_eq!(e.id, "e1");
    }

    #[test]
    fn test_update_entity_not_found() {
        let mut kb = KnowledgeBaseBuilder::new();
        let err = kb
            .update_entity("nonexistent", |e| e.name = "x".to_owned())
            .unwrap_err();
        assert_eq!(err, KbError::EntityNotFound("nonexistent".to_owned()));
    }

    #[test]
    fn test_update_entity_alias_reindexed() {
        let mut kb = KnowledgeBaseBuilder::new();
        let mut e = make_entity("e1", "Alpha", 0);
        e.aliases = vec!["OldAlias".to_owned()];
        kb.add_entity(e).expect("test: add entity with OldAlias");

        kb.update_entity("e1", |ent| {
            ent.aliases = vec!["NewAlias".to_owned()];
        })
        .expect("test: update entity e1 aliases to NewAlias");

        assert!(kb.find_entity_by_alias("newalias").is_some());
    }

    // ── remove_entity ─────────────────────────────────────────────────────────

    #[test]
    fn test_remove_entity_existing() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_entity(make_entity("e1", "Alpha", 0))
            .expect("test: add entity e1 before remove");
        assert!(kb.remove_entity("e1"));
        assert_eq!(kb.entity_count(), 0);
    }

    #[test]
    fn test_remove_entity_nonexistent() {
        let mut kb = KnowledgeBaseBuilder::new();
        assert!(!kb.remove_entity("ghost"));
    }

    #[test]
    fn test_remove_entity_cascades_relations() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_entity(make_entity("e1", "A", 0))
            .expect("test: add entity e1 for cascade test");
        kb.add_entity(make_entity("e2", "B", 0))
            .expect("test: add entity e2 for cascade test");
        kb.add_relation(KbTriple::new("e1", "knows", "e2"), 1.0, "test".into(), 0)
            .expect("test: add relation e1 knows e2 for cascade test");
        assert_eq!(kb.relation_count(), 1);

        kb.remove_entity("e1");
        assert_eq!(kb.relation_count(), 0);
    }

    #[test]
    fn test_remove_entity_cleans_alias_index() {
        let mut kb = KnowledgeBaseBuilder::new();
        let mut e = make_entity("e1", "Alpha", 0);
        e.aliases = vec!["AL".to_owned()];
        kb.add_entity(e).expect("test: add entity with AL alias");
        kb.remove_entity("e1");
        assert!(kb.find_entity_by_alias("al").is_none());
        assert!(kb.find_entity_by_name("alpha").is_none());
    }

    // ── add_relation ──────────────────────────────────────────────────────────

    #[test]
    fn test_add_relation_success() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_entity(make_entity("e1", "A", 0))
            .expect("test: add entity e1 for relation test");
        kb.add_entity(make_entity("e2", "B", 0))
            .expect("test: add entity e2 for relation test");
        let id = kb
            .add_relation(KbTriple::new("e1", "knows", "e2"), 0.9, "src".into(), 0)
            .expect("test: add relation e1 knows e2");
        assert_eq!(id.len(), 16);
        assert_eq!(kb.relation_count(), 1);
    }

    #[test]
    fn test_add_relation_subject_missing() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_entity(make_entity("e2", "B", 0))
            .expect("test: add entity e2 for subject-missing test");
        let err = kb
            .add_relation(KbTriple::new("e1", "knows", "e2"), 1.0, "src".into(), 0)
            .unwrap_err();
        assert_eq!(err, KbError::EntityNotFound("e1".to_owned()));
    }

    #[test]
    fn test_add_relation_object_missing() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_entity(make_entity("e1", "A", 0))
            .expect("test: add entity e1 for object-missing test");
        let err = kb
            .add_relation(KbTriple::new("e1", "knows", "e2"), 1.0, "src".into(), 0)
            .unwrap_err();
        assert_eq!(err, KbError::EntityNotFound("e2".to_owned()));
    }

    #[test]
    fn test_add_relation_duplicate_error() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_entity(make_entity("e1", "A", 0))
            .expect("test: add entity e1 for duplicate relation test");
        kb.add_entity(make_entity("e2", "B", 0))
            .expect("test: add entity e2 for duplicate relation test");
        kb.add_relation(KbTriple::new("e1", "knows", "e2"), 1.0, "src".into(), 0)
            .expect("test: add first relation e1 knows e2");
        let err = kb
            .add_relation(KbTriple::new("e1", "knows", "e2"), 0.5, "src2".into(), 1)
            .unwrap_err();
        matches!(err, KbError::RelationAlreadyExists(_));
    }

    #[test]
    fn test_add_relation_different_predicates_allowed() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_entity(make_entity("e1", "A", 0))
            .expect("test: add entity e1 for multi-predicate test");
        kb.add_entity(make_entity("e2", "B", 0))
            .expect("test: add entity e2 for multi-predicate test");
        kb.add_relation(KbTriple::new("e1", "knows", "e2"), 1.0, "src".into(), 0)
            .expect("test: add relation e1 knows e2");
        kb.add_relation(KbTriple::new("e1", "hates", "e2"), 0.3, "src".into(), 0)
            .expect("test: add relation e1 hates e2");
        assert_eq!(kb.relation_count(), 2);
    }

    // ── remove_relation ───────────────────────────────────────────────────────

    #[test]
    fn test_remove_relation_existing() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_entity(make_entity("e1", "A", 0))
            .expect("test: add entity e1 for remove relation test");
        kb.add_entity(make_entity("e2", "B", 0))
            .expect("test: add entity e2 for remove relation test");
        let rid = kb
            .add_relation(KbTriple::new("e1", "knows", "e2"), 1.0, "src".into(), 0)
            .expect("test: add relation for removal");
        assert!(kb.remove_relation(&rid));
        assert_eq!(kb.relation_count(), 0);
    }

    #[test]
    fn test_remove_relation_nonexistent() {
        let mut kb = KnowledgeBaseBuilder::new();
        assert!(!kb.remove_relation("deadbeefdeadbeef"));
    }

    // ── add_document ──────────────────────────────────────────────────────────

    #[test]
    fn test_add_document_success() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_document(make_doc("d1", vec!["rust", "memory"], vec![]))
            .expect("test: add document d1 with rust and memory");
        assert_eq!(kb.document_count(), 1);
    }

    #[test]
    fn test_add_document_duplicate_error() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_document(make_doc("d1", vec!["rust"], vec![]))
            .expect("test: add document d1 first time");
        let err = kb
            .add_document(make_doc("d1", vec!["rust"], vec![]))
            .unwrap_err();
        assert_eq!(err, KbError::DocumentAlreadyExists("d1".to_owned()));
    }

    #[test]
    fn test_add_document_updates_concept_graph() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_document(make_doc("d1", vec!["rust", "memory"], vec![]))
            .expect("test: add document d1 with rust and memory concepts");
        let node = kb
            .concept_graph()
            .get("rust")
            .expect("test: concept rust should exist in graph");
        assert_eq!(node.frequency, 1);
        assert!(node.related_concepts.contains(&"memory".to_owned()));
    }

    #[test]
    fn test_add_document_accumulates_frequency() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_document(make_doc("d1", vec!["rust"], vec![]))
            .expect("test: add document d1 with rust");
        kb.add_document(make_doc("d2", vec!["rust"], vec![]))
            .expect("test: add document d2 with rust");
        let node = kb
            .concept_graph()
            .get("rust")
            .expect("test: concept rust should exist in graph after two docs");
        assert_eq!(node.frequency, 2);
    }

    // ── find_entity_by_name / find_entity_by_alias ───────────────────────────

    #[test]
    fn test_find_entity_by_name_exact() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_entity(make_entity("e1", "RustLang", 0))
            .expect("test: add entity RustLang for name lookup");
        let e = kb
            .find_entity_by_name("RustLang")
            .expect("test: find entity by exact name RustLang");
        assert_eq!(e.id, "e1");
    }

    #[test]
    fn test_find_entity_by_name_case_insensitive() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_entity(make_entity("e1", "RustLang", 0))
            .expect("test: add entity RustLang for case-insensitive lookup");
        assert!(kb.find_entity_by_name("rustlang").is_some());
        assert!(kb.find_entity_by_name("RUSTLANG").is_some());
    }

    #[test]
    fn test_find_entity_by_name_not_found() {
        let kb = KnowledgeBaseBuilder::new();
        assert!(kb.find_entity_by_name("nobody").is_none());
    }

    #[test]
    fn test_find_entity_by_alias() {
        let mut kb = KnowledgeBaseBuilder::new();
        let e = make_entity_with_aliases("e1", "RustLang", vec!["rs", "rust"], 0);
        kb.add_entity(e)
            .expect("test: add entity with rs and rust aliases");
        assert!(kb.find_entity_by_alias("RS").is_some());
        assert!(kb.find_entity_by_alias("rust").is_some());
    }

    // ── relation queries ──────────────────────────────────────────────────────

    #[test]
    fn test_relations_for_entity() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_entity(make_entity("e1", "A", 0))
            .expect("test: add entity e1 for relations query");
        kb.add_entity(make_entity("e2", "B", 0))
            .expect("test: add entity e2 for relations query");
        kb.add_entity(make_entity("e3", "C", 0))
            .expect("test: add entity e3 for relations query");
        kb.add_relation(KbTriple::new("e1", "knows", "e2"), 1.0, "s".into(), 0)
            .expect("test: add relation e1 knows e2");
        kb.add_relation(KbTriple::new("e3", "mentions", "e1"), 0.8, "s".into(), 0)
            .expect("test: add relation e3 mentions e1");
        kb.add_relation(KbTriple::new("e2", "linked", "e3"), 0.5, "s".into(), 0)
            .expect("test: add relation e2 linked e3");

        let rels = kb.relations_for_entity("e1");
        assert_eq!(rels.len(), 2);
    }

    #[test]
    fn test_outgoing_relations() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_entity(make_entity("e1", "A", 0))
            .expect("test: add entity e1 for outgoing relations");
        kb.add_entity(make_entity("e2", "B", 0))
            .expect("test: add entity e2 for outgoing relations");
        kb.add_entity(make_entity("e3", "C", 0))
            .expect("test: add entity e3 for outgoing relations");
        kb.add_relation(KbTriple::new("e1", "knows", "e2"), 1.0, "s".into(), 0)
            .expect("test: add relation e1 knows e2 for outgoing");
        kb.add_relation(KbTriple::new("e3", "knows", "e1"), 0.5, "s".into(), 0)
            .expect("test: add relation e3 knows e1 for outgoing");

        let out = kb.outgoing_relations("e1");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].object_id, "e2");
    }

    #[test]
    fn test_incoming_relations() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_entity(make_entity("e1", "A", 0))
            .expect("test: add entity e1 for incoming relations");
        kb.add_entity(make_entity("e2", "B", 0))
            .expect("test: add entity e2 for incoming relations");
        kb.add_entity(make_entity("e3", "C", 0))
            .expect("test: add entity e3 for incoming relations");
        kb.add_relation(KbTriple::new("e1", "knows", "e2"), 1.0, "s".into(), 0)
            .expect("test: add relation e1 knows e2 for incoming");
        kb.add_relation(KbTriple::new("e3", "knows", "e2"), 0.5, "s".into(), 0)
            .expect("test: add relation e3 knows e2 for incoming");

        let inc = kb.incoming_relations("e2");
        assert_eq!(inc.len(), 2);
    }

    // ── entity_neighbors ─────────────────────────────────────────────────────

    #[test]
    fn test_entity_neighbors() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_entity(make_entity("e1", "A", 0))
            .expect("test: add entity e1 for neighbors");
        kb.add_entity(make_entity("e2", "B", 0))
            .expect("test: add entity e2 for neighbors");
        kb.add_entity(make_entity("e3", "C", 0))
            .expect("test: add entity e3 for neighbors");
        kb.add_relation(KbTriple::new("e1", "knows", "e2"), 1.0, "s".into(), 0)
            .expect("test: add relation e1 knows e2 for neighbors");
        kb.add_relation(KbTriple::new("e3", "knows", "e1"), 0.8, "s".into(), 0)
            .expect("test: add relation e3 knows e1 for neighbors");

        let neighbors = kb.entity_neighbors("e1");
        let ids: Vec<&str> = neighbors.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"e2"));
        assert!(ids.contains(&"e3"));
    }

    #[test]
    fn test_entity_neighbors_no_duplicates() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_entity(make_entity("e1", "A", 0))
            .expect("test: add entity e1 for no-duplicate neighbors");
        kb.add_entity(make_entity("e2", "B", 0))
            .expect("test: add entity e2 for no-duplicate neighbors");
        // Two different predicates between the same pair.
        kb.add_relation(KbTriple::new("e1", "knows", "e2"), 1.0, "s".into(), 0)
            .expect("test: add relation e1 knows e2 for no-duplicate");
        kb.add_relation(KbTriple::new("e1", "likes", "e2"), 0.5, "s".into(), 0)
            .expect("test: add relation e1 likes e2 for no-duplicate");

        let neighbors = kb.entity_neighbors("e1");
        assert_eq!(neighbors.len(), 1, "neighbor e2 should appear once");
    }

    // ── path_between ──────────────────────────────────────────────────────────

    #[test]
    fn test_path_between_same_node() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_entity(make_entity("e1", "A", 0))
            .expect("test: add entity e1 for same-node path");
        let path = kb
            .path_between("e1", "e1", 5)
            .expect("test: path from e1 to e1 should succeed");
        assert_eq!(path, vec!["e1"]);
    }

    #[test]
    fn test_path_between_direct() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_entity(make_entity("e1", "A", 0))
            .expect("test: add entity e1 for direct path");
        kb.add_entity(make_entity("e2", "B", 0))
            .expect("test: add entity e2 for direct path");
        kb.add_relation(KbTriple::new("e1", "knows", "e2"), 1.0, "s".into(), 0)
            .expect("test: add relation e1 knows e2 for direct path");

        let path = kb
            .path_between("e1", "e2", 5)
            .expect("test: path between e1 and e2 should succeed");
        assert_eq!(path, vec!["e1", "e2"]);
    }

    #[test]
    fn test_path_between_two_hops() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_entity(make_entity("e1", "A", 0))
            .expect("test: add entity e1 for two-hop path");
        kb.add_entity(make_entity("e2", "B", 0))
            .expect("test: add entity e2 for two-hop path");
        kb.add_entity(make_entity("e3", "C", 0))
            .expect("test: add entity e3 for two-hop path");
        kb.add_relation(KbTriple::new("e1", "knows", "e2"), 1.0, "s".into(), 0)
            .expect("test: add relation e1 knows e2 for two-hop");
        kb.add_relation(KbTriple::new("e2", "knows", "e3"), 1.0, "s".into(), 0)
            .expect("test: add relation e2 knows e3 for two-hop");

        let path = kb
            .path_between("e1", "e3", 5)
            .expect("test: two-hop path from e1 to e3 should succeed");
        assert_eq!(path.first().map(|s| s.as_str()), Some("e1"));
        assert_eq!(path.last().map(|s| s.as_str()), Some("e3"));
        assert_eq!(path.len(), 3);
    }

    #[test]
    fn test_path_between_no_path() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_entity(make_entity("e1", "A", 0))
            .expect("test: add entity e1 for no-path test");
        kb.add_entity(make_entity("e2", "B", 0))
            .expect("test: add entity e2 for no-path test");
        // No relation between them.
        assert!(kb.path_between("e1", "e2", 5).is_none());
    }

    #[test]
    fn test_path_between_exceeds_max_hops() {
        let mut kb = KnowledgeBaseBuilder::new();
        for i in 0..5u8 {
            kb.add_entity(make_entity(&format!("e{}", i), &format!("E{}", i), 0))
                .expect("test: add entity in path chain");
        }
        for i in 0..4u8 {
            let s = format!("e{}", i);
            let o = format!("e{}", i + 1);
            kb.add_relation(KbTriple::new(s, "linked", o), 1.0, "s".into(), 0)
                .expect("test: add relation in path chain");
        }
        // Path e0 → e4 requires 4 hops; limit to 2.
        assert!(kb.path_between("e0", "e4", 2).is_none());
        // Limit to 5 should work.
        assert!(kb.path_between("e0", "e4", 5).is_some());
    }

    // ── top_concepts ──────────────────────────────────────────────────────────

    #[test]
    fn test_top_concepts_ordering() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_document(make_doc("d1", vec!["rust"], vec![]))
            .expect("test: add document d1 with rust");
        kb.add_document(make_doc("d2", vec!["rust", "memory"], vec![]))
            .expect("test: add document d2 with rust and memory");
        kb.add_document(make_doc("d3", vec!["memory", "safety"], vec![]))
            .expect("test: add document d3 with memory and safety");

        let top = kb.top_concepts(2);
        assert_eq!(top.len(), 2);
        // "rust" and "memory" both have frequency 2; "safety" has 1.
        let freqs: Vec<u32> = top.iter().map(|n| n.frequency).collect();
        assert!(freqs[0] >= freqs[1]);
    }

    #[test]
    fn test_top_concepts_fewer_than_n() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_document(make_doc("d1", vec!["rust"], vec![]))
            .expect("test: add document d1 for concept count check");
        let top = kb.top_concepts(10);
        assert_eq!(top.len(), 1);
    }

    // ── concept_cooccurrence ─────────────────────────────────────────────────

    #[test]
    fn test_concept_cooccurrence_positive() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_document(make_doc("d1", vec!["rust", "memory"], vec![]))
            .expect("test: add document d1 with rust and memory");
        kb.add_document(make_doc("d2", vec!["rust", "safety"], vec![]))
            .expect("test: add document d2 with rust and safety");
        kb.add_document(make_doc("d3", vec!["memory", "safety"], vec![]))
            .expect("test: add document d3 with memory and safety");

        assert_eq!(kb.concept_cooccurrence("rust", "memory"), 1);
        assert_eq!(kb.concept_cooccurrence("rust", "safety"), 1);
        assert_eq!(kb.concept_cooccurrence("memory", "safety"), 1);
    }

    #[test]
    fn test_concept_cooccurrence_zero() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_document(make_doc("d1", vec!["rust"], vec![]))
            .expect("test: add document d1 with rust");
        kb.add_document(make_doc("d2", vec!["python"], vec![]))
            .expect("test: add document d2 with python");
        assert_eq!(kb.concept_cooccurrence("rust", "python"), 0);
    }

    #[test]
    fn test_concept_cooccurrence_unknown_concept() {
        let kb = KnowledgeBaseBuilder::new();
        assert_eq!(kb.concept_cooccurrence("unknown_a", "unknown_b"), 0);
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_empty() {
        let kb = KnowledgeBaseBuilder::new();
        let s = kb.stats();
        assert_eq!(s.entity_count, 0);
        assert_eq!(s.relation_count, 0);
        assert_eq!(s.document_count, 0);
        assert_eq!(s.concept_count, 0);
        assert!((s.avg_relations_per_entity - 0.0).abs() < 1e-9);
        assert!((s.avg_concepts_per_doc - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_with_data() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_entity(make_entity("e1", "A", 0))
            .expect("test: add entity e1 for stats");
        kb.add_entity(make_entity("e2", "B", 0))
            .expect("test: add entity e2 for stats");
        kb.add_relation(KbTriple::new("e1", "knows", "e2"), 1.0, "s".into(), 0)
            .expect("test: add relation e1 knows e2 for stats");
        kb.add_document(make_doc("d1", vec!["c1", "c2", "c3"], vec![]))
            .expect("test: add document d1 for stats");

        let s = kb.stats();
        assert_eq!(s.entity_count, 2);
        assert_eq!(s.relation_count, 1);
        assert_eq!(s.document_count, 1);
        assert_eq!(s.concept_count, 3);
        // 1 relation * 2 endpoints / 2 entities = 1.0
        assert!((s.avg_relations_per_entity - 1.0).abs() < 1e-9);
        // 3 concepts / 1 document = 3.0
        assert!((s.avg_concepts_per_doc - 3.0).abs() < 1e-9);
    }

    // ── KbBuilderEntity::new ──────────────────────────────────────────────────

    #[test]
    fn test_kb_builder_entity_new() {
        let e = KbBuilderEntity::new("id1", "Name", "person", 42);
        assert_eq!(e.id, "id1");
        assert_eq!(e.name, "Name");
        assert_eq!(e.entity_type, "person");
        assert_eq!(e.created_at, 42);
        assert_eq!(e.updated_at, 42);
        assert!(e.aliases.is_empty());
        assert!(e.embedding.is_none());
    }

    // ── KbTriple::new ─────────────────────────────────────────────────────────

    #[test]
    fn test_kb_triple_new() {
        let t = KbTriple::new("s", "p", "o");
        assert_eq!(t.subject, "s");
        assert_eq!(t.predicate, "p");
        assert_eq!(t.object, "o");
    }

    // ── KbDocument::new ───────────────────────────────────────────────────────

    #[test]
    fn test_kb_document_new() {
        let d = KbDocument::new("d1", "hello", vec!["e1".into()], vec!["c1".into()], 100);
        assert_eq!(d.doc_id, "d1");
        assert_eq!(d.content, "hello");
        assert_eq!(d.entities_mentioned, vec!["e1"]);
        assert_eq!(d.concepts, vec!["c1"]);
        assert_eq!(d.added_at, 100);
    }

    // ── integration ───────────────────────────────────────────────────────────

    #[test]
    fn test_integration_full_graph() {
        let mut kb = KnowledgeBaseBuilder::new();

        // Build a small knowledge base.
        for (id, name) in [("alice", "Alice"), ("bob", "Bob"), ("carol", "Carol")] {
            kb.add_entity(KbBuilderEntity::new(id, name, "person", 0))
                .expect("test: add person entity for integration test");
        }
        kb.add_relation(KbTriple::new("alice", "knows", "bob"), 0.9, "src".into(), 0)
            .expect("test: add relation alice knows bob");
        kb.add_relation(KbTriple::new("bob", "knows", "carol"), 0.8, "src".into(), 0)
            .expect("test: add relation bob knows carol");

        let path = kb
            .path_between("alice", "carol", 3)
            .expect("test: path from alice to carol through bob");
        assert_eq!(path, vec!["alice", "bob", "carol"]);

        let s = kb.stats();
        assert_eq!(s.entity_count, 3);
        assert_eq!(s.relation_count, 2);
    }

    #[test]
    fn test_integration_document_entity_linking() {
        let mut kb = KnowledgeBaseBuilder::new();
        kb.add_entity(make_entity("e1", "Rust", 0))
            .expect("test: add entity Rust for document linking");
        let doc = KbDocument::new(
            "d1",
            "Rust is a systems language",
            vec!["e1".to_owned()],
            vec!["systems".to_owned(), "language".to_owned()],
            0,
        );
        kb.add_document(doc)
            .expect("test: add document d1 linking to Rust entity");

        let node = kb
            .concept_graph()
            .get("systems")
            .expect("test: concept systems should exist in graph");
        assert!(node.documents.contains(&"d1".to_owned()));
        assert!(node.related_concepts.contains(&"language".to_owned()));
    }
}
