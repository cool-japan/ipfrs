//! Logic solver for reasoning queries with semantic integration
//!
//! This module provides integration between semantic search and logic reasoning:
//! - Predicate-to-embedding mapping for similarity-based matching
//! - Logic term similarity for fuzzy unification
//! - Proof tree search using vector indices
//! - Backward chaining with semantic relevance
//! - Subgoal decomposition and dependency tracking

use crate::hnsw::{DistanceMetric, VectorIndex};
use ipfrs_core::{Cid, Error, Result};
use ipfrs_tensorlogic::{
    CycleDetector, GoalDecomposition, InferenceEngine, KnowledgeBase, Predicate, ProofRule,
    Substitution, Term,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

/// Configuration for the logic solver
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// Maximum recursion depth for backward chaining
    pub max_depth: usize,
    /// Similarity threshold for fuzzy matching (0.0 to 1.0)
    pub similarity_threshold: f32,
    /// Number of similar predicates to consider
    pub top_k_similar: usize,
    /// Embedding dimension for predicates
    pub embedding_dim: usize,
    /// Whether to use cycle detection
    pub detect_cycles: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_depth: 100,
            similarity_threshold: 0.8,
            top_k_similar: 10,
            embedding_dim: 384, // Standard embedding size
            detect_cycles: true,
        }
    }
}

/// Maps logic predicates to vector embeddings for similarity search
pub struct PredicateEmbedder {
    /// Embedding dimension
    dim: usize,
    /// Cache of predicate embeddings
    embeddings: Arc<RwLock<HashMap<String, Vec<f32>>>>,
}

impl PredicateEmbedder {
    /// Create a new predicate embedder
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            embeddings: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Generate embedding for a predicate
    ///
    /// This uses a simple compositional approach:
    /// - Predicate name contributes to the embedding
    /// - Each term contributes based on its structure
    /// - Ground terms have higher weight than variables
    pub fn embed_predicate(&self, pred: &Predicate) -> Vec<f32> {
        let cached = self.embeddings.read().get(&pred.to_string()).cloned();
        if let Some(emb) = cached {
            return emb;
        }

        let mut embedding = vec![0.0; self.dim];

        // Hash predicate name to embedding space
        let name_hash = self.hash_string(&pred.name);
        for (i, val) in embedding.iter_mut().enumerate() {
            *val += (((name_hash + i) as f32).sin() * 0.5).abs();
        }

        // Add contribution from each argument
        for (idx, term) in pred.args.iter().enumerate() {
            let term_emb = self.embed_term(term, idx);
            for i in 0..self.dim {
                embedding[i] += term_emb[i] * 0.3; // Weight terms less than predicate name
            }
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        // Cache the embedding
        self.embeddings
            .write()
            .insert(pred.to_string(), embedding.clone());

        embedding
    }

    /// Embed a logic term
    fn embed_term(&self, term: &Term, position: usize) -> Vec<f32> {
        let mut embedding = vec![0.0; self.dim];

        match term {
            Term::Var(name) => {
                // Variables get a low-weight embedding based on position
                let hash = self.hash_string(name) + position;
                for (i, val) in embedding.iter_mut().enumerate() {
                    *val = (((hash + i) as f32).sin() * 0.2).abs();
                }
            }
            Term::Const(constant) => {
                // Constants get higher weight
                let hash = self.hash_string(&format!("{:?}", constant));
                for (i, val) in embedding.iter_mut().enumerate() {
                    *val = (((hash + i) as f32).sin() * 0.8).abs();
                }
            }
            Term::Fun(functor, args) => {
                // Function terms combine functor and arg embeddings
                let hash = self.hash_string(functor);
                for (i, val) in embedding.iter_mut().enumerate() {
                    *val = (((hash + i) as f32).sin() * 0.6).abs();
                }

                for (idx, arg) in args.iter().enumerate() {
                    let arg_emb = self.embed_term(arg, idx);
                    for i in 0..self.dim {
                        embedding[i] += arg_emb[i] * 0.2;
                    }
                }
            }
            Term::Ref(_) => {
                // References get medium weight based on position
                let hash = position;
                for (i, val) in embedding.iter_mut().enumerate() {
                    *val = (((hash + i) as f32).sin() * 0.5).abs();
                }
            }
        }

        embedding
    }

    /// Simple string hash function
    fn hash_string(&self, s: &str) -> usize {
        s.bytes().fold(0usize, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(b as usize)
        })
    }

    /// Compute similarity between two predicates (cosine similarity)
    pub fn similarity(&self, pred1: &Predicate, pred2: &Predicate) -> f32 {
        let emb1 = self.embed_predicate(pred1);
        let emb2 = self.embed_predicate(pred2);

        self.cosine_similarity(&emb1, &emb2)
    }

    /// Cosine similarity between two vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a < 1e-6 || norm_b < 1e-6 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

/// Proof tree node for search
#[derive(Debug, Clone)]
pub struct ProofTreeNode {
    /// Current goal predicate
    pub goal: Predicate,
    /// Substitutions applied so far
    pub substitution: Substitution,
    /// Parent node ID (None for root)
    pub parent: Option<usize>,
    /// Depth in the proof tree
    pub depth: usize,
    /// Relevance score (from semantic search)
    pub relevance: f32,
}

/// Logic solver with semantic integration
pub struct LogicSolver {
    /// Configuration
    config: SolverConfig,
    /// Knowledge base for reasoning
    kb: Arc<RwLock<KnowledgeBase>>,
    /// Inference engine
    engine: Arc<RwLock<InferenceEngine>>,
    /// Predicate embedder for similarity search
    embedder: PredicateEmbedder,
    /// Vector index for predicate search
    predicate_index: Arc<RwLock<Option<VectorIndex>>>,
    /// Cycle detector
    cycle_detector: Arc<RwLock<CycleDetector>>,
    /// Cache of CID to predicate mapping
    cid_to_predicate: Arc<RwLock<HashMap<Cid, Predicate>>>,
}

impl LogicSolver {
    /// Create a new logic solver
    pub fn new(config: SolverConfig) -> Result<Self> {
        let kb = KnowledgeBase::new();
        let engine = InferenceEngine::new();

        Ok(Self {
            embedder: PredicateEmbedder::new(config.embedding_dim),
            config,
            kb: Arc::new(RwLock::new(kb)),
            engine: Arc::new(RwLock::new(engine)),
            predicate_index: Arc::new(RwLock::new(None)),
            cycle_detector: Arc::new(RwLock::new(CycleDetector::new())),
            cid_to_predicate: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Create with default configuration
    pub fn with_defaults() -> Result<Self> {
        Self::new(SolverConfig::default())
    }

    /// Add a fact to the knowledge base and index it
    pub fn add_fact(&mut self, fact: Predicate, cid: Cid) -> Result<()> {
        // Add to knowledge base
        self.kb.write().add_fact(fact.clone());

        // Generate embedding
        let embedding = self.embedder.embed_predicate(&fact);

        // Add to vector index (create if needed)
        {
            let mut index_lock = self.predicate_index.write();
            if index_lock.is_none() {
                *index_lock = Some(VectorIndex::new(
                    self.config.embedding_dim,
                    DistanceMetric::Cosine,
                    32,  // max_nb_connection
                    100, // ef_construction
                )?);
            }

            if let Some(ref mut index) = *index_lock {
                index.insert(&cid, &embedding)?;
            }
        }

        // Store CID mapping
        self.cid_to_predicate.write().insert(cid, fact);

        Ok(())
    }

    /// Add a rule to the knowledge base
    pub fn add_rule(&mut self, head: Predicate, body: Vec<Predicate>) -> Result<()> {
        use ipfrs_tensorlogic::Rule;
        let rule = Rule { head, body };
        self.kb.write().add_rule(rule);
        Ok(())
    }

    /// Find similar predicates using semantic search
    pub fn find_similar_predicates(
        &self,
        query: &Predicate,
        k: usize,
    ) -> Result<Vec<(Cid, Predicate, f32)>> {
        let embedding = self.embedder.embed_predicate(query);

        let index_lock = self.predicate_index.read();
        let index = index_lock
            .as_ref()
            .ok_or_else(|| Error::InvalidInput("Predicate index not initialized".to_string()))?;

        let results = index.search(&embedding, k, 100)?; // ef_search = 100

        let cid_map = self.cid_to_predicate.read();
        let mut similar = Vec::new();

        for result in results {
            if let Some(pred) = cid_map.get(&result.cid) {
                similar.push((result.cid, pred.clone(), result.score));
            }
        }

        Ok(similar)
    }

    /// Query using backward chaining with semantic relevance
    pub fn query(&self, goal: &Predicate) -> Result<Vec<Substitution>> {
        let engine = self.engine.write();
        let kb = self.kb.read();
        let substs = engine.query(goal, &kb)?;
        Ok(substs)
    }

    /// Query with depth limit and semantic guidance
    pub fn query_with_depth(
        &self,
        goal: &Predicate,
        max_depth: usize,
    ) -> Result<Vec<Substitution>> {
        // Use goal decomposition for complex queries
        let decomposition = GoalDecomposition::new(goal.clone(), max_depth);

        let mut all_substs = Vec::new();

        // Solve each subgoal
        for subgoal in &decomposition.subgoals {
            let engine = self.engine.write();
            let kb = self.kb.read();
            let substs = engine.query(subgoal, &kb)?;
            all_substs.extend(substs);
        }

        Ok(all_substs)
    }

    /// Perform backward chaining with semantic search fallback
    ///
    /// This combines traditional backward chaining with semantic search:
    /// 1. Try exact unification first
    /// 2. If that fails, search for similar predicates
    /// 3. Use similarity scores to rank results
    pub fn backward_chain(&self, goal: &Predicate) -> Result<Vec<(Substitution, f32)>> {
        let mut substs_with_scores = Vec::new();

        // Try exact backward chaining first
        let exact_substs = self.query(goal)?;
        for subst in exact_substs {
            substs_with_scores.push((subst, 1.0)); // Exact match = score 1.0
        }

        // If no exact results and similarity search enabled, try semantic search
        if substs_with_scores.is_empty() {
            let similar = self.find_similar_predicates(goal, self.config.top_k_similar)?;

            for (_, similar_pred, score) in similar {
                // Score from vector search is already a similarity score
                let similarity = score;

                if similarity >= self.config.similarity_threshold {
                    // Try to prove the similar predicate
                    let substs = self.query(&similar_pred)?;
                    for subst in substs {
                        substs_with_scores.push((subst, similarity));
                    }
                }
            }
        }

        // Sort by score (descending)
        substs_with_scores
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(substs_with_scores)
    }

    /// Check if a query would create a cycle
    pub fn would_cycle(&self, goal: &Predicate, _depth: usize) -> bool {
        if !self.config.detect_cycles {
            return false;
        }

        let detector = self.cycle_detector.read();
        detector.would_cycle(goal)
    }

    /// Get knowledge base statistics
    pub fn stats(&self) -> SolverStats {
        let kb = self.kb.read();
        let kb_stats = kb.stats();

        let index_lock = self.predicate_index.read();
        let num_indexed = if let Some(ref index) = *index_lock {
            index.len()
        } else {
            0
        };

        SolverStats {
            num_facts: kb_stats.num_facts,
            num_rules: kb_stats.num_rules,
            num_indexed_predicates: num_indexed,
            embedding_dim: self.config.embedding_dim,
        }
    }

    /// Clear all data
    pub fn clear(&mut self) {
        let mut kb = self.kb.write();
        kb.facts.clear();
        kb.rules.clear();
        *self.predicate_index.write() = None;
        self.cid_to_predicate.write().clear();
        *self.cycle_detector.write() = CycleDetector::new();
    }
}

/// Statistics for the logic solver
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverStats {
    /// Number of facts in knowledge base
    pub num_facts: usize,
    /// Number of rules in knowledge base
    pub num_rules: usize,
    /// Number of indexed predicates
    pub num_indexed_predicates: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
}

/// Proof search with semantic guidance
#[allow(dead_code)]
pub struct ProofSearch {
    /// Configuration
    config: SolverConfig,
    /// Embedder for similarity
    embedder: PredicateEmbedder,
    /// Vector index for proof fragments
    proof_index: VectorIndex,
    /// Visited goals (for cycle detection)
    visited: HashSet<String>,
}

impl ProofSearch {
    /// Create a new proof search
    pub fn new(config: SolverConfig) -> Result<Self> {
        Ok(Self {
            embedder: PredicateEmbedder::new(config.embedding_dim),
            proof_index: VectorIndex::new(
                config.embedding_dim,
                DistanceMetric::Cosine,
                32,  // max_nb_connection
                100, // ef_construction
            )?,
            visited: HashSet::new(),
            config,
        })
    }

    /// Search for proof trees using BFS with semantic guidance
    pub fn search_proof_tree(
        &mut self,
        goal: &Predicate,
        kb: &KnowledgeBase,
    ) -> Result<Vec<ProofTreeNode>> {
        let mut queue: VecDeque<ProofTreeNode> = VecDeque::new();
        let mut proof_tree = Vec::new();

        // Initialize with root goal
        let root = ProofTreeNode {
            goal: goal.clone(),
            substitution: HashMap::new(),
            parent: None,
            depth: 0,
            relevance: 1.0,
        };

        queue.push_back(root);
        self.visited.clear();

        while let Some(node) = queue.pop_front() {
            // Check depth limit
            if node.depth >= self.config.max_depth {
                continue;
            }

            // Check if already visited (cycle detection)
            let goal_str = node.goal.to_string();
            if self.visited.contains(&goal_str) {
                continue;
            }
            self.visited.insert(goal_str);

            let node_id = proof_tree.len();
            proof_tree.push(node.clone());

            // Try to match with facts (simplified - exact name match)
            for fact in &kb.facts {
                if node.goal.name == fact.name && node.goal.arity() == fact.arity() {
                    let child = ProofTreeNode {
                        goal: fact.clone(),
                        substitution: HashMap::new(),
                        parent: Some(node_id),
                        depth: node.depth + 1,
                        relevance: node.relevance * 1.0, // Exact match
                    };
                    queue.push_back(child);
                }
            }

            // Try to match with rules (simplified - exact name match)
            for rule in &kb.rules {
                if node.goal.name == rule.head.name && node.goal.arity() == rule.head.arity() {
                    for body_pred in &rule.body {
                        let child = ProofTreeNode {
                            goal: body_pred.clone(),
                            substitution: HashMap::new(),
                            parent: Some(node_id),
                            depth: node.depth + 1,
                            relevance: node.relevance * 0.9, // Rule inference slightly lower
                        };
                        queue.push_back(child);
                    }
                }
            }
        }

        Ok(proof_tree)
    }

    /// Extract proof from proof tree
    pub fn extract_proof(&self, tree: &[ProofTreeNode], leaf_idx: usize) -> Vec<ProofRule> {
        let mut proof_rules = Vec::new();
        let mut current_idx = Some(leaf_idx);

        while let Some(idx) = current_idx {
            if idx >= tree.len() {
                break;
            }

            let node = &tree[idx];
            proof_rules.push(ProofRule {
                head: node.goal.clone(),
                body: Vec::new(), // Simplified for now
                is_fact: true,
            });

            current_idx = node.parent;
        }

        proof_rules.reverse();
        proof_rules
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ipfrs_tensorlogic::Constant;

    #[test]
    fn test_predicate_embedder() {
        let embedder = PredicateEmbedder::new(128);

        let alice = Term::Const(Constant::String("Alice".to_string()));
        let bob = Term::Const(Constant::String("Bob".to_string()));
        let charlie = Term::Const(Constant::String("Charlie".to_string()));

        let pred1 = Predicate::new("parent".to_string(), vec![alice.clone(), bob.clone()]);
        let pred2 = Predicate::new("parent".to_string(), vec![alice.clone(), bob.clone()]);
        let pred3 = Predicate::new("parent".to_string(), vec![alice.clone(), charlie.clone()]);

        // Same predicates should have high similarity
        let sim_same = embedder.similarity(&pred1, &pred2);
        assert!(
            sim_same > 0.99,
            "Expected sim_same > 0.99, got {}",
            sim_same
        );

        // Different arguments should have lower similarity
        let sim_diff_args = embedder.similarity(&pred1, &pred3);
        assert!(
            sim_diff_args < sim_same,
            "Expected {} < {}",
            sim_diff_args,
            sim_same
        );
        assert!(
            sim_diff_args > 0.8,
            "Expected predicates with same name to have reasonable similarity, got {}",
            sim_diff_args
        );
    }

    #[test]
    fn test_solver_creation() {
        let solver = LogicSolver::with_defaults();
        assert!(solver.is_ok());

        let stats = solver
            .expect("test: LogicSolver::with_defaults should succeed")
            .stats();
        assert_eq!(stats.num_facts, 0);
        assert_eq!(stats.num_rules, 0);
    }

    #[test]
    fn test_add_fact() {
        let mut solver =
            LogicSolver::with_defaults().expect("test: LogicSolver::with_defaults should succeed");

        let alice = Term::Const(Constant::String("Alice".to_string()));
        let bob = Term::Const(Constant::String("Bob".to_string()));
        let fact = Predicate::new("parent".to_string(), vec![alice, bob]);

        let cid: Cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse()
            .expect("test: valid CID literal should parse");

        let result = solver.add_fact(fact, cid);
        assert!(result.is_ok());

        let stats = solver.stats();
        assert_eq!(stats.num_facts, 1);
        assert_eq!(stats.num_indexed_predicates, 1);
    }

    #[test]
    fn test_add_rule() {
        let mut solver =
            LogicSolver::with_defaults().expect("test: LogicSolver::with_defaults should succeed");

        let x = Term::Var("X".to_string());
        let y = Term::Var("Y".to_string());
        let z = Term::Var("Z".to_string());

        let head = Predicate::new("ancestor".to_string(), vec![x.clone(), z.clone()]);
        let body1 = Predicate::new("parent".to_string(), vec![x.clone(), y.clone()]);
        let body2 = Predicate::new("ancestor".to_string(), vec![y.clone(), z.clone()]);

        let result = solver.add_rule(head, vec![body1, body2]);
        assert!(result.is_ok());

        let stats = solver.stats();
        assert_eq!(stats.num_rules, 1);
    }

    #[test]
    fn test_query_empty() {
        let solver =
            LogicSolver::with_defaults().expect("test: LogicSolver::with_defaults should succeed");

        let alice = Term::Const(Constant::String("Alice".to_string()));
        let bob = Term::Const(Constant::String("Bob".to_string()));
        let query = Predicate::new("parent".to_string(), vec![alice, bob]);

        let result = solver.query(&query);
        assert!(result.is_ok());
        assert!(result
            .expect("test: query on empty KB should succeed")
            .is_empty());
    }

    #[test]
    fn test_proof_search_creation() {
        let config = SolverConfig::default();
        let search = ProofSearch::new(config);
        assert!(search.is_ok());
    }

    #[test]
    fn test_solver_clear() {
        let mut solver =
            LogicSolver::with_defaults().expect("test: LogicSolver::with_defaults should succeed");

        let alice = Term::Const(Constant::String("Alice".to_string()));
        let bob = Term::Const(Constant::String("Bob".to_string()));
        let fact = Predicate::new("parent".to_string(), vec![alice, bob]);

        let cid: Cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse()
            .expect("test: valid CID literal should parse");

        solver
            .add_fact(fact, cid)
            .expect("test: add_fact with valid predicate and CID should succeed");
        assert_eq!(solver.stats().num_facts, 1);

        solver.clear();
        assert_eq!(solver.stats().num_facts, 0);
    }

    #[test]
    fn test_embedding_normalization() {
        let embedder = PredicateEmbedder::new(64);

        let alice = Term::Const(Constant::String("Alice".to_string()));
        let pred = Predicate::new("person".to_string(), vec![alice]);

        let embedding = embedder.embed_predicate(&pred);

        // Check that embedding is normalized
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }
}
