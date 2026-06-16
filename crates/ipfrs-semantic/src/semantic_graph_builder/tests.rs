//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::functions::{cosine_similarity, xorshift64};
use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    fn emb_node(id: &str, emb: Vec<f64>) -> SgbGraphNode {
        SgbGraphNode::new(id, id, NodeType::Concept).with_embedding(emb)
    }
    fn plain_node(id: &str, nt: NodeType) -> SgbGraphNode {
        SgbGraphNode::new(id, id, nt)
    }
    #[test]
    fn test_cosine_same_vector() {
        let v = vec![1.0, 2.0, 3.0];
        let s = cosine_similarity(&v, &v);
        assert!((s - 1.0).abs() < 1e-9, "same vector should be 1.0, got {s}");
    }
    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let s = cosine_similarity(&a, &b);
        assert!(s.abs() < 1e-9, "orthogonal vectors should be 0.0, got {s}");
    }
    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let s = cosine_similarity(&a, &b);
        assert!(
            (s + 1.0).abs() < 1e-9,
            "opposite vectors should be -1.0, got {s}"
        );
    }
    #[test]
    fn test_cosine_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }
    #[test]
    fn test_cosine_length_mismatch() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
    }
    #[test]
    fn test_cosine_empty() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }
    #[test]
    fn test_xorshift64_different_outputs() {
        let mut state = 12345u64;
        let a = xorshift64(&mut state);
        let b = xorshift64(&mut state);
        assert_ne!(a, b);
    }
    #[test]
    fn test_xorshift64_deterministic() {
        let mut s1 = 99u64;
        let mut s2 = 99u64;
        assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
    }
    #[test]
    fn test_add_node_basic() {
        let mut b = SemanticGraphBuilder::with_defaults();
        let node = plain_node("n1", NodeType::Entity);
        b.add_node(node).expect("should add node");
        assert_eq!(b.node_count(), 1);
    }
    #[test]
    fn test_add_node_duplicate_error() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("n1", NodeType::Entity))
            .expect("test: add first node");
        let err = b.add_node(plain_node("n1", NodeType::Entity)).unwrap_err();
        assert_eq!(err, BuilderError::DuplicateNode("n1".to_owned()));
    }
    #[test]
    fn test_add_node_auto_similarity_edge() {
        let cfg = BuilderConfig {
            similarity_threshold: 0.9,
            ..BuilderConfig::default()
        };
        let mut b = SemanticGraphBuilder::new(cfg);
        b.add_node(emb_node("a", vec![1.0, 0.0]))
            .expect("test: add node a");
        b.add_node(emb_node("b", vec![1.0, 0.0]))
            .expect("test: add node b");
        assert_eq!(b.edge_count(), 1, "auto-similarity edge should be added");
    }
    #[test]
    fn test_add_node_no_auto_edge_below_threshold() {
        let cfg = BuilderConfig {
            similarity_threshold: 0.99,
            ..BuilderConfig::default()
        };
        let mut b = SemanticGraphBuilder::new(cfg);
        b.add_node(emb_node("a", vec![1.0, 0.0]))
            .expect("test: add node a");
        b.add_node(emb_node("b", vec![0.0, 1.0]))
            .expect("test: add node b");
        assert_eq!(b.edge_count(), 0);
    }
    #[test]
    fn test_remove_node_basic() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("n1", NodeType::Entity))
            .expect("test: add node n1");
        b.remove_node("n1").expect("test: remove node n1");
        assert_eq!(b.node_count(), 0);
    }
    #[test]
    fn test_remove_node_not_found() {
        let mut b = SemanticGraphBuilder::with_defaults();
        let err = b.remove_node("ghost").unwrap_err();
        assert_eq!(err, BuilderError::NodeNotFound("ghost".to_owned()));
    }
    #[test]
    fn test_remove_node_clears_edges() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("a", NodeType::Entity))
            .expect("test: add node a");
        b.add_node(plain_node("b", NodeType::Entity))
            .expect("test: add node b");
        b.add_edge(SgbGraphEdge::new("a", "b", EdgeRelation::RelatedTo, 0.5))
            .expect("test: add edge a-b");
        assert_eq!(b.edge_count(), 1);
        b.remove_node("a").expect("test: remove node a");
        assert_eq!(b.edge_count(), 0);
    }
    #[test]
    fn test_add_edge_basic() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("a", NodeType::Entity))
            .expect("test: add node a");
        b.add_node(plain_node("b", NodeType::Entity))
            .expect("test: add node b");
        b.add_edge(SgbGraphEdge::new("a", "b", EdgeRelation::RelatedTo, 0.7))
            .expect("test: add edge a-b");
        assert_eq!(b.edge_count(), 1);
    }
    #[test]
    fn test_add_edge_self_loop_error() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("a", NodeType::Entity))
            .expect("test: add node a");
        let err = b
            .add_edge(SgbGraphEdge::new("a", "a", EdgeRelation::RelatedTo, 0.5))
            .unwrap_err();
        assert_eq!(err, BuilderError::SelfLoop("a".to_owned()));
    }
    #[test]
    fn test_add_edge_invalid_weight_negative() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("a", NodeType::Entity))
            .expect("test: add node a");
        b.add_node(plain_node("b", NodeType::Entity))
            .expect("test: add node b");
        let err = b
            .add_edge(SgbGraphEdge::new("a", "b", EdgeRelation::RelatedTo, -0.1))
            .unwrap_err();
        matches!(err, BuilderError::InvalidWeight(_));
    }
    #[test]
    fn test_add_edge_invalid_weight_above_one() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("a", NodeType::Entity))
            .expect("test: add node a");
        b.add_node(plain_node("b", NodeType::Entity))
            .expect("test: add node b");
        let err = b
            .add_edge(SgbGraphEdge::new("a", "b", EdgeRelation::RelatedTo, 1.5))
            .unwrap_err();
        matches!(err, BuilderError::InvalidWeight(_));
    }
    #[test]
    fn test_add_edge_node_not_found() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("a", NodeType::Entity))
            .expect("test: add node a");
        let err = b
            .add_edge(SgbGraphEdge::new("a", "z", EdgeRelation::RelatedTo, 0.5))
            .unwrap_err();
        assert_eq!(err, BuilderError::NodeNotFound("z".to_owned()));
    }
    #[test]
    fn test_max_edges_per_node_enforced() {
        let cfg = BuilderConfig {
            max_edges_per_node: 2,
            ..Default::default()
        };
        let mut b = SemanticGraphBuilder::new(cfg);
        for i in 0..5_usize {
            b.add_node(plain_node(&format!("n{i}"), NodeType::Entity))
                .expect("test: add node ni");
        }
        for i in 1..5_usize {
            let _ = b.add_edge(SgbGraphEdge::new(
                "n0",
                format!("n{i}"),
                EdgeRelation::RelatedTo,
                i as f64 * 0.1,
            ));
        }
        let deg = b.get_node("n0").map(|n| n.degree).unwrap_or(0);
        assert!(deg <= 2, "degree should be ≤ max_edges_per_node, got {deg}");
    }
    #[test]
    fn test_build_from_text_creates_term_nodes() {
        let mut b = SemanticGraphBuilder::with_defaults();
        let nodes = b
            .build_from_text("doc1", "hello world hello")
            .expect("test: build from text");
        assert_eq!(nodes.len(), 2);
        assert!(b.nodes.contains_key("doc1::hello"));
        assert!(b.nodes.contains_key("doc1::world"));
    }
    #[test]
    fn test_build_from_text_creates_cooccurrence_edges() {
        let cfg = BuilderConfig {
            cooccurrence_window: 2,
            ..Default::default()
        };
        let mut b = SemanticGraphBuilder::new(cfg);
        b.build_from_text("d", "alpha beta gamma")
            .expect("test: build from text");
        assert!(b.edge_count() > 0, "cooccurrence edges should be present");
    }
    #[test]
    fn test_build_from_text_no_duplicate_nodes() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.build_from_text("d1", "rust rust rust")
            .expect("test: build from text with duplicates");
        assert_eq!(b.node_count(), 1, "only one Term node per unique word");
    }
    #[test]
    fn test_build_from_text_empty_string() {
        let mut b = SemanticGraphBuilder::with_defaults();
        let nodes = b
            .build_from_text("d1", "")
            .expect("test: build from empty text");
        assert!(nodes.is_empty());
    }
    #[test]
    fn test_build_from_text_term_node_type() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.build_from_text("d", "hello")
            .expect("test: build from text");
        let node = b.get_node("d::hello").expect("test: get term node");
        assert_eq!(node.node_type, NodeType::Term);
    }
    #[test]
    fn test_build_from_text_cooccur_weight_decreases_with_distance() {
        let cfg = BuilderConfig {
            cooccurrence_window: 5,
            similarity_threshold: 1.1,
            ..Default::default()
        };
        let mut b = SemanticGraphBuilder::new(cfg);
        b.build_from_text("d", "a b c")
            .expect("test: build from text");
        let ab = b
            .edges
            .iter()
            .find(|e| {
                (e.from_id == "d::a" && e.to_id == "d::b")
                    || (e.from_id == "d::b" && e.to_id == "d::a")
            })
            .map(|e| e.weight);
        let ac = b
            .edges
            .iter()
            .find(|e| {
                (e.from_id == "d::a" && e.to_id == "d::c")
                    || (e.from_id == "d::c" && e.to_id == "d::a")
            })
            .map(|e| e.weight);
        assert!(ab.unwrap_or(0.0) > ac.unwrap_or(0.0));
    }
    #[test]
    fn test_build_similarity_edges_adds_edges() {
        let cfg = BuilderConfig {
            similarity_threshold: 1.1,
            ..Default::default()
        };
        let mut b = SemanticGraphBuilder::new(cfg);
        b.add_node(emb_node("x", vec![1.0, 0.0]))
            .expect("test: add node x");
        b.add_node(emb_node("y", vec![1.0, 0.0]))
            .expect("test: add node y");
        let added = b.build_similarity_edges(Some(0.9));
        assert_eq!(added, 1);
    }
    #[test]
    fn test_build_similarity_edges_no_duplicate() {
        let cfg = BuilderConfig {
            similarity_threshold: 0.9,
            ..Default::default()
        };
        let mut b = SemanticGraphBuilder::new(cfg);
        b.add_node(emb_node("x", vec![1.0, 0.0]))
            .expect("test: add node x");
        b.add_node(emb_node("y", vec![1.0, 0.0]))
            .expect("test: add node y");
        let count_before = b.edge_count();
        let added = b.build_similarity_edges(Some(0.9));
        assert_eq!(
            added, 0,
            "no duplicates should be added; count before: {count_before}"
        );
    }
    #[test]
    fn test_build_similarity_edges_below_threshold_skipped() {
        let cfg = BuilderConfig {
            similarity_threshold: 1.1,
            ..Default::default()
        };
        let mut b = SemanticGraphBuilder::new(cfg);
        b.add_node(emb_node("x", vec![1.0, 0.0]))
            .expect("test: add node x");
        b.add_node(emb_node("y", vec![0.0, 1.0]))
            .expect("test: add node y");
        let added = b.build_similarity_edges(Some(0.99));
        assert_eq!(added, 0);
    }
    #[test]
    fn test_build_similarity_edges_no_embedding_skipped() {
        let cfg = BuilderConfig {
            similarity_threshold: 1.1,
            ..Default::default()
        };
        let mut b = SemanticGraphBuilder::new(cfg);
        b.add_node(plain_node("no_emb", NodeType::Entity))
            .expect("test: add node without embedding");
        b.add_node(emb_node("with_emb", vec![1.0, 0.0]))
            .expect("test: add node with embedding");
        let added = b.build_similarity_edges(Some(0.5));
        assert_eq!(added, 0);
    }
    #[test]
    fn test_subgraph_returns_all_nodes_no_start() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("a", NodeType::Entity))
            .expect("test: add node a");
        b.add_node(plain_node("b", NodeType::Concept))
            .expect("test: add node b");
        let q = SgbGraphQuery {
            max_depth: 10,
            ..Default::default()
        };
        let result = b.subgraph(&q).expect("test: compute subgraph");
        assert_eq!(result.len(), 2);
    }
    #[test]
    fn test_subgraph_with_start_node() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("a", NodeType::Entity))
            .expect("test: add node a");
        b.add_node(plain_node("b", NodeType::Entity))
            .expect("test: add node b");
        b.add_node(plain_node("c", NodeType::Entity))
            .expect("test: add node c");
        b.add_edge(SgbGraphEdge::new("a", "b", EdgeRelation::RelatedTo, 1.0))
            .expect("test: add edge a-b");
        let q = SgbGraphQuery {
            start_node: Some("a".to_owned()),
            max_depth: 1,
            ..Default::default()
        };
        let ids: Vec<String> = b
            .subgraph(&q)
            .expect("test: compute subgraph from a")
            .into_iter()
            .map(|n| n.id)
            .collect();
        assert!(ids.contains(&"a".to_owned()));
        assert!(ids.contains(&"b".to_owned()));
        assert!(!ids.contains(&"c".to_owned()), "c not reachable from a");
    }
    #[test]
    fn test_subgraph_node_type_filter() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("ent", NodeType::Entity))
            .expect("test: add entity node");
        b.add_node(plain_node("con", NodeType::Concept))
            .expect("test: add concept node");
        let q = SgbGraphQuery {
            node_types: vec![NodeType::Entity],
            max_depth: 10,
            ..Default::default()
        };
        let result = b.subgraph(&q).expect("test: compute filtered subgraph");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, "ent");
    }
    #[test]
    fn test_subgraph_start_node_not_found() {
        let b = SemanticGraphBuilder::with_defaults();
        let q = SgbGraphQuery {
            start_node: Some("ghost".to_owned()),
            max_depth: 5,
            ..Default::default()
        };
        assert_eq!(
            b.subgraph(&q).unwrap_err(),
            BuilderError::NodeNotFound("ghost".to_owned())
        );
    }
    #[test]
    fn test_subgraph_min_edge_weight_filter() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("a", NodeType::Entity))
            .expect("test: add node a");
        b.add_node(plain_node("b", NodeType::Entity))
            .expect("test: add node b");
        b.add_edge(SgbGraphEdge::new("a", "b", EdgeRelation::RelatedTo, 0.1))
            .expect("test: add low-weight edge a-b");
        let q = SgbGraphQuery {
            start_node: Some("a".to_owned()),
            min_edge_weight: 0.5,
            max_depth: 2,
            ..Default::default()
        };
        let result = b
            .subgraph(&q)
            .expect("test: compute weight-filtered subgraph");
        assert_eq!(result.len(), 1, "only a is reachable at weight 0.5+");
    }
    #[test]
    fn test_neighborhood_depth_0() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("a", NodeType::Entity))
            .expect("test: add node a");
        b.add_node(plain_node("b", NodeType::Entity))
            .expect("test: add node b");
        b.add_edge(SgbGraphEdge::new("a", "b", EdgeRelation::RelatedTo, 1.0))
            .expect("test: add edge a-b");
        let hood = b
            .neighborhood("a", 0)
            .expect("test: get neighborhood depth 0");
        assert_eq!(hood.len(), 1);
        assert_eq!(hood[0].id, "a");
    }
    #[test]
    fn test_neighborhood_depth_1() {
        let mut b = SemanticGraphBuilder::with_defaults();
        for id in ["a", "b", "c"] {
            b.add_node(plain_node(id, NodeType::Entity))
                .expect("test: add node");
        }
        b.add_edge(SgbGraphEdge::new("a", "b", EdgeRelation::RelatedTo, 1.0))
            .expect("test: add edge a-b");
        let hood = b
            .neighborhood("a", 1)
            .expect("test: get neighborhood depth 1");
        assert_eq!(hood.len(), 2);
    }
    #[test]
    fn test_neighborhood_node_not_found() {
        let b = SemanticGraphBuilder::with_defaults();
        assert_eq!(
            b.neighborhood("x", 1).unwrap_err(),
            BuilderError::NodeNotFound("x".to_owned())
        );
    }
    #[test]
    fn test_neighborhood_disconnected_node() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("a", NodeType::Entity))
            .expect("test: add node a");
        b.add_node(plain_node("b", NodeType::Entity))
            .expect("test: add node b");
        let hood = b
            .neighborhood("a", 3)
            .expect("test: get neighborhood of disconnected node");
        assert_eq!(hood.len(), 1);
    }
    #[test]
    fn test_path_direct() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("a", NodeType::Entity))
            .expect("test: add node a");
        b.add_node(plain_node("b", NodeType::Entity))
            .expect("test: add node b");
        b.add_edge(SgbGraphEdge::new("a", "b", EdgeRelation::RelatedTo, 1.0))
            .expect("test: add edge a-b");
        let p = b.path("a", "b").expect("test: find direct path");
        assert_eq!(p, vec!["a", "b"]);
    }
    #[test]
    fn test_path_through_intermediate() {
        let mut b = SemanticGraphBuilder::with_defaults();
        for id in ["a", "b", "c"] {
            b.add_node(plain_node(id, NodeType::Entity))
                .expect("test: add node");
        }
        b.add_edge(SgbGraphEdge::new("a", "b", EdgeRelation::RelatedTo, 1.0))
            .expect("test: add edge a-b");
        b.add_edge(SgbGraphEdge::new("b", "c", EdgeRelation::RelatedTo, 1.0))
            .expect("test: add edge b-c");
        let p = b
            .path("a", "c")
            .expect("test: find path through intermediate");
        assert_eq!(p, vec!["a", "b", "c"]);
    }
    #[test]
    fn test_path_no_path() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("a", NodeType::Entity))
            .expect("test: add node a");
        b.add_node(plain_node("b", NodeType::Entity))
            .expect("test: add node b");
        let p = b
            .path("a", "b")
            .expect("test: find path between disconnected nodes");
        assert!(p.is_empty());
    }
    #[test]
    fn test_path_same_node() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("a", NodeType::Entity))
            .expect("test: add node a");
        let p = b.path("a", "a").expect("test: find path to same node");
        assert_eq!(p, vec!["a"]);
    }
    #[test]
    fn test_path_node_not_found() {
        let b = SemanticGraphBuilder::with_defaults();
        assert!(matches!(
            b.path("x", "y").unwrap_err(),
            BuilderError::NodeNotFound(_)
        ));
    }
    #[test]
    fn test_connected_components_empty() {
        let b = SemanticGraphBuilder::with_defaults();
        assert!(b.connected_components().is_empty());
    }
    #[test]
    fn test_connected_components_single() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("a", NodeType::Entity))
            .expect("test: add node a");
        let comps = b.connected_components();
        assert_eq!(comps.len(), 1);
    }
    #[test]
    fn test_connected_components_two_isolated() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("a", NodeType::Entity))
            .expect("test: add node a");
        b.add_node(plain_node("b", NodeType::Entity))
            .expect("test: add node b");
        let comps = b.connected_components();
        assert_eq!(comps.len(), 2);
    }
    #[test]
    fn test_connected_components_connected() {
        let mut b = SemanticGraphBuilder::with_defaults();
        for id in ["a", "b", "c"] {
            b.add_node(plain_node(id, NodeType::Entity))
                .expect("test: add node");
        }
        b.add_edge(SgbGraphEdge::new("a", "b", EdgeRelation::RelatedTo, 1.0))
            .expect("test: add edge a-b");
        b.add_edge(SgbGraphEdge::new("b", "c", EdgeRelation::RelatedTo, 1.0))
            .expect("test: add edge b-c");
        let comps = b.connected_components();
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].len(), 3);
    }
    #[test]
    fn test_connected_components_two_components() {
        let mut b = SemanticGraphBuilder::with_defaults();
        for id in ["a", "b", "c", "d"] {
            b.add_node(plain_node(id, NodeType::Entity))
                .expect("test: add node");
        }
        b.add_edge(SgbGraphEdge::new("a", "b", EdgeRelation::RelatedTo, 1.0))
            .expect("test: add edge a-b");
        b.add_edge(SgbGraphEdge::new("c", "d", EdgeRelation::RelatedTo, 1.0))
            .expect("test: add edge c-d");
        let comps = b.connected_components();
        assert_eq!(comps.len(), 2);
    }
    #[test]
    fn test_merge_nodes_basic() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("a", NodeType::Entity))
            .expect("test: add node a");
        b.add_node(plain_node("b", NodeType::Entity))
            .expect("test: add node b");
        let merged = b
            .merge_nodes("a", "b", "ab".to_owned())
            .expect("test: merge nodes a and b");
        assert_eq!(merged.id, "ab");
        assert_eq!(b.node_count(), 1);
    }
    #[test]
    fn test_merge_nodes_averages_embeddings() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(emb_node("a", vec![0.0, 1.0]))
            .expect("test: add node a with embedding");
        b.add_node(emb_node("b", vec![1.0, 0.0]))
            .expect("test: add node b with embedding");
        let merged = b
            .merge_nodes("a", "b", "ab".to_owned())
            .expect("test: merge nodes with embeddings");
        let emb = merged.embedding.expect("test: get merged embedding");
        assert!((emb[0] - 0.5).abs() < 1e-9);
        assert!((emb[1] - 0.5).abs() < 1e-9);
    }
    #[test]
    fn test_merge_nodes_redirects_edges() {
        let mut b = SemanticGraphBuilder::with_defaults();
        for id in ["a", "b", "c"] {
            b.add_node(plain_node(id, NodeType::Entity))
                .expect("test: add node");
        }
        b.add_edge(SgbGraphEdge::new("a", "c", EdgeRelation::RelatedTo, 0.9))
            .expect("test: add edge a-c");
        b.merge_nodes("a", "b", "ab".to_owned())
            .expect("test: merge nodes a and b");
        let edge_exists = b.edges.iter().any(|e| {
            (e.from_id == "ab" && e.to_id == "c") || (e.from_id == "c" && e.to_id == "ab")
        });
        assert!(edge_exists, "edge should be redirected to new merged node");
    }
    #[test]
    fn test_merge_nodes_not_found() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("a", NodeType::Entity))
            .expect("test: add node a");
        assert!(matches!(
            b.merge_nodes("a", "z", "az".to_owned()).unwrap_err(),
            BuilderError::NodeNotFound(_)
        ));
    }
    #[test]
    fn test_merge_nodes_combines_attributes() {
        let mut b = SemanticGraphBuilder::with_defaults();
        let na = SgbGraphNode::new("a", "A", NodeType::Entity).with_attribute("color", "red");
        let nb = SgbGraphNode::new("b", "B", NodeType::Entity).with_attribute("size", "large");
        b.add_node(na).expect("test: add node a with attributes");
        b.add_node(nb).expect("test: add node b with attributes");
        let merged = b
            .merge_nodes("a", "b", "ab".to_owned())
            .expect("test: merge nodes with attributes");
        let has_color = merged.attributes.iter().any(|(k, _)| k == "color");
        let has_size = merged.attributes.iter().any(|(k, _)| k == "size");
        assert!(
            has_color && has_size,
            "merged node should have both attributes"
        );
    }
    #[test]
    fn test_merge_nodes_no_self_loop() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("a", NodeType::Entity))
            .expect("test: add node a");
        b.add_node(plain_node("b", NodeType::Entity))
            .expect("test: add node b");
        b.add_edge(SgbGraphEdge::new("a", "b", EdgeRelation::RelatedTo, 0.8))
            .expect("test: add edge a-b");
        b.merge_nodes("a", "b", "ab".to_owned())
            .expect("test: merge nodes a and b");
        let self_loop = b.edges.iter().any(|e| e.from_id == e.to_id);
        assert!(!self_loop, "no self-loops after merge");
    }
    #[test]
    fn test_stats_empty() {
        let b = SemanticGraphBuilder::with_defaults();
        let s = b.stats();
        assert_eq!(s.node_count, 0);
        assert_eq!(s.edge_count, 0);
        assert_eq!(s.connected_components, 0);
    }
    #[test]
    fn test_stats_node_edge_count() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("a", NodeType::Entity))
            .expect("test: add node a");
        b.add_node(plain_node("b", NodeType::Entity))
            .expect("test: add node b");
        b.add_edge(SgbGraphEdge::new("a", "b", EdgeRelation::RelatedTo, 1.0))
            .expect("test: add edge a-b");
        let s = b.stats();
        assert_eq!(s.node_count, 2);
        assert_eq!(s.edge_count, 1);
    }
    #[test]
    fn test_stats_avg_degree() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("a", NodeType::Entity))
            .expect("test: add node a");
        b.add_node(plain_node("b", NodeType::Entity))
            .expect("test: add node b");
        b.add_edge(SgbGraphEdge::new("a", "b", EdgeRelation::RelatedTo, 1.0))
            .expect("test: add edge a-b");
        let s = b.stats();
        assert!(
            (s.avg_degree - 1.0).abs() < 1e-9,
            "avg_degree should be 1.0, got {}",
            s.avg_degree
        );
    }
    #[test]
    fn test_stats_density() {
        let mut b = SemanticGraphBuilder::with_defaults();
        for id in ["a", "b", "c"] {
            b.add_node(plain_node(id, NodeType::Entity))
                .expect("test: add node");
        }
        b.add_edge(SgbGraphEdge::new("a", "b", EdgeRelation::RelatedTo, 1.0))
            .expect("test: add edge a-b");
        let s = b.stats();
        let expected = 1.0 / 6.0;
        assert!((s.density - expected).abs() < 1e-9);
    }
    #[test]
    fn test_stats_components() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("a", NodeType::Entity))
            .expect("test: add node a");
        b.add_node(plain_node("b", NodeType::Entity))
            .expect("test: add node b");
        let s = b.stats();
        assert_eq!(s.connected_components, 2);
        assert_eq!(s.largest_component_size, 1);
    }
    #[test]
    fn test_edge_relation_defined_eq() {
        let a = EdgeRelation::Defined("custom".to_owned());
        let b = EdgeRelation::Defined("custom".to_owned());
        let c = EdgeRelation::Defined("other".to_owned());
        assert!(a.variant_eq(&b));
        assert!(!a.variant_eq(&c));
    }
    #[test]
    fn test_edge_relation_matches_empty_filter() {
        let r = EdgeRelation::RelatedTo;
        assert!(r.matches(&[]));
    }
    #[test]
    fn test_edge_relation_matches_with_filter() {
        let r = EdgeRelation::SimilarTo;
        assert!(r.matches(&[EdgeRelation::SimilarTo]));
        assert!(!r.matches(&[EdgeRelation::PartOf]));
    }
    #[test]
    fn test_transitive_closure_applies() {
        let cfg = BuilderConfig {
            enable_transitive_closure: true,
            similarity_threshold: 1.1,
            ..Default::default()
        };
        let mut b = SemanticGraphBuilder::new(cfg);
        for id in ["a", "b", "c"] {
            b.add_node(plain_node(id, NodeType::Entity))
                .expect("test: add node");
        }
        b.add_edge(SgbGraphEdge::new("a", "b", EdgeRelation::PartOf, 1.0))
            .expect("test: add edge a-b PartOf");
        b.add_edge(SgbGraphEdge::new("b", "c", EdgeRelation::PartOf, 1.0))
            .expect("test: add edge b-c PartOf");
        b.apply_transitive_closure();
        let has_a_c = b.edges.iter().any(|e| {
            e.from_id == "a" && e.to_id == "c" && e.relation.variant_eq(&EdgeRelation::PartOf)
        });
        assert!(has_a_c, "transitive PartOf a→c should exist");
    }
    #[test]
    fn test_transitive_closure_disabled_no_effect() {
        let cfg = BuilderConfig {
            enable_transitive_closure: false,
            similarity_threshold: 1.1,
            ..Default::default()
        };
        let mut b = SemanticGraphBuilder::new(cfg);
        for id in ["a", "b", "c"] {
            b.add_node(plain_node(id, NodeType::Entity))
                .expect("test: add node");
        }
        b.add_edge(SgbGraphEdge::new("a", "b", EdgeRelation::PartOf, 1.0))
            .expect("test: add edge a-b PartOf");
        b.add_edge(SgbGraphEdge::new("b", "c", EdgeRelation::PartOf, 1.0))
            .expect("test: add edge b-c PartOf");
        b.apply_transitive_closure();
        assert_eq!(b.edge_count(), 2, "no new edges when disabled");
    }
    #[test]
    fn test_iter_nodes() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("x", NodeType::Document))
            .expect("test: add document node x");
        b.add_node(plain_node("y", NodeType::Document))
            .expect("test: add document node y");
        assert_eq!(b.iter_nodes().count(), 2);
    }
    #[test]
    fn test_iter_edges() {
        let mut b = SemanticGraphBuilder::with_defaults();
        b.add_node(plain_node("x", NodeType::Entity))
            .expect("test: add node x");
        b.add_node(plain_node("y", NodeType::Entity))
            .expect("test: add node y");
        b.add_edge(SgbGraphEdge::new("x", "y", EdgeRelation::RelatedTo, 1.0))
            .expect("test: add edge x-y");
        assert_eq!(b.iter_edges().count(), 1);
    }
    #[test]
    fn test_node_with_embedding() {
        let n = SgbGraphNode::new("id", "label", NodeType::Cluster).with_embedding(vec![1.0, 2.0]);
        assert!(n.embedding.is_some());
    }
    #[test]
    fn test_node_with_attribute() {
        let n = SgbGraphNode::new("id", "label", NodeType::Entity).with_attribute("key", "value");
        assert_eq!(n.attributes, vec![("key".to_owned(), "value".to_owned())]);
    }
    #[test]
    fn test_builder_error_display() {
        assert!(BuilderError::NodeNotFound("x".to_owned())
            .to_string()
            .contains("x"));
        assert!(BuilderError::DuplicateNode("x".to_owned())
            .to_string()
            .contains("x"));
        assert!(BuilderError::SelfLoop("x".to_owned())
            .to_string()
            .contains("x"));
        assert!(BuilderError::InvalidWeight(9.9).to_string().contains("9.9"));
        assert!(BuilderError::GraphTooLarge(100).to_string().contains("100"));
    }
    #[test]
    fn test_large_graph_stress() {
        let cfg = BuilderConfig {
            max_edges_per_node: 20,
            similarity_threshold: 1.1,
            ..Default::default()
        };
        let mut b = SemanticGraphBuilder::new(cfg);
        let n = 50_usize;
        let mut state: u64 = 0xDEAD_BEEF;
        for i in 0..n {
            let emb: Vec<f64> = (0..8)
                .map(|_| {
                    let r = xorshift64(&mut state);
                    (r % 1000) as f64 / 1000.0
                })
                .collect();
            b.add_node(emb_node(&format!("n{i}"), emb))
                .expect("test: add stress-test node");
        }
        let added = b.build_similarity_edges(Some(0.5));
        let s = b.stats();
        assert_eq!(s.node_count, n);
        assert!(s.edge_count <= added + 1);
    }
    #[test]
    fn test_union_find_basic() {
        let mut uf = UnionFind::new(4);
        uf.union(0, 1);
        uf.union(2, 3);
        assert_eq!(uf.find(0), uf.find(1));
        assert_eq!(uf.find(2), uf.find(3));
        assert_ne!(uf.find(0), uf.find(2));
    }
    #[test]
    fn test_union_find_path_compression() {
        let mut uf = UnionFind::new(5);
        uf.union(0, 1);
        uf.union(1, 2);
        uf.union(2, 3);
        uf.union(3, 4);
        let root = uf.find(0);
        for i in 1..5 {
            assert_eq!(uf.find(i), root);
        }
    }
}
