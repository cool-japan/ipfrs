//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    fn make_tracer() -> CausalChainTracer {
        CausalChainTracer::new(TracerConfig {
            max_chain_depth: 16,
            min_edge_strength: 0.0,
            max_nodes: 10_000,
            enable_cycle_detection: true,
            confidence_threshold: 0.0,
        })
    }
    fn node(id: &str, etype: &str, ts: u64) -> CausalNode {
        CausalNode::new(id, etype, ts, 1.0)
    }
    fn edge(from: &str, to: &str, rel: CausalRelation, strength: f64) -> CausalEdge {
        CausalEdge::new(from, to, rel, strength, 0)
    }
    #[test]
    fn test_add_node_basic() {
        let mut t = make_tracer();
        let n = node("A", "login", 1000);
        assert!(t.add_node(n).is_ok());
        assert_eq!(t.stats().nodes_tracked, 1);
    }
    #[test]
    fn test_add_node_duplicate_overwrite() {
        let mut t = make_tracer();
        t.add_node(node("A", "login", 100))
            .expect("test setup: add_node failed");
        t.add_node(node("A", "logout", 200))
            .expect("test setup: add_node failed");
        assert_eq!(t.stats().nodes_tracked, 1);
    }
    #[test]
    fn test_add_node_max_nodes() {
        let mut t = CausalChainTracer::new(TracerConfig {
            max_nodes: 2,
            ..Default::default()
        });
        t.add_node(node("A", "x", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "x", 0))
            .expect("test setup: add_node failed");
        let err = t.add_node(node("C", "x", 0)).unwrap_err();
        assert!(matches!(err, TracerError::QueryTooExpensive(_)));
    }
    #[test]
    fn test_add_edge_basic() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        assert!(t
            .add_edge(edge("A", "B", CausalRelation::DirectCause, 0.8))
            .is_ok());
        assert_eq!(t.stats().edges_tracked, 1);
    }
    #[test]
    fn test_add_edge_missing_from() {
        let mut t = make_tracer();
        t.add_node(node("B", "e", 0))
            .expect("test setup: add_node failed");
        let err = t
            .add_edge(edge("X", "B", CausalRelation::Enables, 0.5))
            .unwrap_err();
        assert!(matches!(err, TracerError::NodeNotFound(_)));
    }
    #[test]
    fn test_add_edge_missing_to() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        let err = t
            .add_edge(edge("A", "Y", CausalRelation::Enables, 0.5))
            .unwrap_err();
        assert!(matches!(err, TracerError::NodeNotFound(_)));
    }
    #[test]
    fn test_add_edge_strength_too_high() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        let err = t
            .add_edge(edge("A", "B", CausalRelation::Precedes, 1.5))
            .unwrap_err();
        assert!(matches!(err, TracerError::InvalidStrength(_)));
    }
    #[test]
    fn test_add_edge_strength_negative() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        let err = t
            .add_edge(edge("A", "B", CausalRelation::Precedes, -0.1))
            .unwrap_err();
        assert!(matches!(err, TracerError::InvalidStrength(_)));
    }
    #[test]
    fn test_add_edge_boundary_strengths() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_node(node("C", "e", 2))
            .expect("test setup: add_node failed");
        assert!(t
            .add_edge(edge("A", "B", CausalRelation::Precedes, 0.0))
            .is_ok());
        assert!(t
            .add_edge(edge("B", "C", CausalRelation::Precedes, 1.0))
            .is_ok());
    }
    #[test]
    fn test_cycle_direct() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let err = t
            .add_edge(edge("B", "A", CausalRelation::DirectCause, 0.9))
            .unwrap_err();
        assert!(matches!(err, TracerError::CycleDetected { .. }));
    }
    #[test]
    fn test_cycle_indirect() {
        let mut t = make_tracer();
        for id in ["A", "B", "C"] {
            t.add_node(node(id, "e", 0))
                .expect("test setup: add_node failed");
        }
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.8))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "C", CausalRelation::DirectCause, 0.8))
            .expect("test setup: add_edge failed");
        let err = t
            .add_edge(edge("C", "A", CausalRelation::DirectCause, 0.8))
            .unwrap_err();
        assert!(matches!(err, TracerError::CycleDetected { .. }));
    }
    #[test]
    fn test_cycle_self_loop() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        let err = t
            .add_edge(edge("A", "A", CausalRelation::DirectCause, 0.5))
            .unwrap_err();
        assert!(matches!(err, TracerError::CycleDetected { .. }));
    }
    #[test]
    fn test_cycle_detection_disabled() {
        let mut t = CausalChainTracer::new(TracerConfig {
            enable_cycle_detection: false,
            ..Default::default()
        });
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        assert!(t
            .add_edge(edge("B", "A", CausalRelation::DirectCause, 0.9))
            .is_ok());
    }
    #[test]
    fn test_trace_simple_chain() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let q = TraceQuery {
            root_event_id: Some("A".to_string()),
            max_depth: 4,
            ..Default::default()
        };
        let chains = t.trace(&q).expect("test setup: trace failed");
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].nodes.len(), 2);
        assert!((chains[0].chain_confidence - 0.9).abs() < 1e-9);
    }
    #[test]
    fn test_trace_confidence_product() {
        let mut t = make_tracer();
        for (id, ts) in [("A", 0u64), ("B", 1), ("C", 2)] {
            t.add_node(node(id, "e", ts))
                .expect("test setup: add_node failed");
        }
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.8))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "C", CausalRelation::DirectCause, 0.5))
            .expect("test setup: add_edge failed");
        let q = TraceQuery {
            root_event_id: Some("A".to_string()),
            max_depth: 4,
            ..Default::default()
        };
        let chains = t.trace(&q).expect("test setup: trace failed");
        assert_eq!(chains.len(), 1);
        assert!((chains[0].chain_confidence - 0.4).abs() < 1e-9);
    }
    #[test]
    fn test_trace_max_depth() {
        let mut t = make_tracer();
        for (id, ts) in [("A", 0u64), ("B", 1), ("C", 2), ("D", 3)] {
            t.add_node(node(id, "e", ts))
                .expect("test setup: add_node failed");
        }
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "C", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("C", "D", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let q = TraceQuery {
            root_event_id: Some("A".to_string()),
            max_depth: 1,
            ..Default::default()
        };
        let chains = t.trace(&q).expect("test setup: trace failed");
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].depth, 1);
    }
    #[test]
    fn test_trace_min_strength_filter() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_node(node("C", "e", 2))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.3))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("A", "C", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let q = TraceQuery {
            root_event_id: Some("A".to_string()),
            min_strength: 0.5,
            max_depth: 4,
            ..Default::default()
        };
        let chains = t.trace(&q).expect("test setup: trace failed");
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].leaf_ids[0], "C");
    }
    #[test]
    fn test_trace_event_type_filter() {
        let mut t = make_tracer();
        t.add_node(node("A", "request", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "response", 1))
            .expect("test setup: add_node failed");
        t.add_node(node("C", "error", 2))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("A", "C", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let q = TraceQuery {
            root_event_id: Some("A".to_string()),
            event_types: vec!["request".to_string(), "response".to_string()],
            max_depth: 4,
            ..Default::default()
        };
        let chains = t.trace(&q).expect("test setup: trace failed");
        assert_eq!(chains.len(), 1);
        assert!(chains[0].nodes.iter().all(|n| n.event_type != "error"));
    }
    #[test]
    fn test_trace_time_window_filter() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 100))
            .expect("test setup: add_node failed");
        t.add_node(node("C", "e", 500))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("A", "C", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let q = TraceQuery {
            root_event_id: Some("A".to_string()),
            time_window_us: Some((0, 200)),
            max_depth: 4,
            ..Default::default()
        };
        let chains = t.trace(&q).expect("test setup: trace failed");
        assert!(chains
            .iter()
            .all(|c| !c.leaf_ids.contains(&"C".to_string())));
    }
    #[test]
    fn test_trace_relation_filter() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_node(node("C", "e", 2))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("A", "C", CausalRelation::Correlates, 0.9))
            .expect("test setup: add_edge failed");
        let q = TraceQuery {
            root_event_id: Some("A".to_string()),
            include_relations: vec![CausalRelation::DirectCause],
            max_depth: 4,
            ..Default::default()
        };
        let chains = t.trace(&q).expect("test setup: trace failed");
        assert!(chains
            .iter()
            .all(|c| !c.leaf_ids.contains(&"C".to_string())));
    }
    #[test]
    fn test_trace_confidence_threshold() {
        let mut t = CausalChainTracer::new(TracerConfig {
            confidence_threshold: 0.5,
            ..Default::default()
        });
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_node(node("C", "e", 2))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("A", "C", CausalRelation::DirectCause, 0.3))
            .expect("test setup: add_edge failed");
        let q = TraceQuery {
            root_event_id: Some("A".to_string()),
            max_depth: 4,
            ..Default::default()
        };
        let chains = t.trace(&q).expect("test setup: trace failed");
        assert!(chains.iter().all(|c| c.chain_confidence >= 0.5));
    }
    #[test]
    fn test_trace_branching() {
        let mut t = make_tracer();
        for id in ["A", "B", "C", "D"] {
            t.add_node(node(id, "e", 0))
                .expect("test setup: add_node failed");
        }
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.8))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("A", "C", CausalRelation::DirectCause, 0.7))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "D", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("C", "D", CausalRelation::DirectCause, 0.6))
            .expect("test setup: add_edge failed");
        let q = TraceQuery {
            root_event_id: Some("A".to_string()),
            max_depth: 8,
            ..Default::default()
        };
        let chains = t.trace(&q).expect("test setup: trace failed");
        assert_eq!(chains.len(), 2);
    }
    #[test]
    fn test_trace_empty_graph() {
        let mut t = make_tracer();
        let q = TraceQuery::default();
        let chains = t.trace(&q).expect("test setup: trace failed");
        assert!(chains.is_empty());
    }
    #[test]
    fn test_trace_root_not_found() {
        let mut t = make_tracer();
        let q = TraceQuery {
            root_event_id: Some("MISSING".to_string()),
            ..Default::default()
        };
        let err = t.trace(&q).unwrap_err();
        assert!(matches!(err, TracerError::NodeNotFound(_)));
    }
    #[test]
    fn test_shortest_path_direct() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let path = t
            .shortest_path("A", "B")
            .expect("test setup: shortest_path failed")
            .expect("test setup: expected Some path");
        assert_eq!(path, vec!["A", "B"]);
    }
    #[test]
    fn test_shortest_path_multi_hop() {
        let mut t = make_tracer();
        for id in ["A", "B", "C"] {
            t.add_node(node(id, "e", 0))
                .expect("test setup: add_node failed");
        }
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.8))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "C", CausalRelation::DirectCause, 0.8))
            .expect("test setup: add_edge failed");
        let path = t
            .shortest_path("A", "C")
            .expect("test setup: shortest_path failed")
            .expect("test setup: expected Some path");
        assert_eq!(path.len(), 3);
    }
    #[test]
    fn test_shortest_path_no_path() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        let result = t
            .shortest_path("A", "B")
            .expect("test setup: shortest_path failed");
        assert!(result.is_none());
    }
    #[test]
    fn test_shortest_path_same_node() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        let path = t
            .shortest_path("A", "A")
            .expect("test setup: shortest_path failed")
            .expect("test setup: expected Some path");
        assert_eq!(path, vec!["A"]);
    }
    #[test]
    fn test_shortest_path_missing_node() {
        let t = make_tracer();
        let err = t.shortest_path("X", "Y").unwrap_err();
        assert!(matches!(err, TracerError::NodeNotFound(_)));
    }
    #[test]
    fn test_shortest_path_is_shortest() {
        let mut t = make_tracer();
        for id in ["A", "B", "C", "D"] {
            t.add_node(node(id, "e", 0))
                .expect("test setup: add_node failed");
        }
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "D", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("A", "C", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("C", "D", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let path = t
            .shortest_path("A", "D")
            .expect("test setup: shortest_path failed")
            .expect("test setup: expected Some path");
        assert_eq!(path.len(), 3);
    }
    #[test]
    fn test_strongest_path_simple() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.7))
            .expect("test setup: add_edge failed");
        let chain = t
            .strongest_path("A", "B")
            .expect("test setup: strongest_path failed")
            .expect("test setup: expected Some chain");
        assert!((chain.chain_confidence - 0.7).abs() < 1e-9);
    }
    #[test]
    fn test_strongest_path_picks_higher() {
        let mut t = make_tracer();
        for id in ["A", "B", "C", "D"] {
            t.add_node(node(id, "e", 0))
                .expect("test setup: add_node failed");
        }
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "D", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("A", "C", CausalRelation::DirectCause, 0.5))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("C", "D", CausalRelation::DirectCause, 0.5))
            .expect("test setup: add_edge failed");
        let chain = t
            .strongest_path("A", "D")
            .expect("test setup: strongest_path failed")
            .expect("test setup: expected Some chain");
        assert!(chain.chain_confidence > 0.8);
    }
    #[test]
    fn test_strongest_path_same_node() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        let chain = t
            .strongest_path("A", "A")
            .expect("test setup: strongest_path failed")
            .expect("test setup: expected Some chain");
        assert!((chain.chain_confidence - 1.0).abs() < 1e-9);
        assert_eq!(chain.depth, 0);
    }
    #[test]
    fn test_strongest_path_no_path() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        let result = t
            .strongest_path("A", "B")
            .expect("test setup: strongest_path failed");
        assert!(result.is_none());
    }
    #[test]
    fn test_strongest_path_missing() {
        let t = make_tracer();
        let err = t.strongest_path("X", "Y").unwrap_err();
        assert!(matches!(err, TracerError::NodeNotFound(_)));
    }
    #[test]
    fn test_root_causes_single() {
        let mut t = make_tracer();
        for id in ["A", "B", "C"] {
            t.add_node(node(id, "e", 0))
                .expect("test setup: add_node failed");
        }
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "C", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let roots = t.root_causes("C").expect("test setup: root_causes failed");
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].id, "A");
    }
    #[test]
    fn test_root_causes_multiple() {
        let mut t = make_tracer();
        for id in ["A", "B", "C"] {
            t.add_node(node(id, "e", 0))
                .expect("test setup: add_node failed");
        }
        t.add_edge(edge("A", "C", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "C", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let mut roots = t.root_causes("C").expect("test setup: root_causes failed");
        roots.sort_by(|a, b| a.id.cmp(&b.id));
        assert_eq!(roots.len(), 2);
        assert_eq!(roots[0].id, "A");
        assert_eq!(roots[1].id, "B");
    }
    #[test]
    fn test_root_causes_no_ancestors() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        let roots = t.root_causes("A").expect("test setup: root_causes failed");
        assert!(roots.is_empty());
    }
    #[test]
    fn test_root_causes_missing() {
        let t = make_tracer();
        let err = t.root_causes("MISSING").unwrap_err();
        assert!(matches!(err, TracerError::NodeNotFound(_)));
    }
    #[test]
    fn test_downstream_effects_depth1() {
        let mut t = make_tracer();
        for id in ["A", "B", "C", "D"] {
            t.add_node(node(id, "e", 0))
                .expect("test setup: add_node failed");
        }
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "C", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "D", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let effects = t
            .downstream_effects("A", 1)
            .expect("test setup: downstream_effects failed");
        let ids: Vec<&str> = effects.iter().map(|n| n.id.as_str()).collect();
        assert!(ids.contains(&"B"));
        assert!(!ids.contains(&"C"));
    }
    #[test]
    fn test_downstream_effects_deep() {
        let mut t = make_tracer();
        for (id, ts) in [("A", 0u64), ("B", 1), ("C", 2), ("D", 3)] {
            t.add_node(node(id, "e", ts))
                .expect("test setup: add_node failed");
        }
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "C", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("C", "D", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let effects = t
            .downstream_effects("A", 10)
            .expect("test setup: downstream_effects failed");
        assert_eq!(effects.len(), 3);
    }
    #[test]
    fn test_downstream_effects_missing() {
        let t = make_tracer();
        let err = t.downstream_effects("MISSING", 5).unwrap_err();
        assert!(matches!(err, TracerError::NodeNotFound(_)));
    }
    #[test]
    fn test_downstream_effects_zero_depth() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let effects = t
            .downstream_effects("A", 0)
            .expect("test setup: downstream_effects failed");
        assert!(effects.is_empty());
    }
    #[test]
    fn test_remove_node_basic() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.8))
            .expect("test setup: add_edge failed");
        t.remove_node("A").expect("test setup: remove_node failed");
        assert_eq!(t.stats().nodes_tracked, 1);
        assert_eq!(t.stats().edges_tracked, 0);
    }
    #[test]
    fn test_remove_node_removes_edges() {
        let mut t = make_tracer();
        for id in ["A", "B", "C"] {
            t.add_node(node(id, "e", 0))
                .expect("test setup: add_node failed");
        }
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("B", "C", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        t.remove_node("B").expect("test setup: remove_node failed");
        assert_eq!(t.stats().edges_tracked, 0);
    }
    #[test]
    fn test_remove_node_missing() {
        let mut t = make_tracer();
        let err = t.remove_node("MISSING").unwrap_err();
        assert!(matches!(err, TracerError::NodeNotFound(_)));
    }
    #[test]
    fn test_stats_basic() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let s = t.stats();
        assert_eq!(s.nodes_tracked, 2);
        assert_eq!(s.edges_tracked, 1);
    }
    #[test]
    fn test_stats_cycles_detected() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let _ = t.add_edge(edge("B", "A", CausalRelation::DirectCause, 0.9));
        assert_eq!(t.stats().cycles_detected, 1);
    }
    #[test]
    fn test_xorshift64_deterministic() {
        let mut state = 12345u64;
        let v1 = xorshift64(&mut state);
        let mut state2 = 12345u64;
        let v2 = xorshift64(&mut state2);
        assert_eq!(v1, v2);
        assert_ne!(v1, 0);
    }
    #[test]
    fn test_large_graph_stress() {
        let mut t = make_tracer();
        let mut rng = 999_999u64;
        let n = 50usize;
        for i in 0..n {
            t.add_node(node(&i.to_string(), "stress", xorshift64(&mut rng) % 1000))
                .expect("test setup: add_node failed");
        }
        let mut added = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                let r = xorshift64(&mut rng);
                if r.is_multiple_of(5) {
                    let strength = (r % 100) as f64 / 100.0;
                    let _ = t.add_edge(edge(
                        &i.to_string(),
                        &j.to_string(),
                        CausalRelation::DirectCause,
                        strength,
                    ));
                    added += 1;
                }
            }
        }
        assert!(added > 0, "expected some edges");
        let q = TraceQuery {
            root_event_id: Some("0".to_string()),
            max_depth: 5,
            ..Default::default()
        };
        let chains = t.trace(&q).expect("test setup: trace failed");
        assert!(chains.len() <= 10_000);
    }
    #[test]
    fn test_trace_no_relation_filter() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_node(node("C", "e", 2))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::Inhibits, 0.9))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("A", "C", CausalRelation::IndirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let q = TraceQuery {
            root_event_id: Some("A".to_string()),
            include_relations: vec![],
            max_depth: 4,
            ..Default::default()
        };
        let chains = t.trace(&q).expect("test setup: trace failed");
        assert_eq!(chains.len(), 2);
    }
    #[test]
    fn test_causal_node_with_attribute() {
        let n = CausalNode::new("A", "login", 0, 1.0)
            .with_attribute("ip", "192.168.1.1")
            .with_attribute("user", "bob");
        assert_eq!(n.attributes.len(), 2);
        assert_eq!(
            n.attributes[0],
            ("ip".to_string(), "192.168.1.1".to_string())
        );
    }
    #[test]
    fn test_trace_implicit_roots() {
        let mut t = make_tracer();
        t.add_node(node("R1", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("R2", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("X", "e", 1))
            .expect("test setup: add_node failed");
        t.add_edge(edge("R1", "X", CausalRelation::DirectCause, 0.8))
            .expect("test setup: add_edge failed");
        t.add_edge(edge("R2", "X", CausalRelation::DirectCause, 0.7))
            .expect("test setup: add_edge failed");
        let q = TraceQuery {
            root_event_id: None,
            max_depth: 4,
            ..Default::default()
        };
        let chains = t.trace(&q).expect("test setup: trace failed");
        assert_eq!(chains.len(), 2);
    }
    #[test]
    fn test_tracer_error_display() {
        let e = TracerError::NodeNotFound("ABC".to_string());
        assert!(e.to_string().contains("ABC"));
        let e2 = TracerError::InvalidStrength(2.0);
        assert!(e2.to_string().contains("2"));
        let e3 = TracerError::CycleDetected {
            path: vec!["A".to_string(), "B".to_string()],
        };
        assert!(e3.to_string().contains("A -> B"));
    }
    #[test]
    fn test_chains_traced_counter() {
        let mut t = make_tracer();
        t.add_node(node("A", "e", 0))
            .expect("test setup: add_node failed");
        t.add_node(node("B", "e", 1))
            .expect("test setup: add_node failed");
        t.add_edge(edge("A", "B", CausalRelation::DirectCause, 0.9))
            .expect("test setup: add_edge failed");
        let q = TraceQuery {
            root_event_id: Some("A".to_string()),
            max_depth: 4,
            ..Default::default()
        };
        t.trace(&q).expect("test setup: trace failed");
        t.trace(&q).expect("test setup: trace failed");
        assert_eq!(t.stats().chains_traced, 2);
    }
}
