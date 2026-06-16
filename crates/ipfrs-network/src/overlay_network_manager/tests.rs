//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    fn make_manager() -> OverlayNetworkManager {
        OverlayNetworkManager::new(OverlayConfig::default())
    }
    fn add_nodes(mgr: &mut OverlayNetworkManager, ids: &[&str]) {
        for &id in ids {
            let n = OverlayNode::new(id, "127.0.0.1:4001", "10.0.0.1", "us-east", 1000);
            mgr.add_node(n).expect("add_node failed");
        }
    }
    fn add_link(mgr: &mut OverlayNetworkManager, a: &str, b: &str) {
        let link = OverlayLink::new(a, b, 10, 1_000_000);
        mgr.add_link(link).expect("add_link failed");
    }
    fn three_node_line() -> OverlayNetworkManager {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["a", "b", "c"]);
        add_link(&mut mgr, "a", "b");
        add_link(&mut mgr, "b", "c");
        mgr
    }
    #[test]
    fn t001_add_node_ok() {
        let mut mgr = make_manager();
        let n = OverlayNode::new("x", "1.2.3.4:4001", "10.0.0.1", "eu", 500);
        assert!(mgr.add_node(n).is_ok());
        assert_eq!(mgr.node_count(), 1);
    }
    #[test]
    fn t002_add_duplicate_node_replaces() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["x"]);
        add_nodes(&mut mgr, &["x"]);
        assert_eq!(mgr.node_count(), 1);
    }
    #[test]
    fn t003_remove_node_ok() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["x", "y"]);
        add_link(&mut mgr, "x", "y");
        assert!(mgr.remove_node("x").is_ok());
        assert_eq!(mgr.node_count(), 1);
        assert_eq!(mgr.link_count(), 0);
    }
    #[test]
    fn t004_remove_unknown_node_err() {
        let mut mgr = make_manager();
        let r = mgr.remove_node("ghost");
        assert!(matches!(r, Err(OverlayError::NodeNotFound(_))));
    }
    #[test]
    fn t005_max_nodes_exceeded() {
        let cfg = OverlayConfig {
            max_nodes: 2,
            ..Default::default()
        };
        let mut mgr = OverlayNetworkManager::new(cfg);
        add_nodes(&mut mgr, &["a", "b"]);
        let n = OverlayNode::new("c", "", "", "x", 0);
        assert!(matches!(
            mgr.add_node(n),
            Err(OverlayError::MaxNodesExceeded)
        ));
    }
    #[test]
    fn t006_node_count_reflects_removes() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["a", "b", "c"]);
        mgr.remove_node("b").expect("test: remove node");
        assert_eq!(mgr.node_count(), 2);
    }
    #[test]
    fn t007_node_links_pruned_on_remove() {
        let mut mgr = three_node_line();
        mgr.remove_node("b").expect("test: remove node");
        assert_eq!(mgr.link_count(), 0);
    }
    #[test]
    fn t008_get_node_existing() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["z"]);
        assert!(mgr.get_node("z").is_some());
    }
    #[test]
    fn t009_get_node_missing() {
        let mgr = make_manager();
        assert!(mgr.get_node("ghost").is_none());
    }
    #[test]
    fn t010_gateway_flag_preserved() {
        let mut mgr = make_manager();
        let mut n = OverlayNode::new("gw", "1.2.3.4:4001", "10.0.0.1", "eu", 1000);
        n.is_gateway = true;
        mgr.add_node(n).expect("test: add node");
        let gws = mgr.gateway_nodes();
        assert_eq!(gws.len(), 1);
        assert_eq!(gws[0].id, "gw");
    }
    #[test]
    fn t011_add_link_ok() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["a", "b"]);
        add_link(&mut mgr, "a", "b");
        assert_eq!(mgr.link_count(), 1);
    }
    #[test]
    fn t012_add_link_missing_from() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["b"]);
        let link = OverlayLink::new("ghost", "b", 5, 1000);
        assert!(matches!(
            mgr.add_link(link),
            Err(OverlayError::NodeNotFound(_))
        ));
    }
    #[test]
    fn t013_add_link_missing_to() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["a"]);
        let link = OverlayLink::new("a", "ghost", 5, 1000);
        assert!(matches!(
            mgr.add_link(link),
            Err(OverlayError::NodeNotFound(_))
        ));
    }
    #[test]
    fn t014_remove_link_ok() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["a", "b"]);
        add_link(&mut mgr, "a", "b");
        assert!(mgr.remove_link("a", "b").is_ok());
        assert_eq!(mgr.link_count(), 0);
    }
    #[test]
    fn t015_remove_link_missing() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["a", "b"]);
        let r = mgr.remove_link("a", "b");
        assert!(matches!(r, Err(OverlayError::LinkNotFound { .. })));
    }
    #[test]
    fn t016_link_is_bidirectional() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["a", "b"]);
        add_link(&mut mgr, "a", "b");
        assert!(mgr.route("b", "a").is_ok());
    }
    #[test]
    fn t017_link_count_after_multiple() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["a", "b", "c"]);
        add_link(&mut mgr, "a", "b");
        add_link(&mut mgr, "b", "c");
        add_link(&mut mgr, "a", "c");
        assert_eq!(mgr.link_count(), 3);
    }
    #[test]
    fn t018_remove_link_canonical_reverse() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["a", "b"]);
        add_link(&mut mgr, "a", "b");
        assert!(mgr.remove_link("b", "a").is_ok());
        assert_eq!(mgr.link_count(), 0);
    }
    #[test]
    fn t019_add_link_updates_bandwidth() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["a", "b"]);
        add_link(&mut mgr, "a", "b");
        let updated = OverlayLink {
            from_id: "a".into(),
            to_id: "b".into(),
            latency_ms: 5,
            bandwidth_bps: 9_999_999,
            reliability: 0.9,
            is_tunnel: true,
        };
        mgr.add_link(updated).expect("test: add link");
        assert_eq!(mgr.link_count(), 1);
    }
    #[test]
    fn t020_link_properties_stored() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["a", "b"]);
        let link = OverlayLink {
            from_id: "a".into(),
            to_id: "b".into(),
            latency_ms: 42,
            bandwidth_bps: 5_000_000,
            reliability: 0.95,
            is_tunnel: true,
        };
        mgr.add_link(link).expect("test: add link");
        let route = mgr.route("a", "b").expect("test: compute route");
        assert_eq!(route.total_latency_ms, 42);
        assert_eq!(route.bottleneck_bandwidth_bps, 5_000_000);
        assert!((route.path_reliability - 0.95).abs() < 1e-9);
    }
    #[test]
    fn t021_shortest_path_direct() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["a", "b"]);
        add_link(&mut mgr, "a", "b");
        let r = mgr.route("a", "b").expect("test: compute route");
        assert_eq!(r.hops, vec!["a", "b"]);
    }
    #[test]
    fn t022_shortest_path_two_hops() {
        let mgr = three_node_line();
        let r = mgr.route("a", "c").expect("test: compute route");
        assert_eq!(r.hops, vec!["a", "b", "c"]);
        assert_eq!(r.hop_count(), 2);
    }
    #[test]
    fn t023_shortest_path_no_path() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["a", "b"]);
        let r = mgr.route("a", "b");
        assert!(matches!(r, Err(OverlayError::NoPathExists { .. })));
    }
    #[test]
    fn t024_shortest_path_missing_src() {
        let mgr = make_manager();
        let r = mgr.route("ghost", "b");
        assert!(matches!(r, Err(OverlayError::NodeNotFound(_))));
    }
    #[test]
    fn t025_shortest_path_missing_dst() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["a"]);
        let r = mgr.route("a", "ghost");
        assert!(matches!(r, Err(OverlayError::NodeNotFound(_))));
    }
    #[test]
    fn t026_shortest_path_self() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["a"]);
        let r = mgr.route("a", "a").expect("test: compute route");
        assert_eq!(r.hops, vec!["a"]);
        assert_eq!(r.hop_count(), 0);
    }
    #[test]
    fn t027_max_bandwidth_prefers_high_bw() {
        let mut mgr = OverlayNetworkManager::new(OverlayConfig {
            routing_policy: RoutingPolicy::MaxBandwidth,
            ..Default::default()
        });
        add_nodes(&mut mgr, &["a", "b", "c"]);
        mgr.add_link(OverlayLink {
            from_id: "a".into(),
            to_id: "b".into(),
            latency_ms: 5,
            bandwidth_bps: 1_000_000_000,
            reliability: 1.0,
            is_tunnel: false,
        })
        .expect("test: add link");
        mgr.add_link(OverlayLink {
            from_id: "b".into(),
            to_id: "c".into(),
            latency_ms: 5,
            bandwidth_bps: 1_000_000_000,
            reliability: 1.0,
            is_tunnel: false,
        })
        .expect("test: add link");
        mgr.add_link(OverlayLink {
            from_id: "a".into(),
            to_id: "c".into(),
            latency_ms: 1,
            bandwidth_bps: 100,
            reliability: 1.0,
            is_tunnel: false,
        })
        .expect("test: add link");
        let r = mgr.route("a", "c").expect("test: compute route");
        assert_eq!(r.bottleneck_bandwidth_bps, 1_000_000_000);
    }
    #[test]
    fn t028_max_bandwidth_no_path() {
        let mut mgr = OverlayNetworkManager::new(OverlayConfig {
            routing_policy: RoutingPolicy::MaxBandwidth,
            ..Default::default()
        });
        add_nodes(&mut mgr, &["a", "b"]);
        let r = mgr.route("a", "b");
        assert!(matches!(r, Err(OverlayError::NoPathExists { .. })));
    }
    #[test]
    fn t029_max_bandwidth_single_hop() {
        let mut mgr = OverlayNetworkManager::new(OverlayConfig {
            routing_policy: RoutingPolicy::MaxBandwidth,
            ..Default::default()
        });
        add_nodes(&mut mgr, &["a", "b"]);
        mgr.add_link(OverlayLink::new("a", "b", 10, 999_000))
            .expect("test: add link");
        let r = mgr.route("a", "b").expect("test: compute route");
        assert_eq!(r.bottleneck_bandwidth_bps, 999_000);
    }
    #[test]
    fn t030_max_bandwidth_bottleneck_on_multihop() {
        let mut mgr = OverlayNetworkManager::new(OverlayConfig {
            routing_policy: RoutingPolicy::MaxBandwidth,
            ..Default::default()
        });
        add_nodes(&mut mgr, &["a", "b", "c"]);
        mgr.add_link(OverlayLink {
            from_id: "a".into(),
            to_id: "b".into(),
            latency_ms: 5,
            bandwidth_bps: 500,
            reliability: 1.0,
            is_tunnel: false,
        })
        .expect("test: add link");
        mgr.add_link(OverlayLink {
            from_id: "b".into(),
            to_id: "c".into(),
            latency_ms: 5,
            bandwidth_bps: 1000,
            reliability: 1.0,
            is_tunnel: false,
        })
        .expect("test: add link");
        let r = mgr.route("a", "c").expect("test: compute route");
        assert_eq!(r.bottleneck_bandwidth_bps, 500);
    }
    #[test]
    fn t031_max_reliability_avoids_bad_link() {
        let mut mgr = OverlayNetworkManager::new(OverlayConfig {
            routing_policy: RoutingPolicy::MaxReliability,
            ..Default::default()
        });
        add_nodes(&mut mgr, &["a", "b", "c"]);
        mgr.add_link(OverlayLink {
            from_id: "a".into(),
            to_id: "b".into(),
            latency_ms: 1,
            bandwidth_bps: 1000,
            reliability: 0.5,
            is_tunnel: false,
        })
        .expect("test: add link");
        mgr.add_link(OverlayLink {
            from_id: "b".into(),
            to_id: "c".into(),
            latency_ms: 1,
            bandwidth_bps: 1000,
            reliability: 0.5,
            is_tunnel: false,
        })
        .expect("test: add link");
        mgr.add_link(OverlayLink {
            from_id: "a".into(),
            to_id: "c".into(),
            latency_ms: 100,
            bandwidth_bps: 1000,
            reliability: 0.99,
            is_tunnel: false,
        })
        .expect("test: add link");
        let r = mgr.route("a", "c").expect("test: compute route");
        assert_eq!(r.hops, vec!["a", "c"]);
        assert!((r.path_reliability - 0.99).abs() < 1e-9);
    }
    #[test]
    fn t032_max_reliability_full_path() {
        let mut mgr = OverlayNetworkManager::new(OverlayConfig {
            routing_policy: RoutingPolicy::MaxReliability,
            ..Default::default()
        });
        add_nodes(&mut mgr, &["a", "b"]);
        mgr.add_link(OverlayLink {
            from_id: "a".into(),
            to_id: "b".into(),
            latency_ms: 1,
            bandwidth_bps: 1000,
            reliability: 1.0,
            is_tunnel: false,
        })
        .expect("test: add link");
        let r = mgr.route("a", "b").expect("test: compute route");
        assert!((r.path_reliability - 1.0).abs() < 1e-9);
    }
    #[test]
    fn t033_max_reliability_product_of_links() {
        let mut mgr = OverlayNetworkManager::new(OverlayConfig {
            routing_policy: RoutingPolicy::MaxReliability,
            ..Default::default()
        });
        add_nodes(&mut mgr, &["a", "b", "c"]);
        mgr.add_link(OverlayLink {
            from_id: "a".into(),
            to_id: "b".into(),
            latency_ms: 1,
            bandwidth_bps: 1000,
            reliability: 0.9,
            is_tunnel: false,
        })
        .expect("test: add link");
        mgr.add_link(OverlayLink {
            from_id: "b".into(),
            to_id: "c".into(),
            latency_ms: 1,
            bandwidth_bps: 1000,
            reliability: 0.8,
            is_tunnel: false,
        })
        .expect("test: add link");
        let r = mgr.route("a", "c").expect("test: compute route");
        assert!((r.path_reliability - 0.72).abs() < 1e-9);
    }
    #[test]
    fn t034_max_reliability_no_path() {
        let mut mgr = OverlayNetworkManager::new(OverlayConfig {
            routing_policy: RoutingPolicy::MaxReliability,
            ..Default::default()
        });
        add_nodes(&mut mgr, &["a", "b"]);
        assert!(matches!(
            mgr.route("a", "b"),
            Err(OverlayError::NoPathExists { .. })
        ));
    }
    #[test]
    fn t035_load_balanced_prefers_low_load() {
        let mut mgr = OverlayNetworkManager::new(OverlayConfig {
            routing_policy: RoutingPolicy::LoadBalanced,
            ..Default::default()
        });
        let mut b = OverlayNode::new("b", "", "", "r", 1000);
        b.load = 900;
        let mut d = OverlayNode::new("d", "", "", "r", 1000);
        d.load = 10;
        mgr.add_node(OverlayNode::new("a", "", "", "r", 1000))
            .expect("test: add node");
        mgr.add_node(b).expect("test: add node");
        mgr.add_node(d).expect("test: add node");
        mgr.add_node(OverlayNode::new("c", "", "", "r", 1000))
            .expect("test: add node");
        add_link(&mut mgr, "a", "b");
        add_link(&mut mgr, "b", "c");
        add_link(&mut mgr, "a", "d");
        add_link(&mut mgr, "d", "c");
        let r = mgr.route("a", "c").expect("test: compute route");
        assert!(
            r.hops.contains(&"d".to_owned()),
            "expected low-load path through d"
        );
    }
    #[test]
    fn t036_load_balanced_direct_when_no_alternative() {
        let mut mgr = OverlayNetworkManager::new(OverlayConfig {
            routing_policy: RoutingPolicy::LoadBalanced,
            ..Default::default()
        });
        add_nodes(&mut mgr, &["a", "b"]);
        add_link(&mut mgr, "a", "b");
        let r = mgr.route("a", "b").expect("test: compute route");
        assert_eq!(r.hops, vec!["a", "b"]);
    }
    #[test]
    fn t037_update_load_affects_routing() {
        let mut mgr = OverlayNetworkManager::new(OverlayConfig {
            routing_policy: RoutingPolicy::LoadBalanced,
            ..Default::default()
        });
        add_nodes(&mut mgr, &["a", "b", "c", "d"]);
        add_link(&mut mgr, "a", "b");
        add_link(&mut mgr, "b", "d");
        add_link(&mut mgr, "a", "c");
        add_link(&mut mgr, "c", "d");
        mgr.update_load("b", 900).expect("test: update load");
        let r = mgr.route("a", "d").expect("test: compute route");
        assert!(r.hops.contains(&"c".to_owned()));
    }
    #[test]
    fn t038_geo_prefers_same_region() {
        let mut mgr = OverlayNetworkManager::new(OverlayConfig {
            routing_policy: RoutingPolicy::GeographicProximity,
            ..Default::default()
        });
        mgr.add_node(OverlayNode::new("a", "", "", "eu", 1000))
            .expect("test: add node");
        mgr.add_node(OverlayNode::new("b", "", "", "eu", 1000))
            .expect("test: add node");
        mgr.add_node(OverlayNode::new("c", "", "", "us", 1000))
            .expect("test: add node");
        mgr.add_node(OverlayNode::new("d", "", "", "eu", 1000))
            .expect("test: add node");
        add_link(&mut mgr, "a", "b");
        add_link(&mut mgr, "b", "d");
        add_link(&mut mgr, "a", "c");
        add_link(&mut mgr, "c", "d");
        let r = mgr.route("a", "d").expect("test: compute route");
        assert!(
            r.hops.contains(&"b".to_owned()),
            "expected same-region path a-b-d"
        );
    }
    #[test]
    fn t039_geo_direct_cross_region_if_no_alt() {
        let mut mgr = OverlayNetworkManager::new(OverlayConfig {
            routing_policy: RoutingPolicy::GeographicProximity,
            ..Default::default()
        });
        mgr.add_node(OverlayNode::new("a", "", "", "eu", 1000))
            .expect("test: add node");
        mgr.add_node(OverlayNode::new("b", "", "", "us", 1000))
            .expect("test: add node");
        add_link(&mut mgr, "a", "b");
        let r = mgr.route("a", "b").expect("test: compute route");
        assert_eq!(r.hops, vec!["a", "b"]);
    }
    #[test]
    fn t040_update_load_unknown_node_err() {
        let mut mgr = make_manager();
        assert!(matches!(
            mgr.update_load("ghost", 50),
            Err(OverlayError::NodeNotFound(_))
        ));
    }
    #[test]
    fn t041_apply_full_mesh() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["a", "b", "c"]);
        let links = mgr
            .apply_topology(OverlayTopology::FullMesh)
            .expect("test: apply topology");
        assert_eq!(links.len(), 3);
        assert_eq!(mgr.link_count(), 3);
    }
    #[test]
    fn t042_apply_ring() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["a", "b", "c", "d"]);
        let links = mgr
            .apply_topology(OverlayTopology::Ring)
            .expect("test: apply topology");
        assert_eq!(links.len(), 4);
    }
    #[test]
    fn t043_apply_star() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["hub", "s1", "s2", "s3"]);
        let links = mgr
            .apply_topology(OverlayTopology::Star {
                center_id: "hub".into(),
            })
            .expect("test: apply topology");
        assert_eq!(links.len(), 3);
        for l in &links {
            assert!(l.from_id == "hub" || l.to_id == "hub");
        }
    }
    #[test]
    fn t044_apply_star_missing_centre() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["a", "b"]);
        let r = mgr.apply_topology(OverlayTopology::Star {
            center_id: "ghost".into(),
        });
        assert!(matches!(r, Err(OverlayError::TopologyError(_))));
    }
    #[test]
    fn t045_apply_hypercube_2d() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["n0", "n1", "n2", "n3"]);
        let links = mgr
            .apply_topology(OverlayTopology::Hypercube(2))
            .expect("test: apply topology");
        assert_eq!(links.len(), 4);
    }
    #[test]
    fn t046_apply_hypercube_insufficient_nodes() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["n0", "n1", "n2"]);
        let r = mgr.apply_topology(OverlayTopology::Hypercube(2));
        assert!(matches!(r, Err(OverlayError::TopologyError(_))));
    }
    #[test]
    fn t047_apply_custom_no_links() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["a", "b"]);
        let links = mgr
            .apply_topology(OverlayTopology::Custom)
            .expect("test: apply topology");
        assert!(links.is_empty());
        assert_eq!(mgr.link_count(), 0);
    }
    #[test]
    fn t048_all_routes_k1() {
        let mgr = three_node_line();
        let routes = mgr
            .all_routes("a", "c", 1)
            .expect("test: compute all routes");
        assert_eq!(routes.len(), 1);
        assert_eq!(routes[0].hops, vec!["a", "b", "c"]);
    }
    #[test]
    fn t049_all_routes_k0_empty() {
        let mgr = three_node_line();
        let routes = mgr
            .all_routes("a", "c", 0)
            .expect("test: compute all routes");
        assert!(routes.is_empty());
    }
    #[test]
    fn t050_all_routes_k_greater_than_available() {
        let mgr = three_node_line();
        let routes = mgr
            .all_routes("a", "c", 5)
            .expect("test: compute all routes");
        assert!(!routes.is_empty());
        assert!(routes.len() <= 2);
    }
    #[test]
    fn t051_all_routes_two_paths() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["a", "b", "c", "d"]);
        add_link(&mut mgr, "a", "b");
        add_link(&mut mgr, "b", "d");
        add_link(&mut mgr, "a", "c");
        add_link(&mut mgr, "c", "d");
        let routes = mgr
            .all_routes("a", "d", 2)
            .expect("test: compute all routes");
        assert_eq!(routes.len(), 2);
        for r in &routes {
            assert_eq!(r.hop_count(), 2);
        }
    }
    #[test]
    fn t052_all_routes_no_path() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["a", "b"]);
        let r = mgr.all_routes("a", "b", 3);
        assert!(matches!(r, Err(OverlayError::NoPathExists { .. })));
    }
    #[test]
    fn t053_single_component() {
        let mgr = three_node_line();
        let comps = mgr.connected_components();
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].len(), 3);
    }
    #[test]
    fn t054_two_components() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["a", "b", "c", "d"]);
        add_link(&mut mgr, "a", "b");
        add_link(&mut mgr, "c", "d");
        let comps = mgr.connected_components();
        assert_eq!(comps.len(), 2);
    }
    #[test]
    fn t055_isolated_nodes_each_own_component() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["x", "y", "z"]);
        let comps = mgr.connected_components();
        assert_eq!(comps.len(), 3);
    }
    #[test]
    fn t056_empty_graph_components() {
        let mgr = make_manager();
        let comps = mgr.connected_components();
        assert!(comps.is_empty());
    }
    #[test]
    fn t057_stats_empty() {
        let mgr = make_manager();
        let s = mgr.stats();
        assert_eq!(s.node_count, 0);
        assert_eq!(s.link_count, 0);
        assert!((s.connectivity - 0.0).abs() < 1e-9);
    }
    #[test]
    fn t058_stats_single_node() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["a"]);
        let s = mgr.stats();
        assert_eq!(s.node_count, 1);
        assert!((s.connectivity - 1.0).abs() < 1e-9);
    }
    #[test]
    fn t059_stats_fully_connected_line() {
        let mgr = three_node_line();
        let s = mgr.stats();
        assert_eq!(s.node_count, 3);
        assert_eq!(s.link_count, 2);
        assert!((s.connectivity - 1.0).abs() < 1e-9);
        assert_eq!(s.network_diameter, 2);
    }
    #[test]
    fn t060_stats_disconnected_graph() {
        let mut mgr = make_manager();
        add_nodes(&mut mgr, &["a", "b", "c", "d"]);
        add_link(&mut mgr, "a", "b");
        let s = mgr.stats();
        assert!((s.connectivity - (2.0 / 12.0)).abs() < 1e-9);
    }
    #[test]
    fn t061_xorshift64_non_zero() {
        let mut state = 1u64;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
        assert_ne!(state, 1);
    }
    #[test]
    fn t062_xorshift64_sequence() {
        let mut s = 12345u64;
        let a = xorshift64(&mut s);
        let b = xorshift64(&mut s);
        assert_ne!(a, b);
    }
    #[test]
    fn t063_fnv1a_64_deterministic() {
        let h1 = fnv1a_64(b"hello");
        let h2 = fnv1a_64(b"hello");
        let h3 = fnv1a_64(b"world");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }
}
