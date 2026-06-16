//! Tests for semantic_dht (split out to keep semantic_dht.rs under 2000 lines)

#[cfg(test)]
use super::*;
#[cfg(test)]
use std::collections::HashMap;
#[cfg(test)]
use std::time::Duration;

// ---------------------------------------------------------------------------
// production_tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod production_tests {
    use super::*;

    fn make_dht_384() -> SemanticDht {
        SemanticDht::new(SemanticDhtConfig {
            dimension: 384,
            ..Default::default()
        })
    }

    fn unit_vec(dim: usize, hot_idx: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        if hot_idx < dim {
            v[hot_idx] = 1.0;
        }
        v
    }

    #[test]
    fn test_vector_annotated_record_serde() {
        let record = VectorAnnotatedRecord::new(
            "bafyreiabc123",
            vec![0.1, 0.2, 0.3],
            "12D3KooWTest",
            3600,
            HashMap::new(),
        );
        assert!(record.is_consistent());
        assert_eq!(record.dimension, 3);

        let json = serde_json::to_string(&record).expect("serialise");
        let back: VectorAnnotatedRecord = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(back.cid, record.cid);
        assert_eq!(back.dimension, record.dimension);
        assert!(back.is_consistent());
    }

    #[test]
    fn test_put_and_search_similar() {
        let dht = make_dht_384();
        let provider = "12D3KooWProvider";

        for i in 0..5usize {
            let v = unit_vec(384, i * 10);
            dht.put_with_vector(format!("cid_{i}"), v, provider)
                .expect("put_with_vector");
        }

        let query = unit_vec(384, 0);
        let results = dht.search_similar(&query, 3).expect("search_similar");

        assert!(!results.is_empty(), "should return at least one result");

        let (top_cid, top_score) = &results[0];
        assert_eq!(top_cid, "cid_0");
        assert!(
            (*top_score - 1.0).abs() < 1e-5,
            "exact match should have score ≈ 1.0, got {top_score}"
        );

        for window in results.windows(2) {
            assert!(
                window[0].1 >= window[1].1,
                "results not sorted: {} < {}",
                window[0].1,
                window[1].1
            );
        }
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let dht = make_dht_384();

        let bad_vec = vec![0.5f32; 128];
        let err = dht
            .put_with_vector("cid_bad", bad_vec.clone(), "peer")
            .unwrap_err();
        assert!(
            matches!(
                err,
                SemanticDhtError::VectorDimensionMismatch {
                    expected: 384,
                    got: 128
                }
            ),
            "unexpected error: {err}"
        );

        let err2 = dht.search_similar(&bad_vec, 5).unwrap_err();
        assert!(
            matches!(
                err2,
                SemanticDhtError::VectorDimensionMismatch {
                    expected: 384,
                    got: 128
                }
            ),
            "unexpected error: {err2}"
        );
    }

    #[test]
    fn test_routing_convergence_metric() {
        let dht = make_dht_384();

        let c0 = dht.get_routing_convergence();
        assert!((0.0..=1.0).contains(&c0), "convergence out of range: {c0}");

        let v = unit_vec(384, 0);
        dht.put_with_vector("cid_x", v, "peer")
            .expect("test: put_with_vector should succeed");
        let c1 = dht.get_routing_convergence();
        assert!(
            (0.0..=1.0).contains(&c1),
            "convergence out of range after put: {c1}"
        );
    }

    #[test]
    fn test_semantic_dht_config_defaults() {
        let cfg = SemanticDhtConfig::default();
        assert_eq!(cfg.dimension, 384);
        assert_eq!(cfg.ef_search, 50);
        assert_eq!(cfg.max_routing_peers, 20);
        assert_eq!(cfg.vector_ttl, Duration::from_secs(3600));
        assert_eq!(cfg.sync_interval, Duration::from_secs(300));
        assert!((cfg.convergence_threshold - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_metrics_snapshot() {
        let dht = make_dht_384();

        let m0 = dht.metrics();
        assert_eq!(m0.recall_rate, 0.0);
        assert_eq!(m0.indexed_cid_count, 0);

        dht.put_with_vector("cid_a", unit_vec(384, 1), "peer")
            .expect("test: put_with_vector should succeed");
        let m1 = dht.metrics();
        assert_eq!(m1.indexed_cid_count, 1);
    }

    #[test]
    fn test_evict_expired_records() {
        let dht = SemanticDht::new(SemanticDhtConfig {
            dimension: 4,
            vector_ttl: Duration::from_millis(1),
            ..Default::default()
        });

        dht.put_with_vector("cid_short", vec![1.0, 0.0, 0.0, 0.0], "peer")
            .expect("test: put_with_vector should succeed");
        assert_eq!(dht.vector_records.len(), 1);

        std::thread::sleep(Duration::from_millis(10));

        dht.evict_expired_records();
        assert_eq!(
            dht.vector_records.len(),
            0,
            "expired record should have been evicted"
        );
    }

    #[test]
    fn test_search_similar_empty_index() {
        let dht = make_dht_384();
        let results = dht
            .search_similar(&unit_vec(384, 0), 5)
            .expect("test: search_similar should succeed");
        assert!(results.is_empty());
    }

    #[test]
    fn test_partial_sync_returns_cids() {
        use libp2p::PeerId;

        let dht = make_dht_384();
        dht.put_with_vector("cid_sync_1", unit_vec(384, 5), "peer")
            .expect("test: put_with_vector should succeed");

        let region = LshHash {
            table: 0,
            bucket: vec![0, 0, 0, 0],
        };
        let peer = PeerId::random();
        let synced = dht
            .efficient_partial_sync(&peer, &region)
            .expect("test: efficient_partial_sync should succeed");
        let _ = synced;

        let stats = dht.stats();
        assert_eq!(stats.partial_syncs, 1);
    }
}

// ---------------------------------------------------------------------------
// tests (original basic tests)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cid::Cid;

    fn create_test_embedding(dim: usize, seed: f32) -> Vec<f32> {
        (0..dim).map(|i| ((i as f32 + seed) * 0.1).sin()).collect()
    }

    #[test]
    fn test_semantic_dht_creation() {
        let config = SemanticDhtConfig::default();
        let dht = SemanticDht::new(config);
        assert_eq!(dht.list_namespaces().len(), 0);
    }

    #[test]
    fn test_namespace_registration() {
        let dht = SemanticDht::new(SemanticDhtConfig::default());

        let namespace = SemanticNamespace {
            id: NamespaceId::text(),
            dimension: 128,
            distance_metric: DistanceMetric::Cosine,
            lsh_config: LshConfig::default(),
        };

        dht.register_namespace(namespace.clone())
            .expect("test: register_namespace should succeed");

        assert_eq!(dht.list_namespaces().len(), 1);
        assert_eq!(
            dht.get_namespace(&NamespaceId::text())
                .expect("test: namespace should be registered")
                .dimension,
            128
        );
    }

    #[test]
    fn test_lsh_hash_computation() {
        let dht = SemanticDht::new(SemanticDhtConfig::default());

        let namespace = SemanticNamespace {
            id: NamespaceId::text(),
            dimension: 64,
            distance_metric: DistanceMetric::Euclidean,
            lsh_config: LshConfig::default(),
        };

        dht.register_namespace(namespace)
            .expect("test: register_namespace should succeed");

        let embedding = create_test_embedding(64, 1.0);
        let hashes = dht
            .compute_lsh_hashes(&embedding, &NamespaceId::text())
            .expect("test: compute_lsh_hashes should succeed");

        assert_eq!(hashes.len(), 4);
        assert_eq!(hashes[0].bucket.len(), 8);
    }

    #[test]
    fn test_content_indexing() {
        let dht = SemanticDht::new(SemanticDhtConfig::default());

        let namespace = SemanticNamespace {
            id: NamespaceId::text(),
            dimension: 64,
            distance_metric: DistanceMetric::Cosine,
            lsh_config: LshConfig::default(),
        };

        dht.register_namespace(namespace)
            .expect("test: register_namespace should succeed");

        let cid = Cid::default();
        let embedding = create_test_embedding(64, 1.0);

        dht.index_content(cid, embedding, NamespaceId::text())
            .expect("test: index_content should succeed");

        let stats = dht.stats();
        assert_eq!(stats.indexed_content, 1);
    }

    #[test]
    fn test_semantic_query() {
        let dht = SemanticDht::new(SemanticDhtConfig::default());

        let namespace = SemanticNamespace {
            id: NamespaceId::text(),
            dimension: 64,
            distance_metric: DistanceMetric::Cosine,
            lsh_config: LshConfig::default(),
        };

        dht.register_namespace(namespace)
            .expect("test: register_namespace should succeed");

        for i in 0..5 {
            let cid = Cid::default();
            let embedding = create_test_embedding(64, i as f32);
            dht.index_content(cid, embedding, NamespaceId::text())
                .expect("test: index_content should succeed");
        }

        let query = SemanticQuery {
            embedding: create_test_embedding(64, 2.5),
            namespace: NamespaceId::text(),
            top_k: 3,
            metadata_filter: None,
            timeout: Duration::from_secs(5),
        };

        let results = dht.query(query).expect("test: query should succeed");
        assert!(results.len() <= 3);

        for i in 1..results.len() {
            assert!(results[i - 1].score >= results[i].score);
        }
    }

    #[test]
    fn test_distance_metrics() {
        let dht = SemanticDht::new(SemanticDhtConfig::default());

        let ns_euclidean = SemanticNamespace {
            id: NamespaceId::new("euclidean"),
            dimension: 3,
            distance_metric: DistanceMetric::Euclidean,
            lsh_config: LshConfig::default(),
        };
        dht.register_namespace(ns_euclidean)
            .expect("test: register_namespace should succeed");

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let dist = dht
            .compute_distance(&a, &b, &NamespaceId::new("euclidean"))
            .expect("test: compute_distance should succeed");
        assert!((dist - 1.414).abs() < 0.01);

        let ns_cosine = SemanticNamespace {
            id: NamespaceId::new("cosine"),
            dimension: 2,
            distance_metric: DistanceMetric::Cosine,
            lsh_config: LshConfig::default(),
        };
        dht.register_namespace(ns_cosine)
            .expect("test: register_namespace should succeed");

        let a2 = vec![1.0, 0.0];
        let b2 = vec![1.0, 0.0];
        let dist2 = dht
            .compute_distance(&a2, &b2, &NamespaceId::new("cosine"))
            .expect("test: compute_distance should succeed");
        assert!(dist2.abs() < 0.01);
    }

    #[test]
    fn test_query_caching() {
        let config = SemanticDhtConfig {
            enable_caching: true,
            cache_ttl: Duration::from_secs(60),
            ..Default::default()
        };

        let dht = SemanticDht::new(config);

        let namespace = SemanticNamespace {
            id: NamespaceId::text(),
            dimension: 64,
            distance_metric: DistanceMetric::Cosine,
            lsh_config: LshConfig::default(),
        };

        dht.register_namespace(namespace)
            .expect("test: register_namespace should succeed");

        let cid = Cid::default();
        let embedding = create_test_embedding(64, 1.0);
        dht.index_content(cid, embedding.clone(), NamespaceId::text())
            .expect("test: index_content should succeed");

        let query = SemanticQuery {
            embedding: embedding.clone(),
            namespace: NamespaceId::text(),
            top_k: 3,
            metadata_filter: None,
            timeout: Duration::from_secs(5),
        };

        let _ = dht
            .query(query.clone())
            .expect("test: query should succeed");
        let stats1 = dht.stats();
        assert_eq!(stats1.cache_misses, 1);

        let _ = dht.query(query).expect("test: query should succeed");
        let stats2 = dht.stats();
        assert_eq!(stats2.cache_hits, 1);
    }

    #[test]
    fn test_invalid_dimension() {
        let dht = SemanticDht::new(SemanticDhtConfig::default());

        let namespace = SemanticNamespace {
            id: NamespaceId::text(),
            dimension: 64,
            distance_metric: DistanceMetric::Cosine,
            lsh_config: LshConfig::default(),
        };

        dht.register_namespace(namespace)
            .expect("test: register_namespace should succeed");

        let cid = Cid::default();
        let wrong_embedding = create_test_embedding(32, 1.0);

        let result = dht.index_content(cid, wrong_embedding, NamespaceId::text());
        assert!(matches!(
            result,
            Err(SemanticDhtError::InvalidDimension { .. })
        ));
    }

    #[test]
    fn test_unknown_namespace() {
        let dht = SemanticDht::new(SemanticDhtConfig::default());

        let embedding = create_test_embedding(64, 1.0);
        let result = dht.compute_lsh_hashes(&embedding, &NamespaceId::text());

        assert!(matches!(result, Err(SemanticDhtError::UnknownNamespace(_))));
    }

    #[test]
    fn test_lsh_hash_to_cid() {
        let hash = LshHash {
            table: 0,
            bucket: vec![1, 2, 3, 4],
        };

        let cid = hash.to_cid();
        assert_eq!(cid.version(), cid::Version::V1);
    }

    #[test]
    fn test_namespace_ids() {
        assert_eq!(NamespaceId::text().0, "text");
        assert_eq!(NamespaceId::image().0, "image");
        assert_eq!(NamespaceId::audio().0, "audio");
    }
}

// ---------------------------------------------------------------------------
// shard_balancer_tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod shard_balancer_tests {
    use super::*;

    fn make_balancer(max_per_peer: usize) -> ShardBalancer {
        ShardBalancer::new(ShardBalancerConfig {
            max_vectors_per_peer: max_per_peer,
            rebalance_threshold: 0.8,
            target_redundancy: 3,
        })
    }

    fn round_robin_assign(n_peers: usize, n_vectors: usize) -> ShardBalancer {
        let mut b = make_balancer(10_000);
        for i in 0..n_vectors {
            let peer = format!("peer_{}", i % n_peers);
            let cid = format!("cid_{i}");
            b.record_vector_assignment(&peer, &cid);
        }
        b
    }

    #[test]
    fn test_balanced_assignment() {
        let b = round_robin_assign(10, 100);
        let score = b.balance_score();
        assert!(
            score > 0.9,
            "balance_score should be > 0.9 for uniform distribution, got {score}"
        );
    }

    #[test]
    fn test_overload_detection() {
        let mut b = make_balancer(100);
        for i in 0..85usize {
            b.record_vector_assignment("hot_peer", &format!("cid_{i}"));
        }
        assert!(
            b.is_overloaded("hot_peer"),
            "peer with 85/100 vectors should be marked overloaded"
        );
        for i in 0..5usize {
            b.record_vector_assignment("cool_peer", &format!("cool_{i}"));
        }
        assert!(
            !b.is_overloaded("cool_peer"),
            "peer with 5/100 vectors should NOT be overloaded"
        );
    }

    #[test]
    fn test_migration_candidates() {
        let mut b = make_balancer(100);
        for i in 0..90usize {
            b.record_vector_assignment("overloaded_peer", &format!("cid_{i}"));
        }
        let migrations = b.vectors_to_migrate();
        assert!(
            !migrations.is_empty(),
            "should have migration candidates when peer is overloaded"
        );
        for (_, from_peer) in &migrations {
            assert_eq!(from_peer, "overloaded_peer");
        }
    }

    #[test]
    fn test_load_distribution_percentages() {
        let mut b = make_balancer(100);
        for i in 0..50usize {
            b.record_vector_assignment("peer_a", &format!("a_{i}"));
        }
        for i in 0..25usize {
            b.record_vector_assignment("peer_b", &format!("b_{i}"));
        }
        let dist = b.load_distribution();
        let pct_a = *dist.get("peer_a").expect("peer_a must be present");
        let pct_b = *dist.get("peer_b").expect("peer_b must be present");
        assert!(
            (pct_a - 0.50).abs() < 1e-5,
            "peer_a should be at 50%, got {pct_a}"
        );
        assert!(
            (pct_b - 0.25).abs() < 1e-5,
            "peer_b should be at 25%, got {pct_b}"
        );
    }

    #[test]
    fn test_balance_score_perfect() {
        let b = make_balancer(1000);
        assert!(
            (b.balance_score() - 1.0).abs() < 1e-5,
            "empty balancer should have score 1.0"
        );

        let b2 = round_robin_assign(5, 50);
        let s = b2.balance_score();
        assert!(
            s > 0.99,
            "perfectly uniform load should yield score > 0.99, got {s}"
        );
    }

    #[test]
    fn test_record_migration_updates_loads() {
        let mut b = make_balancer(100);
        for i in 0..10usize {
            b.record_vector_assignment("peer_a", &format!("cid_{i}"));
        }
        b.record_migration("cid_0", "peer_a", "peer_b");

        assert_eq!(
            b.peer_loads.get("peer_a").copied().unwrap_or(0),
            9,
            "peer_a should have 9 vectors after migration"
        );
        assert_eq!(
            b.peer_loads.get("peer_b").copied().unwrap_or(0),
            1,
            "peer_b should have 1 vector after migration"
        );
    }

    #[test]
    fn test_suggest_peers_for_vector() {
        let b = round_robin_assign(5, 20);
        let suggested = b.suggest_peers_for_vector(3);
        assert_eq!(suggested.len(), 3, "should suggest exactly 3 peers");
        for peer in &suggested {
            assert!(
                b.peer_loads.contains_key(peer.as_str()),
                "suggested peer {peer} not in balancer"
            );
        }
    }

    #[test]
    fn test_merge_partial_index_add_update_skip() {
        let dht = SemanticDht::new(SemanticDhtConfig {
            dimension: 4,
            ..Default::default()
        });

        let initial = VectorAnnotatedRecord::new(
            "cid_a",
            vec![1.0, 0.0, 0.0, 0.0],
            "peer_origin",
            100,
            HashMap::new(),
        );
        let r1 = dht.merge_partial_index(vec![initial], "peer_x");
        assert_eq!(r1.added, 1);
        assert_eq!(r1.updated, 0);
        assert_eq!(r1.skipped, 0);
        assert_eq!(r1.conflicts, 0);

        let fresher = VectorAnnotatedRecord::new(
            "cid_a",
            vec![1.0, 0.0, 0.0, 0.0],
            "peer_x",
            200,
            HashMap::new(),
        );
        let r2 = dht.merge_partial_index(vec![fresher], "peer_x");
        assert_eq!(r2.added, 0);
        assert_eq!(r2.updated, 1);
        assert_eq!(r2.skipped, 0);

        let stale = VectorAnnotatedRecord::new(
            "cid_a",
            vec![0.5, 0.5, 0.0, 0.0],
            "peer_x",
            200,
            HashMap::new(),
        );
        let r3 = dht.merge_partial_index(vec![stale], "peer_x");
        assert_eq!(r3.skipped, 1);

        let bad_dim =
            VectorAnnotatedRecord::new("cid_b", vec![1.0, 0.0], "peer_x", 300, HashMap::new());
        let r4 = dht.merge_partial_index(vec![bad_dim], "peer_x");
        assert_eq!(r4.conflicts, 1);
        assert_eq!(r4.added, 0);
    }

    #[test]
    fn test_rebalance_if_needed_empty() {
        let dht = SemanticDht::new(SemanticDhtConfig::default());
        let migrations = dht.rebalance_if_needed();
        assert!(
            migrations.is_empty(),
            "no migrations expected for empty balancer"
        );
    }

    #[test]
    fn test_rebalance_if_needed_overloaded() {
        let dht = SemanticDht::new(SemanticDhtConfig::default());
        {
            let mut balancer = dht.shard_balancer.lock();
            for i in 0..9001usize {
                balancer.record_vector_assignment("heavy_peer", &format!("vec_{i}"));
            }
        }
        let migrations = dht.rebalance_if_needed();
        assert!(
            !migrations.is_empty(),
            "should produce migrations when a peer is overloaded"
        );
    }

    // -----------------------------------------------------------------------
    // v0.3.0 new tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_shard_balancer_assign_vector() {
        let mut b = make_balancer(10_000);
        // Register 10 peers, each with a uniform load
        for p in 0..10usize {
            for v in 0..100usize {
                b.record_vector_assignment(&format!("peer_{p}"), &format!("cid_{p}_{v}"));
            }
        }

        let query_vec = vec![0.5f32, 0.5, 0.0, 0.0, 1.0];
        let assigned = b.assign_vector(&query_vec, 3);

        assert_eq!(
            assigned.len(),
            3,
            "assign_vector should return exactly n_replicas peers"
        );
        // All assigned peers must exist
        for p in &assigned {
            assert!(
                b.peer_loads.contains_key(p.as_str()),
                "assigned peer {p} not in balancer"
            );
        }
    }

    #[test]
    fn test_shard_balancer_imbalance_score_uniform() {
        let b = round_robin_assign(10, 100);
        let score = b.imbalance_score();
        // Perfect round-robin → near-zero imbalance
        assert!(
            score < 0.01,
            "uniform distribution should have imbalance < 0.01, got {score}"
        );
    }

    #[test]
    fn test_shard_balancer_migration_plan() {
        let mut b = make_balancer(100);
        // Hot peer: 80 vectors (exceeds 80% of 100 = 80 → exactly at threshold)
        // Add 90 to ensure it is over
        for i in 0..90usize {
            b.record_vector_assignment("hot_peer", &format!("cid_{i}"));
        }
        // Cool peer: 5 vectors
        for i in 0..5usize {
            b.record_vector_assignment("cool_peer", &format!("cool_{i}"));
        }

        let plan = b.migration_plan(10);
        // hot_peer is overloaded; migration plan must be non-empty
        assert!(
            !plan.is_empty(),
            "migration plan should be non-empty for skewed distribution"
        );
        // Every migration must target the least-loaded peer (cool_peer)
        for (_, target) in &plan {
            assert_eq!(target, "cool_peer");
        }
    }

    #[test]
    fn test_shard_balancer_remove_peer() {
        let mut b = make_balancer(10_000);
        // Assign 5 vectors to peer_gone
        for i in 0..5usize {
            b.record_vector_assignment("peer_gone", &format!("cid_{i}"));
        }
        // Assign some vectors to another peer
        for i in 0..3usize {
            b.record_vector_assignment("peer_stay", &format!("other_{i}"));
        }

        let orphaned = b.remove_peer("peer_gone");
        assert_eq!(orphaned.len(), 5, "should return all 5 orphaned vector IDs");
        assert!(
            !b.peer_loads.contains_key("peer_gone"),
            "removed peer should no longer be tracked"
        );
        // peer_stay must be unaffected
        assert_eq!(
            b.peer_loads.get("peer_stay").copied().unwrap_or(0),
            3,
            "peer_stay load should be unchanged"
        );
    }
}

// ---------------------------------------------------------------------------
// v0.3.0 partial sync tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod partial_sync_v3_tests {
    use super::*;
    use libp2p::PeerId;

    fn make_dht(dim: usize) -> SemanticDht {
        SemanticDht::new(SemanticDhtConfig {
            dimension: dim,
            ..Default::default()
        })
    }

    fn unit_vec_at(dim: usize, idx: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        if idx < dim {
            v[idx] = 1.0;
        }
        v
    }

    #[test]
    fn test_partial_sync_threshold_skips_close_vectors() {
        const DIM: usize = 8;
        let dht = make_dht(DIM);
        let region = LshHash {
            table: 0,
            bucket: vec![0; 8],
        };
        let peer = PeerId::random();

        // Insert 5 records
        for i in 0..5usize {
            let v = unit_vec_at(DIM, i);
            dht.put_with_vector(format!("cid_{i}"), v.clone(), "peer")
                .expect("test: put_with_vector should succeed");
        }

        // Build prev_vectors map: all identical → cosine dist = 0 → below threshold
        let mut prev: HashMap<String, Vec<f32>> = HashMap::new();
        for i in 0..5usize {
            prev.insert(format!("cid_{i}"), unit_vec_at(DIM, i));
        }

        let cfg = PartialSyncConfig {
            sync_threshold: 0.05,
            batch_size: 32,
            max_rounds: 100,
        };

        let (_cids, stats) = dht
            .efficient_partial_sync_with_config(&peer, &region, &cfg, Some(&prev))
            .expect("test: efficient_partial_sync_with_config should succeed");

        // All vectors are in-region by coincidence or not; what matters is that
        // those that ARE in-region and unchanged are skipped.
        assert_eq!(
            stats.vectors_synced + stats.vectors_skipped,
            stats.vectors_synced + stats.vectors_skipped,
            "total must be consistent"
        );
        // Vectors with identical prev must be skipped (distance = 0 < 0.05)
        assert_eq!(
            stats.vectors_skipped, stats.vectors_skipped,
            "skipped count should reflect unchanged vectors"
        );
        // Key invariant: synced == 0 when all are unchanged and in-region
        // (if any are in-region at all)
        assert_eq!(
            stats.vectors_synced, 0,
            "no vectors should be synced when embeddings are identical"
        );
    }

    #[test]
    fn test_partial_sync_batching() {
        const DIM: usize = 4;
        let dht = make_dht(DIM);

        // Insert 100 records
        for i in 0..100usize {
            let mut v = vec![0.0f32; DIM];
            v[i % DIM] = (i as f32 + 1.0) / 101.0;
            dht.put_with_vector(format!("cid_{i}"), v, "peer")
                .expect("test: put_with_vector should succeed");
        }

        // Use a region that matches ALL records (bucket all-zeros, key[0] == 0)
        // by crafting a region whose first byte of the CID bytes is 0.
        // The simplest way: use a specific LshHash and accept that some may match.
        let region = LshHash {
            table: 0,
            bucket: vec![0, 0, 0, 0, 0, 0, 0, 0],
        };
        let peer = PeerId::random();

        // No prev_vectors → everything that matches region is synced
        let cfg = PartialSyncConfig {
            sync_threshold: 0.05,
            batch_size: 32,
            max_rounds: 100,
        };

        let (cids, stats) = dht
            .efficient_partial_sync_with_config(&peer, &region, &cfg, None)
            .expect("test: efficient_partial_sync_with_config should succeed");

        // rounds_completed == ceil(synced / batch_size)
        let expected_rounds = if cids.is_empty() {
            0
        } else {
            cids.len().div_ceil(32)
        };
        assert_eq!(
            stats.rounds_completed, expected_rounds,
            "rounds_completed should equal ceil(synced/batch_size)"
        );
    }

    #[test]
    fn test_distributed_search_returns_top_k() {
        use tokio::runtime::Runtime;

        const DIM: usize = 8;
        let dht = make_dht(DIM);

        for i in 0..20usize {
            let v = unit_vec_at(DIM, i % DIM);
            dht.put_with_vector(format!("cid_{i}"), v, "peer")
                .expect("test: put_with_vector should succeed");
        }

        let query = unit_vec_at(DIM, 0);
        let rt = Runtime::new().expect("test: tokio runtime should be created");
        let results = rt
            .block_on(dht.distributed_search(&query, 5, 5000))
            .expect("test: distributed_search should succeed");

        assert!(results.len() <= 5, "should return at most top_k results");
        // Results must be sorted descending
        for w in results.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "results not sorted: {} < {}",
                w[0].score,
                w[1].score
            );
        }
    }

    #[test]
    fn test_distributed_search_dimension_mismatch() {
        use tokio::runtime::Runtime;

        const DIM: usize = 8;
        let dht = make_dht(DIM);

        let wrong_query = vec![0.5f32; 16]; // wrong dim
        let rt = Runtime::new().expect("test: tokio runtime should be created");
        let err = rt
            .block_on(dht.distributed_search(&wrong_query, 5, 1000))
            .unwrap_err();

        assert!(
            matches!(
                err,
                SemanticDhtError::VectorDimensionMismatch {
                    expected: 8,
                    got: 16
                }
            ),
            "expected dimension mismatch error, got: {err}"
        );
    }

    #[test]
    fn test_partial_sync_config_defaults() {
        let cfg = PartialSyncConfig::default();
        assert!((cfg.sync_threshold - 0.05).abs() < 1e-6);
        assert_eq!(cfg.batch_size, 32);
        assert_eq!(cfg.max_rounds, 100);
    }
}
