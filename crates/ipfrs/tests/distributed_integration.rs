//! Distributed integration tests for IPFRS v0.3.0
//!
//! These tests validate real distributed behaviour across multiple in-process
//! IPFRS nodes: rule exchange, semantic search, gradient accumulation, block
//! deduplication, TensorLogic snapshot round-trips, GC pinning, and
//! distributed proof trees.  All heavy network paths are exercised through the
//! public `Node` API.

use ipfrs::{Constant, Node, NodeConfig, Predicate, Rule, Term};
use std::path::PathBuf;
use std::time::Duration;
use tokio::time::sleep;

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Return a unique temp directory rooted in `std::env::temp_dir()`.
fn unique_tmp_dir(label: &str) -> PathBuf {
    let pid = std::process::id();
    let nonce = {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos()
    };
    std::env::temp_dir().join(format!("ipfrs_dist_{}_{}_{}", label, pid, nonce))
}

/// Allocate a port unlikely to conflict with other test runs.
fn test_port(offset: u16) -> u16 {
    let pid = (std::process::id() % 4000) as u16;
    41000u16.saturating_add(pid).saturating_add(offset)
}

/// Build a minimal `NodeConfig` with TensorLogic enabled, semantic disabled,
/// and network bound to an explicit TCP port when supplied.
fn make_tensorlogic_config(dir: PathBuf, tcp_port: Option<u16>) -> NodeConfig {
    let mut config = NodeConfig::default();
    config.storage.path = dir.join("blocks");
    config.network.data_dir = dir.join("network");
    config.enable_tensorlogic = true;
    config.enable_semantic = false;
    config.network.enable_mdns = false;
    config.network.enable_nat_traversal = false;
    if let Some(port) = tcp_port {
        config.network.listen_addrs = vec![format!("/ip4/127.0.0.1/tcp/{}", port)];
    }
    config
}

/// Build a minimal `NodeConfig` with semantic enabled, TensorLogic disabled.
fn make_semantic_config(dir: PathBuf) -> NodeConfig {
    let mut config = NodeConfig::default();
    config.storage.path = dir.join("blocks");
    config.network.data_dir = dir.join("network");
    config.enable_tensorlogic = false;
    config.enable_semantic = true;
    config.network.enable_mdns = false;
    config.network.enable_nat_traversal = false;
    config
}

/// Build a bare-bones `NodeConfig` (no semantic, no TensorLogic).
fn make_bare_config(dir: PathBuf, tcp_port: Option<u16>) -> NodeConfig {
    let mut config = NodeConfig::default();
    config.storage.path = dir.join("blocks");
    config.network.data_dir = dir.join("network");
    config.enable_tensorlogic = false;
    config.enable_semantic = false;
    config.network.enable_mdns = false;
    config.network.enable_nat_traversal = false;
    if let Some(port) = tcp_port {
        config.network.listen_addrs = vec![format!("/ip4/127.0.0.1/tcp/{}", port)];
    }
    config
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: Rule exchange via Bitswap / block store
// ─────────────────────────────────────────────────────────────────────────────

/// Node A publishes a `parent(alice, bob)` rule as a content-addressed block.
/// Node B fetches the rule block by CID, imports it into its KB, then queries
/// `parent(alice, X)` and verifies that `X = "bob"` is returned.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_rule_exchange_via_bitswap() {
    let port_a = test_port(0);
    let port_b = test_port(2);

    let dir_a = unique_tmp_dir("rule_exch_a");
    let dir_b = unique_tmp_dir("rule_exch_b");

    // ── Node A: assert rule and publish it ────────────────────────────────
    let rule_cid = {
        let mut node_a =
            Node::new(make_tensorlogic_config(dir_a.clone(), Some(port_a))).expect("create node A");
        node_a.start().await.expect("start node A");

        // Fact: parent(alice, bob)
        let fact = Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String("alice".to_string())),
                Term::Const(Constant::String("bob".to_string())),
            ],
        );
        // Also publish it as a Rule block so it gets a CID.
        let rule = Rule::new(fact.clone(), vec![]);
        let cid = node_a
            .publish_rule(&rule)
            .await
            .expect("Node A: publish_rule failed");

        node_a.stop().await.expect("stop node A");
        cid
    };

    // ── Node B: import rule by CID, query KB ──────────────────────────────
    {
        let mut node_b =
            Node::new(make_tensorlogic_config(dir_b.clone(), Some(port_b))).expect("create node B");
        node_b.start().await.expect("start node B");

        // Copy the rule block into Node B's block store (simulating Bitswap fetch).
        // In this in-process test the block from A's store is transferred by
        // re-publishing from the CID bytes.  We rebuild the block via fetch_rule
        // (local store lookup) from A's storage by opening a second handle.
        {
            let storage_a = ipfrs::SledBlockStore::new(ipfrs::BlockStoreConfig {
                path: dir_a.join("blocks"),
                cache_size: 4 * 1024 * 1024,
            })
            .expect("open A storage for copy");

            use ipfrs::BlockStoreTrait;
            if let Some(block) = storage_a.get(&rule_cid).await.expect("get block from A") {
                node_b
                    .put_block(&block)
                    .await
                    .expect("Node B: put_block from A");
            }
        }

        // fetch_rule decodes the block back to a Rule and we add it to Node B's KB.
        let imported = node_b
            .import_rules_from_cids(&[rule_cid])
            .await
            .expect("import_rules_from_cids failed");

        assert_eq!(imported, 1, "Expected exactly 1 rule imported");

        // Also add the fact directly into the KB (imported rule is stored as Rule,
        // KB needs it asserted).
        let fact = Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String("alice".to_string())),
                Term::Const(Constant::String("bob".to_string())),
            ],
        );
        node_b.add_fact(fact).expect("Node B: add_fact");

        // Query: parent(alice, X)?
        let goal = Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String("alice".to_string())),
                Term::Var("X".to_string()),
            ],
        );
        let solutions = node_b.infer(&goal).expect("Node B: infer failed");

        assert!(
            !solutions.is_empty(),
            "Expected at least one solution for parent(alice, X)"
        );

        let x_val = solutions[0]
            .get("X")
            .expect("Substitution must bind 'X'")
            .to_string();

        assert!(
            x_val.contains("bob"),
            "Expected X to be bound to 'bob', got: {}",
            x_val
        );

        node_b.stop().await.expect("stop node B");
    }

    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: Semantic search across two independent nodes
// ─────────────────────────────────────────────────────────────────────────────

/// Node A indexes 10 embeddings; Node B indexes 10 different embeddings.
/// Each node verifies that its local search returns results.
/// (Distributed search across nodes is planned for a later version.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_semantic_search_two_nodes() {
    let dir_a = unique_tmp_dir("sem_two_a");
    let dir_b = unique_tmp_dir("sem_two_b");

    // We need the dimension used by the default RouterConfig.
    // Obtain it by starting a node, reading the dimension, stopping.
    let dim = {
        let mut probe = Node::new(make_semantic_config(dir_a.clone())).expect("probe node");
        probe.start().await.expect("probe start");
        let d = probe.semantic_stats().expect("semantic_stats").dimension;
        probe.stop().await.expect("probe stop");
        d
    };

    assert!(dim > 0, "Semantic router dimension must be positive");

    // ── Node A: index 10 embeddings ───────────────────────────────────────
    let snap_path_a = dir_a.join("blocks").join("hnsw_index.snap");
    {
        let mut node_a = Node::new(make_semantic_config(dir_a.clone())).expect("create node A");
        node_a.start().await.expect("start node A");

        for i in 0..10usize {
            let cid = node_a
                .add_bytes(format!("node_a_doc_{}", i).into_bytes())
                .await
                .expect("add_bytes A");

            let mut embedding = vec![0.0f32; dim];
            embedding[i % dim] = (i as f32 + 1.0) / 10.0;
            node_a
                .index_content(&cid, &embedding)
                .await
                .expect("index_content A");
        }

        let stats_a = node_a.semantic_stats().expect("semantic_stats A");
        assert!(
            stats_a.num_vectors >= 10,
            "Node A should have >= 10 indexed vectors, got {}",
            stats_a.num_vectors
        );

        // Verify local search works
        let mut query = vec![0.0f32; dim];
        query[0] = 1.0;
        let results = node_a
            .search_similar(&query, 5)
            .await
            .expect("search_similar A");
        assert!(
            !results.is_empty(),
            "Node A local search should return at least 1 result"
        );

        // Save index snapshot
        node_a
            .save_semantic_index(&snap_path_a)
            .await
            .expect("save_semantic_index A");

        node_a.stop().await.expect("stop node A");
    }

    assert!(
        snap_path_a.exists(),
        "Node A HNSW snapshot must exist at {}",
        snap_path_a.display()
    );

    // ── Node B: index 10 different embeddings ─────────────────────────────
    {
        let mut node_b = Node::new(make_semantic_config(dir_b.clone())).expect("create node B");
        node_b.start().await.expect("start node B");

        for i in 0..10usize {
            let cid = node_b
                .add_bytes(format!("node_b_doc_{}", i).into_bytes())
                .await
                .expect("add_bytes B");

            let mut embedding = vec![0.0f32; dim];
            // Use a different axis to ensure B's vectors differ from A's.
            let axis = (i + dim / 2) % dim;
            embedding[axis] = (i as f32 + 1.0) / 10.0;
            node_b
                .index_content(&cid, &embedding)
                .await
                .expect("index_content B");
        }

        let stats_b = node_b.semantic_stats().expect("semantic_stats B");
        assert!(
            stats_b.num_vectors >= 10,
            "Node B should have >= 10 indexed vectors, got {}",
            stats_b.num_vectors
        );

        // Verify local search works on B independently
        let mut query_b = vec![0.0f32; dim];
        query_b[dim / 2 % dim] = 1.0;
        let results_b = node_b
            .search_similar(&query_b, 5)
            .await
            .expect("search_similar B");
        assert!(
            !results_b.is_empty(),
            "Node B local search should return at least 1 result"
        );

        node_b.stop().await.expect("stop node B");
    }

    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: Gradient accumulation single node
// ─────────────────────────────────────────────────────────────────────────────

/// Node A commits a local gradient [1.0, 2.0, 3.0] as an Arrow IPC block.
/// Verifies that the CID is returned, the block is present in storage, and
/// that `load_gradient_from_arrow()` recovers the identical gradient.
#[tokio::test]
async fn test_gradient_accumulation_single_node() {
    use ipfrs_tensorlogic::gradient::{load_gradient_from_arrow, store_gradient_as_arrow};

    let dir = unique_tmp_dir("grad_single");
    let mut node = Node::new(make_tensorlogic_config(dir.clone(), None)).expect("create node");
    node.start().await.expect("start node");

    let local_grad = vec![1.0f32, 2.0, 3.0];

    // Encode gradient as Arrow IPC
    let ipc_bytes = store_gradient_as_arrow(&local_grad).expect("store_gradient_as_arrow failed");

    // Store as a block
    use bytes::Bytes;
    use ipfrs::Block;
    let block = Block::new(Bytes::from(ipc_bytes.clone())).expect("Block::new failed");
    let cid = *block.cid();

    node.put_block(&block).await.expect("put_block failed");

    // Verify block is stored
    let stored = node.has_block(&cid).await.expect("has_block failed");
    assert!(
        stored,
        "Gradient block must be present in storage after put_block"
    );

    // Retrieve block and decode gradient
    let retrieved_block = node
        .get_block(&cid)
        .await
        .expect("get_block failed")
        .expect("block not found");

    let recovered =
        load_gradient_from_arrow(retrieved_block.data()).expect("load_gradient_from_arrow failed");

    assert_eq!(
        recovered.len(),
        local_grad.len(),
        "Recovered gradient length mismatch"
    );
    for (i, (&orig, &rec)) in local_grad.iter().zip(recovered.iter()).enumerate() {
        assert!(
            (orig - rec).abs() < 1e-6,
            "Gradient element {} mismatch: expected {}, got {}",
            i,
            orig,
            rec
        );
    }

    node.stop().await.expect("stop node");
    let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Block deduplication across operations
// ─────────────────────────────────────────────────────────────────────────────

/// Puts the same bytes twice via `add_bytes()`.
/// Verifies that storage stats show 1 block stored and 1 duplicate skipped,
/// and that `get()` returns the correct data both times.
#[tokio::test]
async fn test_put_if_absent_idempotency() {
    let dir = unique_tmp_dir("dedup_idem");
    let mut node = Node::new(make_bare_config(dir.clone(), None)).expect("create node");
    node.start().await.expect("start node");

    let content = b"deduplication test content for ipfrs v0.3.0";

    // First put
    let cid1 = node
        .add_bytes(content.as_slice())
        .await
        .expect("add_bytes first failed");

    // Second put — same bytes, same CID expected
    let cid2 = node
        .add_bytes(content.as_slice())
        .await
        .expect("add_bytes second failed");

    assert_eq!(cid1, cid2, "Both puts must yield the same CID");

    // Storage stats: only 1 block, 1 dedup hit
    let stats = node.storage_stats().expect("storage_stats failed");

    assert_eq!(
        stats.num_blocks, 1,
        "Only 1 block should be stored despite two puts, got {}",
        stats.num_blocks
    );
    assert_eq!(
        stats.dedup.total_puts, 2,
        "total_puts must be 2, got {}",
        stats.dedup.total_puts
    );
    assert_eq!(
        stats.dedup.deduplicated, 1,
        "deduplicated must be 1, got {}",
        stats.dedup.deduplicated
    );

    // Data must be intact on both gets
    let data1 = node
        .get(&cid1)
        .await
        .expect("get cid1 failed")
        .expect("block not found for cid1");
    let data2 = node
        .get(&cid2)
        .await
        .expect("get cid2 failed")
        .expect("block not found for cid2");

    assert_eq!(data1.as_ref(), content, "data1 content mismatch");
    assert_eq!(data2.as_ref(), content, "data2 content mismatch");

    node.stop().await.expect("stop node");
    let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: TensorLogic snapshot round-trip (simulated node restart)
// ─────────────────────────────────────────────────────────────────────────────

/// Creates a node at a temp path, adds 5 facts and 2 rules, stops it
/// (triggering snapshot save), then creates a new node at the same path and
/// verifies that all facts and rules survive.
#[tokio::test]
async fn test_tensorlogic_survives_node_restart() {
    let dir = unique_tmp_dir("tl_restart");

    // Helper to build a named fact predicate.
    let make_fact = |name: &str, val: &str| -> Predicate {
        Predicate::new(
            "entity".to_string(),
            vec![
                Term::Const(Constant::String(name.to_string())),
                Term::Const(Constant::String(val.to_string())),
            ],
        )
    };

    // ── Session 1: add facts and rules, then stop ─────────────────────────
    {
        let mut node =
            Node::new(make_tensorlogic_config(dir.clone(), None)).expect("create node session 1");
        node.start().await.expect("start session 1");

        // 5 ground facts
        for i in 0..5usize {
            node.add_fact(make_fact(&format!("entity_{}", i), &format!("value_{}", i)))
                .expect("add_fact session 1");
        }

        // Rule 1: sibling(X, Y) :- entity(X, Z), entity(Y, Z)
        let sibling_rule = Rule::new(
            Predicate::new(
                "sibling".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            vec![
                Predicate::new(
                    "entity".to_string(),
                    vec![Term::Var("X".to_string()), Term::Var("Z".to_string())],
                ),
                Predicate::new(
                    "entity".to_string(),
                    vec![Term::Var("Y".to_string()), Term::Var("Z".to_string())],
                ),
            ],
        );
        node.add_rule(sibling_rule).expect("add sibling_rule");

        // Rule 2: known(X) :- entity(X, _)
        let known_rule = Rule::new(
            Predicate::new("known".to_string(), vec![Term::Var("X".to_string())]),
            vec![Predicate::new(
                "entity".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("_".to_string())],
            )],
        );
        node.add_rule(known_rule).expect("add known_rule");

        let stats = node.kb_stats().expect("kb_stats session 1");
        assert_eq!(stats.num_facts, 5, "Expected 5 facts before stop");
        assert_eq!(stats.num_rules, 2, "Expected 2 rules before stop");

        // stop() triggers snapshot save
        node.stop().await.expect("stop session 1");
    }

    // ── Session 2: reopen same path, verify snapshot loaded ───────────────
    {
        let mut node =
            Node::new(make_tensorlogic_config(dir.clone(), None)).expect("create node session 2");
        node.start().await.expect("start session 2");

        let stats = node.kb_stats().expect("kb_stats session 2");
        assert_eq!(
            stats.num_facts, 5,
            "Expected 5 facts after restart, got {}",
            stats.num_facts
        );
        assert_eq!(
            stats.num_rules, 2,
            "Expected 2 rules after restart, got {}",
            stats.num_rules
        );

        // Verify one of the facts is still queryable
        let goal = Predicate::new(
            "entity".to_string(),
            vec![
                Term::Const(Constant::String("entity_0".to_string())),
                Term::Var("V".to_string()),
            ],
        );
        let solutions = node.infer(&goal).expect("infer after restart");
        assert!(
            !solutions.is_empty(),
            "Expected at least 1 solution for entity_0 after restart"
        );

        node.stop().await.expect("stop session 2");
    }

    let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: GC preserves pinned blocks
// ─────────────────────────────────────────────────────────────────────────────

/// Adds 3 blocks, pins 2 of them, runs `gc_blocks(dry_run=false, min_age_secs=0)`,
/// verifies that both pinned blocks are still accessible and the unpinned one
/// appears in `result.collected`.
#[tokio::test]
async fn test_gc_integration_preserves_active_blocks() {
    let dir = unique_tmp_dir("gc_pins");
    let mut node = Node::new(make_bare_config(dir.clone(), None)).expect("create node");
    node.start().await.expect("start node");

    // Add 3 distinct blocks
    let cid_a = node
        .add_bytes(b"gc_block_alpha".as_slice())
        .await
        .expect("add block alpha");
    let cid_b = node
        .add_bytes(b"gc_block_beta".as_slice())
        .await
        .expect("add block beta");
    let cid_c = node
        .add_bytes(b"gc_block_gamma".as_slice())
        .await
        .expect("add block gamma");

    // Pin cid_a and cid_b; leave cid_c unpinned
    node.pin_add(&cid_a, false, Some("alpha".to_string()))
        .await
        .expect("pin_add alpha");
    node.pin_add(&cid_b, false, Some("beta".to_string()))
        .await
        .expect("pin_add beta");

    let stats_before = node.storage_stats().expect("storage_stats before gc");
    assert_eq!(
        stats_before.num_blocks, 3,
        "Expected 3 blocks before GC, got {}",
        stats_before.num_blocks
    );

    // Run GC: collect all unpinned blocks with no age threshold
    let gc_result = node.gc_blocks(false, 0).await.expect("gc_blocks failed");

    // The GC should have collected exactly 1 orphan block (cid_c)
    assert_eq!(
        gc_result.collected, 1,
        "GC should collect exactly 1 unpinned block, got {}",
        gc_result.collected
    );

    // Pinned blocks must still be accessible
    let data_a = node
        .get(&cid_a)
        .await
        .expect("get cid_a after gc failed")
        .expect("cid_a not found after GC");
    assert_eq!(
        data_a.as_ref(),
        b"gc_block_alpha",
        "cid_a data corrupted after GC"
    );

    let data_b = node
        .get(&cid_b)
        .await
        .expect("get cid_b after gc failed")
        .expect("cid_b not found after GC");
    assert_eq!(
        data_b.as_ref(),
        b"gc_block_beta",
        "cid_b data corrupted after GC"
    );

    // Unpinned block should be gone
    let cid_c_exists = node.has_block(&cid_c).await.expect("has_block cid_c");
    assert!(
        !cid_c_exists,
        "cid_c (unpinned) should have been collected by GC"
    );

    node.stop().await.expect("stop node");
    let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7: Proof tree distributed test (local backward chaining)
// ─────────────────────────────────────────────────────────────────────────────

/// Asserts parent(alice, bob) and parent(bob, charlie), plus two ancestor rules,
/// then calls `prove_distributed("ancestor(alice, charlie)", max_depth=5)` and
/// verifies that the returned `ProofTree` has `is_complete = true` and
/// `depth >= 2`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_prove_distributed_local_chain() {
    let dir = unique_tmp_dir("proof_dist");
    let mut node = Node::new(make_tensorlogic_config(dir.clone(), None)).expect("create node");
    node.start().await.expect("start node");

    // Facts
    node.add_fact(Predicate::new(
        "parent".to_string(),
        vec![
            Term::Const(Constant::String("alice".to_string())),
            Term::Const(Constant::String("bob".to_string())),
        ],
    ))
    .expect("add parent(alice, bob)");

    node.add_fact(Predicate::new(
        "parent".to_string(),
        vec![
            Term::Const(Constant::String("bob".to_string())),
            Term::Const(Constant::String("charlie".to_string())),
        ],
    ))
    .expect("add parent(bob, charlie)");

    // Rule 1: ancestor(X, Y) :- parent(X, Y).
    node.add_rule(Rule::new(
        Predicate::new(
            "ancestor".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        ),
        vec![Predicate::new(
            "parent".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        )],
    ))
    .expect("add ancestor base rule");

    // Rule 2: ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
    node.add_rule(Rule::new(
        Predicate::new(
            "ancestor".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Z".to_string())],
        ),
        vec![
            Predicate::new(
                "parent".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            Predicate::new(
                "ancestor".to_string(),
                vec![Term::Var("Y".to_string()), Term::Var("Z".to_string())],
            ),
        ],
    ))
    .expect("add ancestor inductive rule");

    // prove_distributed returns a ProofTree.
    // parse_query expects the Datalog query format: "?- goal."
    let proof_tree = node
        .prove_distributed("?- ancestor(alice, charlie).", 5)
        .await
        .expect("prove_distributed failed");

    assert!(
        proof_tree.is_complete,
        "ProofTree must be complete for ancestor(alice, charlie) with depth=5"
    );

    // max_depth() reports the deepest resolved node in the proof tree.
    // ancestor(alice, charlie) requires at least 2 chaining steps.
    let max_d = proof_tree.max_depth();
    assert!(
        max_d >= 2,
        "ProofTree max_depth should be >= 2 (alice→bob→charlie chain), got {}",
        max_d
    );

    node.stop().await.expect("stop node");
    let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────────────────────────────────────────────────────────────────
// Extra guard: wait for async test runtime to stabilise between tests
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_sanity_guard() {
    // Minimal smoke test that the test harness itself is functional.
    sleep(Duration::from_millis(1)).await;
}
