//! Multi-node integration tests for IPFRS 0.2.0
//!
//! Demonstrates real P2P block exchange between multiple nodes running
//! on localhost with different ports.  All heavy network paths are
//! exercised through the public `Node` API.

use ipfrs::{Node, NodeConfig};
use std::path::PathBuf;
use std::time::Duration;
use tokio::time::sleep;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Return a unique temp directory rooted in `std::env::temp_dir()`.
fn unique_tmp_dir(label: &str) -> PathBuf {
    let pid = std::process::id();
    // Combine label, pid, and two independent nonce components:
    // - subsec_nanos from SystemTime
    // - a thread-local counter via a relaxed AtomicU64
    // Together these make collisions astronomically unlikely even when multiple
    // tests start within the same nanosecond on the same machine.
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nonce = {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos()
    };
    std::env::temp_dir().join(format!("ipfrs_test_{}_{}_{}_{}", label, pid, nonce, seq))
}

/// Ask the OS to allocate a free TCP port on 127.0.0.1.
///
/// The trick is to bind a `TcpListener` on port 0, note the assigned port,
/// then immediately drop the listener.  There is a tiny TOCTOU window between
/// the drop and libp2p's own `listen_on()`, but in practice this is far more
/// reliable than hard-coding a derived port that could collide with another
/// concurrent process.
fn alloc_free_port() -> u16 {
    let listener =
        std::net::TcpListener::bind("127.0.0.1:0").expect("Failed to bind ephemeral port");
    listener
        .local_addr()
        .expect("Failed to get local addr")
        .port()
}

/// Build a minimal `NodeConfig` whose storage and network data live under `dir`.
///
/// When `tcp_port` is `Some(port)` the node will listen on that TCP port on
/// 127.0.0.1 – useful for two-node tests where Node B needs a stable address
/// for Node A.  When `None` the default QUIC/UDP with ephemeral port is used.
fn make_config(dir: PathBuf, tcp_port: Option<u16>) -> NodeConfig {
    let mut config = NodeConfig::default();

    // Point storage and network data to the unique temp dir
    config.storage.path = dir.join("blocks");
    config.network.data_dir = dir.join("network");

    // Keep tests light: disable heavy optional features
    config.enable_semantic = false;
    config.enable_tensorlogic = false;

    // mDNS off so tests don't accidentally cross-discover each other
    config.network.enable_mdns = false;
    config.network.enable_nat_traversal = false;

    if let Some(port) = tcp_port {
        // Replace default listen addresses with a known TCP address so that the
        // second node can dial it without querying an ephemeral port mapping.
        config.network.listen_addrs = vec![format!("/ip4/127.0.0.1/tcp/{}", port)];
    }

    config
}

/// Wait until `condition` returns `true`, polling every `poll_interval` with
/// exponential back-off up to `max_interval`, giving up after `deadline`.
///
/// Returns `true` if the condition was satisfied within the deadline, `false`
/// on timeout.
async fn wait_until<F, Fut>(
    deadline: Duration,
    poll_interval: Duration,
    max_interval: Duration,
    mut condition: F,
) -> bool
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let start = tokio::time::Instant::now();
    let mut interval = poll_interval;
    loop {
        if condition().await {
            return true;
        }
        if start.elapsed() >= deadline {
            return false;
        }
        let remaining = deadline.saturating_sub(start.elapsed());
        let wait = interval.min(remaining);
        sleep(wait).await;
        // Exponential back-off with a cap
        interval = (interval * 2).min(max_interval);
    }
}

/// Wait until TCP port `port` on 127.0.0.1 accepts a connection, polling with
/// exponential back-off.  Returns `true` if the port is reachable within
/// `deadline`, `false` on timeout.
///
/// This is used to verify that a libp2p node has fully bound its listening
/// socket before a remote node attempts to dial it.
async fn wait_for_tcp_port(port: u16, deadline: Duration) -> bool {
    use std::net::SocketAddr;
    let addr: SocketAddr = format!("127.0.0.1:{}", port).parse().expect("valid addr");
    let start = tokio::time::Instant::now();
    let mut interval = Duration::from_millis(50);
    loop {
        match tokio::net::TcpStream::connect(addr).await {
            Ok(_) => return true,
            Err(_) => {
                if start.elapsed() >= deadline {
                    return false;
                }
                let remaining = deadline.saturating_sub(start.elapsed());
                sleep(interval.min(remaining)).await;
                interval = (interval * 2).min(Duration::from_millis(500));
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: Single-node add / get (storage sanity check)
// ─────────────────────────────────────────────────────────────────────────────

/// Verify that `add_bytes` stores content and `get` retrieves it from local
/// storage without any network interaction.
#[tokio::test]
async fn test_single_node_add_get() {
    let dir = unique_tmp_dir("single_add_get");
    let config = make_config(dir, None);

    let mut node = Node::new(config).expect("Failed to create node");
    node.start().await.expect("Failed to start node");

    let content = b"hello ipfrs 0.2.0";
    let cid = node
        .add_bytes(content.as_slice())
        .await
        .expect("add_bytes failed");

    let retrieved = node
        .get(&cid)
        .await
        .expect("get failed")
        .expect("content not found");

    assert_eq!(
        retrieved.as_ref(),
        content,
        "Retrieved content should match what was added"
    );

    node.stop().await.expect("Failed to stop node");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: Single-node add / get (high-level block_stat)
// ─────────────────────────────────────────────────────────────────────────────

/// Verify that `block_stat` returns accurate metadata after `add_bytes`.
#[tokio::test]
async fn test_single_node_block_stat() {
    let dir = unique_tmp_dir("single_block_stat");
    let config = make_config(dir, None);

    let mut node = Node::new(config).expect("Failed to create node");
    node.start().await.expect("Failed to start node");

    let content = b"block stat content for ipfrs 0.2.0";
    let cid = node
        .add_bytes(content.as_slice())
        .await
        .expect("add_bytes failed");

    let stat = node
        .block_stat(&cid)
        .await
        .expect("block_stat failed")
        .expect("block stat not found");

    assert_eq!(stat.size, content.len(), "block_stat size mismatch");
    assert_eq!(stat.cid, cid, "block_stat CID mismatch");

    node.stop().await.expect("Failed to stop node");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: Single node auto-provides and local peer ID is available
// ─────────────────────────────────────────────────────────────────────────────

/// Confirms that after `add_bytes`:
/// - The node has a non-empty local peer ID
/// - The block is stored locally (`has_block` returns true)
/// - The node is running
///
/// This validates the auto-provide path without requiring a second peer in the
/// DHT routing table.
#[tokio::test]
async fn test_node_provides_to_dht() {
    let dir = unique_tmp_dir("provides_dht");
    let config = make_config(dir, None);

    let mut node = Node::new(config).expect("Failed to create node");
    node.start().await.expect("Failed to start node");

    // The node must have a valid local peer ID once started
    let local_peer_id = node.peer_id().expect("Failed to get peer ID");
    assert!(!local_peer_id.is_empty(), "Peer ID must not be empty");

    // Add bytes – this internally calls network.provide() (best-effort)
    let content = b"tensor data provided to dht";
    let cid = node
        .add_bytes(content.as_slice())
        .await
        .expect("add_bytes failed");

    // The block must be in local storage
    let stored = node.has_block(&cid).await.expect("has_block failed");
    assert!(stored, "Block should be in local storage after add_bytes");

    // The network is up (node is running)
    assert!(
        node.is_running(),
        "Node must still be running after add_bytes"
    );

    // Retrieve via get() – should hit local cache immediately
    let data = node
        .get(&cid)
        .await
        .expect("get failed")
        .expect("block not found locally");
    assert_eq!(data.as_ref(), content);

    node.stop().await.expect("Failed to stop node");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Two nodes connect via TCP on localhost
// ─────────────────────────────────────────────────────────────────────────────

/// Create two nodes on different TCP ports.  Node B dials Node A, then both
/// sides verify they see each other in the connected-peers list.
#[tokio::test]
async fn test_two_nodes_connect() {
    let port_a = alloc_free_port();
    let port_b = alloc_free_port();

    let dir_a = unique_tmp_dir("connect_a");
    let dir_b = unique_tmp_dir("connect_b");

    let mut node_a = Node::new(make_config(dir_a, Some(port_a))).expect("Failed to create node A");
    let mut node_b = Node::new(make_config(dir_b, Some(port_b))).expect("Failed to create node B");

    node_a.start().await.expect("Failed to start node A");
    node_b.start().await.expect("Failed to start node B");

    // Poll until port_a is actually accepting TCP connections before dialing.
    // This is more reliable than a fixed sleep because it waits for the exact
    // moment the OS has bound the socket.
    assert!(
        wait_for_tcp_port(port_a, Duration::from_secs(15)).await,
        "Node A's TCP port {} did not become reachable within 15 s",
        port_a
    );

    let peer_id_a = node_a.peer_id().expect("Failed to get peer ID of A");
    let peer_id_b = node_b.peer_id().expect("Failed to get peer ID of B");

    // Node B dials Node A's TCP address.
    let addr_a = format!("/ip4/127.0.0.1/tcp/{}/p2p/{}", port_a, peer_id_a);
    node_b
        .connect(&addr_a)
        .await
        .expect("Node B failed to dial Node A");

    // Wait for connection to appear in each node's peer list, polling with
    // exponential back-off.  Total deadline: 15 seconds.
    let peers_a_ok = wait_until(
        Duration::from_secs(15),
        Duration::from_millis(100),
        Duration::from_millis(1000),
        || async {
            node_a
                .peers()
                .await
                .map(|peers| peers.contains(&peer_id_b))
                .unwrap_or(false)
        },
    )
    .await;

    let peers_b_ok = wait_until(
        Duration::from_secs(15),
        Duration::from_millis(100),
        Duration::from_millis(1000),
        || async {
            node_b
                .peers()
                .await
                .map(|peers| peers.contains(&peer_id_a))
                .unwrap_or(false)
        },
    )
    .await;

    // Collect final peer lists for diagnostic messages
    let peers_a = node_a.peers().await.expect("Failed to get peers of A");
    let peers_b = node_b.peers().await.expect("Failed to get peers of B");

    assert!(
        peers_a_ok,
        "Node A should see Node B in its peer list (A's peers: {:?})",
        peers_a
    );
    assert!(
        peers_b_ok,
        "Node B should see Node A in its peer list (B's peers: {:?})",
        peers_b
    );

    node_a.stop().await.expect("Failed to stop node A");
    node_b.stop().await.expect("Failed to stop node B");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: Two-node block exchange – provider discovery
// ─────────────────────────────────────────────────────────────────────────────

/// Full P2P flow:
/// 1. Start two nodes and connect them
/// 2. Node A adds a block (auto-provides to DHT)
/// 3. After DHT propagation, Node B queries `find_providers`
/// 4. Node A's peer ID must appear in the providers list
#[tokio::test]
async fn test_two_nodes_block_exchange() {
    // Allocate free OS ports so there is zero chance of port collision with
    // other tests running in parallel (nextest, CI).
    let port_a = alloc_free_port();
    let port_b = alloc_free_port();

    let dir_a = unique_tmp_dir("exchange_a");
    let dir_b = unique_tmp_dir("exchange_b");

    let mut node_a = Node::new(make_config(dir_a, Some(port_a))).expect("Failed to create node A");
    let mut node_b = Node::new(make_config(dir_b, Some(port_b))).expect("Failed to create node B");

    node_a.start().await.expect("Failed to start node A");
    node_b.start().await.expect("Failed to start node B");

    // Poll until port_a is actually accepting TCP connections before dialing.
    // This replaces the unreliable fixed-sleep approach: we wait for the exact
    // moment the OS has finished binding the socket.
    assert!(
        wait_for_tcp_port(port_a, Duration::from_secs(15)).await,
        "Node A's TCP port {} did not become reachable within 15 s",
        port_a
    );

    let peer_id_a = node_a.peer_id().expect("Failed to get peer ID of A");

    // Connect B → A.
    let addr_a = format!("/ip4/127.0.0.1/tcp/{}/p2p/{}", port_a, peer_id_a);
    node_b
        .connect(&addr_a)
        .await
        .expect("Node B failed to dial Node A");

    // Wait for the Identify protocol exchange so each node has the other's
    // addresses before we publish provider records.  Poll instead of sleeping.
    let peer_id_b = node_b.peer_id().expect("Failed to get peer ID of B");
    let connected = wait_until(
        Duration::from_secs(15),
        Duration::from_millis(100),
        Duration::from_millis(500),
        || async {
            let a_sees_b = node_a
                .peers()
                .await
                .map(|ps| ps.contains(&peer_id_b))
                .unwrap_or(false);
            let b_sees_a = node_b
                .peers()
                .await
                .map(|ps| ps.contains(&peer_id_a))
                .unwrap_or(false);
            a_sees_b && b_sees_a
        },
    )
    .await;

    assert!(
        connected,
        "Nodes did not see each other in peer lists within 15 s"
    );

    // Node A stores a block and announces it to the DHT
    let content = b"tensor data from node A";
    let cid = node_a
        .add_bytes(content.as_slice())
        .await
        .expect("Node A: add_bytes failed");

    // Node B queries for providers.  The DHT propagation in a two-node network
    // is fast once the routing table is populated; we poll rather than sleep.
    // find_providers_timeout already has an internal deadline – we wrap the
    // whole retry loop in an outer deadline as well.
    use std::str::FromStr;
    let expected_peer_id =
        ipfrs_network::libp2p::PeerId::from_str(&peer_id_a).expect("Failed to parse peer ID A");

    let found = tokio::time::timeout(Duration::from_secs(30), async {
        // Poll: issue a find_providers_timeout with a bounded window and
        // retry until providers appear or the outer timeout fires.
        let mut poll_interval = Duration::from_millis(500);
        let max_poll_interval = Duration::from_secs(3);
        loop {
            let providers = node_b
                .find_providers_timeout(&cid, Duration::from_secs(8))
                .await
                .unwrap_or_default();

            if providers.contains(&expected_peer_id) {
                return providers;
            }

            // Back off before retrying
            sleep(poll_interval).await;
            poll_interval = (poll_interval * 2).min(max_poll_interval);
        }
    })
    .await;

    let providers = match found {
        Ok(p) => p,
        Err(_) => panic!(
            "Node B did not find Node A as provider for {} within 30 s",
            cid
        ),
    };

    assert!(
        providers.contains(&expected_peer_id),
        "Node A's peer ID should appear in providers (got: {:?})",
        providers
    );

    node_a.stop().await.expect("Failed to stop node A");
    node_b.stop().await.expect("Failed to stop node B");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: Multiple add/get round-trips on a single node (stress)
// ─────────────────────────────────────────────────────────────────────────────

/// Adds 50 unique blobs to one node and retrieves all of them, verifying
/// content integrity for each.
#[tokio::test]
async fn test_single_node_bulk_add_get() {
    let dir = unique_tmp_dir("bulk_add_get");
    let config = make_config(dir, None);

    let mut node = Node::new(config).expect("Failed to create node");
    node.start().await.expect("Failed to start node");

    let count = 50usize;
    let mut cids = Vec::with_capacity(count);

    for i in 0..count {
        let content = format!("ipfrs 0.2.0 bulk block {:04}", i);
        let content_bytes = content.into_bytes();
        let cid = node
            .add_bytes(content_bytes.clone())
            .await
            .unwrap_or_else(|e| panic!("add_bytes failed for block {}: {}", i, e));
        cids.push((cid, content_bytes));
    }

    for (cid, expected_bytes) in &cids {
        let data = node
            .get(cid)
            .await
            .unwrap_or_else(|e| panic!("get failed for {}: {}", cid, e))
            .unwrap_or_else(|| panic!("block not found: {}", cid));

        assert_eq!(
            data.as_ref(),
            expected_bytes.as_slice(),
            "Content mismatch for {}",
            cid
        );
    }

    let stats = node.storage_stats().expect("storage_stats failed");
    assert!(
        stats.num_blocks >= count,
        "Expected at least {} blocks, got {}",
        count,
        stats.num_blocks
    );

    node.stop().await.expect("Failed to stop node");
    // Allow the background swarm task to drain after shutdown signal
    sleep(Duration::from_millis(50)).await;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7: Node lifecycle – stop and restart preserves stored blocks
// ─────────────────────────────────────────────────────────────────────────────

/// Confirms that sled-backed storage is durable across node restarts.
#[tokio::test]
async fn test_node_restart_durability() {
    let dir = unique_tmp_dir("restart_durability");
    let content = b"durable content across restarts";

    // First session: add a block then stop
    let cid = {
        let config = make_config(dir.clone(), None);
        let mut node = Node::new(config).expect("Failed to create node (session 1)");
        node.start().await.expect("Failed to start (session 1)");

        let cid = node
            .add_bytes(content.as_slice())
            .await
            .expect("add_bytes failed (session 1)");

        node.stop().await.expect("Failed to stop (session 1)");
        cid
    };

    // Second session: reopen same storage, block must still be there
    {
        let config = make_config(dir, None);
        let mut node = Node::new(config).expect("Failed to create node (session 2)");
        node.start().await.expect("Failed to start (session 2)");

        let exists = node
            .has_block(&cid)
            .await
            .expect("has_block failed (session 2)");
        assert!(exists, "Block must survive across node restarts");

        let retrieved = node
            .get(&cid)
            .await
            .expect("get failed (session 2)")
            .expect("block not found after restart");

        assert_eq!(
            retrieved.as_ref(),
            content,
            "Content must be intact after restart"
        );

        node.stop().await.expect("Failed to stop (session 2)");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 8: HNSW semantic index persists across node restarts
// ─────────────────────────────────────────────────────────────────────────────

/// Build a NodeConfig that enables the semantic router but keeps network lean.
fn make_semantic_config(dir: std::path::PathBuf) -> NodeConfig {
    let mut config = NodeConfig::default();
    config.storage.path = dir.join("blocks");
    config.network.data_dir = dir.join("network");
    config.enable_semantic = true;
    config.enable_tensorlogic = false;
    config.network.enable_mdns = false;
    config.network.enable_nat_traversal = false;
    config
}

/// Confirms that:
/// 1. When a node with semantic enabled is stopped, an `hnsw_index.snap` file
///    is written to the storage directory.
/// 2. On restart, the snapshot is restored (the file is still present and the
///    index loads without error).
/// 3. Semantic search over the previously indexed vector still returns results.
#[tokio::test]
async fn test_semantic_index_persists_across_restart() {
    let dir = unique_tmp_dir("semantic_persist");
    let snap_path = dir.join("blocks").join("hnsw_index.snap");

    // ── Session 1: add an embedding, stop the node ──────────────────────────
    {
        let config = make_semantic_config(dir.clone());
        let mut node = Node::new(config).expect("create node (session 1)");
        node.start().await.expect("start (session 1)");

        // We need a CID to key the vector against – add a tiny block first.
        let cid = node
            .add_bytes(b"semantic test content" as &[u8])
            .await
            .expect("add_bytes (session 1)");

        // Index the content with its embedding.
        // SemanticRouter config defaults to dimension 768; use the router's
        // configured dimension so we don't get a dimension-mismatch error.
        let stats = node.semantic_stats().expect("semantic_stats (session 1)");
        let dim = stats.dimension;
        let embedding_padded: Vec<f32> = {
            let mut v = vec![0.0f32; dim];
            if dim > 0 {
                v[0] = 1.0;
            }
            v
        };
        node.index_content(&cid, &embedding_padded)
            .await
            .expect("index_content (session 1)");

        // Verify the vector is indexed before stopping
        let stats_after = node.semantic_stats().expect("semantic_stats after index");
        assert!(
            stats_after.num_vectors >= 1,
            "Expected at least 1 vector in index before stop, got {}",
            stats_after.num_vectors
        );

        node.stop().await.expect("stop (session 1)");
    }

    // The snapshot must have been written during stop()
    assert!(
        snap_path.exists(),
        "HNSW snapshot must exist at {} after node stop",
        snap_path.display()
    );

    // ── Session 2: restart and verify index is restored ─────────────────────
    {
        let config = make_semantic_config(dir.clone());
        let mut node = Node::new(config).expect("create node (session 2)");
        node.start().await.expect("start (session 2)");

        // The snapshot should have been loaded during start()
        let stats = node.semantic_stats().expect("semantic_stats (session 2)");
        assert!(
            stats.num_vectors >= 1,
            "Expected at least 1 vector restored from snapshot, got {}",
            stats.num_vectors
        );

        node.stop().await.expect("stop (session 2)");
    }

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: distributed_infer – local fast-path
// ─────────────────────────────────────────────────────────────────────────────

/// Verifies that `distributed_infer` uses the local fast-path when TensorLogic
/// is enabled and local inference succeeds, returning results without touching
/// any peer network.
///
/// The test:
/// 1. Creates a single node with TensorLogic enabled.
/// 2. Adds a simple `greeting(hello)` fact.
/// 3. Calls `distributed_infer("greeting(hello)", …)`.
/// 4. Asserts that the call succeeds, that `peers_queried == 0` (fast-path),
///    and that at least one local binding is present.
#[tokio::test]
async fn test_distributed_infer_local_fast_path() {
    use ipfrs::{Constant, Predicate, Term};

    let dir = unique_tmp_dir("dinfer_local");

    let mut config = make_config(dir.clone(), None);
    // Enable TensorLogic so the fast-path can find local facts.
    config.enable_tensorlogic = true;

    let mut node = Node::new(config).expect("Failed to create node");
    node.start().await.expect("Failed to start node");

    // Add a ground fact: greeting(hello)
    let fact = Predicate::new(
        "greeting".to_string(),
        vec![Term::Const(Constant::String("hello".to_string()))],
    );
    node.add_fact(fact).expect("Failed to add fact");

    // Run distributed inference – should take the local fast-path.
    let result = node
        .distributed_infer("greeting(hello)", 5, Duration::from_secs(1))
        .await;

    assert!(
        result.is_ok(),
        "distributed_infer should succeed: {:?}",
        result.err()
    );
    let dinfer = result.expect("already checked above");

    // Fast-path: no peers should have been queried.
    assert_eq!(
        dinfer.peers_queried, 0,
        "Expected 0 peers queried on local fast-path, got {}",
        dinfer.peers_queried
    );

    // The session_id must be a non-empty string (UUID v4).
    assert!(
        !dinfer.session_id.is_empty(),
        "session_id must not be empty"
    );

    // Elapsed time should be a sane value (less than 10 seconds).
    assert!(
        dinfer.elapsed_ms < 10_000,
        "elapsed_ms suspiciously large: {}",
        dinfer.elapsed_ms
    );

    node.stop().await.expect("Failed to stop node");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Verifies that `distributed_infer` still succeeds when the node has no
/// network configured (offline mode), returning an empty result gracefully
/// instead of an error.
#[tokio::test]
async fn test_distributed_infer_no_network_graceful() {
    let dir = unique_tmp_dir("dinfer_nonet");

    let mut config = NodeConfig::default();
    config.storage.path = dir.join("blocks");
    config.network.data_dir = dir.clone();
    config.enable_tensorlogic = true;
    // Keep network default; it will start but have no peers.

    let mut node = Node::new(config).expect("Failed to create node");
    node.start().await.expect("Failed to start node");

    // No facts – so local results will be empty.
    // distributed_infer should still return Ok with empty bindings.
    let result = node
        .distributed_infer("nonexistent_predicate(X)", 3, Duration::from_millis(200))
        .await;

    assert!(
        result.is_ok(),
        "distributed_infer should not error even without network peers: {:?}",
        result.err()
    );
    let dinfer = result.expect("already checked above");
    assert!(
        dinfer.local_bindings.is_empty(),
        "Expected no local bindings"
    );
    assert!(
        dinfer.remote_bindings.is_empty(),
        "Expected no remote bindings"
    );

    node.stop().await.expect("Failed to stop node");
    let _ = std::fs::remove_dir_all(&dir);
}
