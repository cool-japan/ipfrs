//! Benchmarks for ipfrs-transport
//!
//! Run with: cargo bench
//! For latency percentiles: cargo bench --bench transport_bench -- --verbose

use bytes::Bytes;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ipfrs_core::Cid;
use ipfrs_transport::{
    ConcurrentWantList, EdgeNode, ErasureConfig, LatencyTracker, MemoryTracker, MessageWantEntry,
    MessageWantList, MulticastConfig, MulticastManager, PeerManager, PeerScoringConfig, Priority,
    Session, SessionConfig, SimpleErasureEncoder, StreamRequest, StreamRequestQueue,
    TensorMetadata, ThroughputTracker, Timer, WantListConfig,
};
use multihash::Multihash;
use std::hint::black_box;
use std::time::{Duration, Instant};

/// Create a dummy CID for testing
fn dummy_cid(seed: u64) -> Cid {
    let data = seed.to_le_bytes();
    let hash = Multihash::wrap(0x12, &data).unwrap();
    Cid::new_v1(0x55, hash)
}

/// Benchmark want list operations
fn bench_want_list(c: &mut Criterion) {
    let mut group = c.benchmark_group("want_list");

    // Benchmark adding entries
    group.bench_function("add_100_entries", |b| {
        b.iter(|| {
            let want_list = ConcurrentWantList::new(WantListConfig::default());
            for i in 0..100 {
                want_list.add_simple(black_box(dummy_cid(i)), black_box(100));
            }
        });
    });

    // Benchmark priority updates
    group.bench_function("update_priority", |b| {
        let want_list = ConcurrentWantList::new(WantListConfig::default());
        for i in 0..100 {
            want_list.add_simple(dummy_cid(i), 100);
        }
        let cid = dummy_cid(50);

        b.iter(|| {
            want_list.update_priority(black_box(&cid), black_box(200));
        });
    });

    // Benchmark config creation
    group.bench_function("create_config", |b| {
        b.iter(|| {
            let config = WantListConfig::default();
            black_box(config);
        });
    });

    // Benchmark batch add - individual
    group.bench_function("add_100_individual", |b| {
        b.iter(|| {
            let want_list = ConcurrentWantList::new(WantListConfig::default());
            for i in 0..100 {
                want_list.add_simple(black_box(dummy_cid(i)), black_box(100));
            }
        });
    });

    // Benchmark batch add - using batch operation
    group.bench_function("add_100_batch", |b| {
        b.iter(|| {
            let want_list = ConcurrentWantList::new(WantListConfig::default());
            let cids: Vec<_> = (0..100).map(dummy_cid).collect();
            want_list.add_batch_same_priority(black_box(&cids), black_box(100));
        });
    });

    // Benchmark batch remove - individual
    group.bench_function("remove_100_individual", |b| {
        let want_list = ConcurrentWantList::new(WantListConfig::default());
        let cids: Vec<_> = (0..100).map(dummy_cid).collect();
        for cid in &cids {
            want_list.add_simple(*cid, 100);
        }

        b.iter(|| {
            let temp_list = want_list.clone();
            for cid in &cids {
                temp_list.remove(black_box(cid));
            }
        });
    });

    // Benchmark batch remove - using batch operation
    group.bench_function("remove_100_batch", |b| {
        let want_list = ConcurrentWantList::new(WantListConfig::default());
        let cids: Vec<_> = (0..100).map(dummy_cid).collect();
        for cid in &cids {
            want_list.add_simple(*cid, 100);
        }

        b.iter(|| {
            let temp_list = want_list.clone();
            temp_list.remove_batch(black_box(&cids));
        });
    });

    // Benchmark batch priority update - individual
    group.bench_function("update_100_priorities_individual", |b| {
        let want_list = ConcurrentWantList::new(WantListConfig::default());
        let cids: Vec<_> = (0..100).map(dummy_cid).collect();
        for cid in &cids {
            want_list.add_simple(*cid, 100);
        }

        b.iter(|| {
            for (i, cid) in cids.iter().enumerate() {
                want_list.update_priority(black_box(cid), black_box(200 + i as i32));
            }
        });
    });

    // Benchmark batch priority update - using batch operation
    group.bench_function("update_100_priorities_batch", |b| {
        let want_list = ConcurrentWantList::new(WantListConfig::default());
        let cids: Vec<_> = (0..100).map(dummy_cid).collect();
        for cid in &cids {
            want_list.add_simple(*cid, 100);
        }

        let updates: Vec<_> = cids
            .iter()
            .enumerate()
            .map(|(i, cid)| (*cid, 200 + i as i32))
            .collect();

        b.iter(|| {
            want_list.update_priorities_batch(black_box(&updates));
        });
    });

    group.finish();
}

/// Benchmark peer manager operations
fn bench_peer_manager(c: &mut Criterion) {
    let mut group = c.benchmark_group("peer_manager");

    // Benchmark manager creation
    group.bench_function("create_manager", |b| {
        b.iter(|| {
            let config = PeerScoringConfig::default();
            let manager = PeerManager::new(black_box(config));
            black_box(manager);
        });
    });

    // Benchmark peer selection
    group.bench_function("select_best_peer", |b| {
        let config = PeerScoringConfig::default();
        let mut manager = PeerManager::new(config);

        // Add 50 peers
        for i in 0..50 {
            let peer_id = format!("peer{}", i);
            manager.add_peer(peer_id.clone());
            manager.record_success(&peer_id, 1000, Duration::from_millis(10));
        }

        b.iter(|| {
            black_box(manager.best_peer());
        });
    });

    group.finish();
}

/// Benchmark message serialization
fn bench_messages(c: &mut Criterion) {
    let mut group = c.benchmark_group("messages");

    // Benchmark want list serialization
    for size in [10, 50, 100, 500].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut entries = Vec::new();
            for i in 0..size {
                entries.push(MessageWantEntry {
                    cid: dummy_cid(i as u64),
                    priority: i,
                    send_dont_have: true,
                    cancel: false,
                });
            }
            let want_list = MessageWantList {
                entries,
                full: true,
            };

            b.iter(|| {
                let serialized = oxicode::serde::encode_to_vec(
                    black_box(&want_list),
                    oxicode::config::standard(),
                )
                .unwrap();
                black_box(serialized);
            });
        });
    }

    group.finish();
}

/// Benchmark erasure coding
fn bench_erasure_coding(c: &mut Criterion) {
    let mut group = c.benchmark_group("erasure_coding");

    // Benchmark encoding different data sizes
    for size in [1024, 10240, 102400, 1048576].iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let config = ErasureConfig::new(8, 4).unwrap();
            let encoder = SimpleErasureEncoder::new(config);
            let data = vec![0u8; size];

            b.iter(|| {
                let shards = encoder.encode(black_box(&data)).unwrap();
                black_box(shards);
            });
        });
    }

    // Benchmark decoding
    group.bench_function("decode_1mb", |b| {
        let config = ErasureConfig::new(8, 4).unwrap();
        let encoder = SimpleErasureEncoder::new(config);
        let data = vec![0u8; 1048576];
        let original_size = data.len();
        let shards = encoder.encode(&data).unwrap();

        b.iter(|| {
            let decoded = encoder
                .decode(black_box(&shards), black_box(original_size))
                .unwrap();
            black_box(decoded);
        });
    });

    group.finish();
}

/// Benchmark multicast configuration
fn bench_multicast(c: &mut Criterion) {
    let mut group = c.benchmark_group("multicast");

    // Benchmark config creation
    group.bench_function("create_config", |b| {
        b.iter(|| {
            let config = MulticastConfig::default();
            black_box(config);
        });
    });

    // Benchmark manager creation
    group.bench_function("create_manager", |b| {
        b.iter(|| {
            let config = MulticastConfig::default();
            let manager = MulticastManager::new(config);
            black_box(manager);
        });
    });

    group.finish();
}

/// Benchmark tensor metadata creation
fn bench_tensor_metadata(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_metadata");

    // Benchmark metadata creation
    group.bench_function("create_metadata", |b| {
        b.iter(|| {
            let cid = dummy_cid(1);
            let metadata = TensorMetadata::new(black_box(cid));
            black_box(metadata);
        });
    });

    // Benchmark metadata with dependencies
    group.bench_function("metadata_with_deps", |b| {
        let cid = dummy_cid(1);
        let deps: Vec<Cid> = (0..10).map(dummy_cid).collect();

        b.iter(|| {
            let mut metadata = TensorMetadata::new(black_box(cid));
            metadata.dependencies = black_box(deps.clone());
            metadata.is_critical = true;
            black_box(metadata);
        });
    });

    group.finish();
}

/// Benchmark CID operations
fn bench_cid_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("cid_operations");

    // Benchmark CID creation
    group.bench_function("create_cid", |b| {
        b.iter(|| {
            let cid = dummy_cid(black_box(123));
            black_box(cid);
        });
    });

    // Benchmark CID comparison
    group.bench_function("compare_cids", |b| {
        let cid1 = dummy_cid(1);
        let cid2 = dummy_cid(2);

        b.iter(|| {
            let result = black_box(cid1) == black_box(cid2);
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark latency tracking
fn bench_latency_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_tracking");

    // Benchmark recording latency
    group.bench_function("record_latency", |b| {
        let tracker = LatencyTracker::new();
        b.iter(|| {
            tracker.record(black_box(Duration::from_micros(100)));
        });
    });

    // Benchmark stats calculation
    group.bench_function("calculate_stats", |b| {
        let tracker = LatencyTracker::new();
        for i in 0..1000 {
            tracker.record(Duration::from_micros(i));
        }

        b.iter(|| {
            let stats = tracker.stats();
            black_box(stats);
        });
    });

    // Benchmark timer
    group.bench_function("timer_overhead", |b| {
        let tracker = LatencyTracker::new();
        b.iter(|| {
            let timer = Timer::start();
            // Simulate some work
            black_box(42);
            timer.stop_and_record(&tracker);
        });
    });

    group.finish();
}

/// Benchmark memory tracking
fn bench_memory_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_tracking");

    // Benchmark allocation recording
    group.bench_function("record_allocation", |b| {
        let tracker = MemoryTracker::new();
        b.iter(|| {
            tracker.record_allocation(black_box(1024));
        });
    });

    // Benchmark deallocation recording
    group.bench_function("record_deallocation", |b| {
        let tracker = MemoryTracker::new();
        b.iter(|| {
            tracker.record_deallocation(black_box(1024));
        });
    });

    // Benchmark stats retrieval
    group.bench_function("get_stats", |b| {
        let tracker = MemoryTracker::new();
        tracker.record_allocation(1024);

        b.iter(|| {
            let stats = tracker.stats();
            black_box(stats);
        });
    });

    group.finish();
}

/// Benchmark throughput tracking
fn bench_throughput_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_tracking");

    // Benchmark bytes recording
    group.bench_function("record_bytes", |b| {
        let tracker = ThroughputTracker::new();
        b.iter(|| {
            tracker.record_bytes(black_box(1024));
        });
    });

    // Benchmark throughput calculation
    group.bench_function("calculate_throughput", |b| {
        let tracker = ThroughputTracker::new();
        for _ in 0..1000 {
            tracker.record_bytes(1024);
        }

        b.iter(|| {
            let throughput = tracker.throughput_bps();
            black_box(throughput);
        });
    });

    group.finish();
}

/// Benchmark CDN edge operations
fn bench_cdn_edge(c: &mut Criterion) {
    let mut group = c.benchmark_group("cdn_edge");

    // Benchmark cache put
    group.bench_function("cache_put", |b| {
        let edge = EdgeNode::new();
        let cid = dummy_cid(1);
        let data = bytes::Bytes::from(vec![0u8; 1024]);

        b.iter(|| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                edge.put(black_box(cid), black_box(data.clone())).await.ok();
            });
        });
    });

    // Benchmark cache get
    group.bench_function("cache_get", |b| {
        let edge = EdgeNode::new();
        let cid = dummy_cid(1);
        let data = bytes::Bytes::from(vec![0u8; 1024]);

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            edge.put(cid, data).await.ok();
        });

        b.iter(|| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let result = edge.get(&black_box(cid)).await;
                black_box(result);
            });
        });
    });

    group.finish();
}

/// Benchmark with latency distribution analysis
fn bench_with_latency_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_distribution");

    // Benchmark want list operations with latency tracking
    group.bench_function("want_list_with_tracking", |b| {
        let want_list = ConcurrentWantList::new(WantListConfig::default());
        let tracker = LatencyTracker::new();

        b.iter(|| {
            let timer = Timer::start();
            for i in 0..100 {
                want_list.add_simple(dummy_cid(i), 100);
            }
            timer.stop_and_record(&tracker);
        });

        // Print latency stats after benchmark
        let stats = tracker.stats();
        println!("\nWant List Latency Distribution:");
        println!("{}", stats);
    });

    group.finish();
}

/// Benchmark session management operations
fn bench_session_management(c: &mut Criterion) {
    let mut group = c.benchmark_group("session_management");

    // Benchmark session creation
    group.bench_function("create_session", |b| {
        let config = SessionConfig {
            timeout: Duration::from_secs(60),
            default_priority: Priority::Normal,
            max_concurrent_blocks: 100,
            progress_notifications: false,
        };

        b.iter(|| {
            let session = Session::new(black_box(1), black_box(config.clone()), None);
            black_box(session);
        });
    });

    // Benchmark adding blocks to session
    group.bench_function("add_blocks_100", |b| {
        let config = SessionConfig {
            timeout: Duration::from_secs(60),
            default_priority: Priority::Normal,
            max_concurrent_blocks: 100,
            progress_notifications: false,
        };
        let session = Session::new(1, config, None);
        let cids: Vec<Cid> = (0..100).map(dummy_cid).collect();

        b.iter(|| {
            session.add_blocks(black_box(&cids), None).ok();
        });
    });

    // Benchmark marking blocks as received
    group.bench_function("mark_received", |b| {
        let config = SessionConfig {
            timeout: Duration::from_secs(60),
            default_priority: Priority::Normal,
            max_concurrent_blocks: 100,
            progress_notifications: false,
        };
        let session = Session::new(1, config, None);
        let cids: Vec<Cid> = (0..100).map(dummy_cid).collect();
        session.add_blocks(&cids, None).ok();
        let data = Bytes::from(vec![0u8; 1024]);

        b.iter(|| {
            for cid in &cids {
                session.mark_received(black_box(cid), black_box(&data)).ok();
            }
        });
    });

    // Benchmark session stats retrieval
    group.bench_function("get_stats", |b| {
        let config = SessionConfig {
            timeout: Duration::from_secs(60),
            default_priority: Priority::Normal,
            max_concurrent_blocks: 100,
            progress_notifications: false,
        };
        let session = Session::new(1, config, None);
        let cids: Vec<Cid> = (0..10).map(dummy_cid).collect();
        session.add_blocks(&cids, None).ok();

        b.iter(|| {
            let stats = session.stats();
            black_box(stats);
        });
    });

    group.finish();
}

/// Benchmark stream request queue operations
fn bench_stream_queue(c: &mut Criterion) {
    let mut group = c.benchmark_group("stream_queue");

    // Benchmark queue operations
    for size in [10, 50, 100, 500].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut queue = StreamRequestQueue::new(1000);
                for i in 0..size {
                    let request = StreamRequest {
                        cid: dummy_cid(i as u64),
                        priority: i,
                        deadline: None,
                        queued_at: Instant::now(),
                    };
                    queue.push(black_box(request));
                }
                // Pop all requests
                while let Some(req) = queue.pop() {
                    black_box(req);
                }
            });
        });
    }

    // Benchmark priority-based insertion
    group.bench_function("priority_insertion", |b| {
        let mut queue = StreamRequestQueue::new(1000);
        // Pre-fill with some requests
        for i in 0..50 {
            queue.push(StreamRequest {
                cid: dummy_cid(i),
                priority: (i * 10) as i32,
                deadline: None,
                queued_at: Instant::now(),
            });
        }

        b.iter(|| {
            // Insert a high-priority request that should go near the front
            let request = StreamRequest {
                cid: dummy_cid(999),
                priority: black_box(450), // High priority
                deadline: None,
                queued_at: Instant::now(),
            };
            queue.push(request);
            // Remove it to keep queue size stable
            queue.pop();
        });
    });

    group.finish();
}

/// Benchmark concurrent operations
fn bench_concurrent_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_ops");

    // Benchmark concurrent want list operations
    group.bench_function("concurrent_want_list_adds", |b| {
        let want_list = ConcurrentWantList::new(WantListConfig::default());

        b.iter(|| {
            // Simulate concurrent adds from multiple threads
            let handles: Vec<_> = (0..4)
                .map(|thread_id| {
                    let wl = want_list.clone();
                    std::thread::spawn(move || {
                        for i in 0..25 {
                            let cid = dummy_cid((thread_id * 25 + i) as u64);
                            wl.add_simple(cid, 100);
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().ok();
            }
        });
    });

    group.finish();
}

/// Benchmark utility helper functions
fn bench_utility_helpers(c: &mut Criterion) {
    let mut group = c.benchmark_group("utility_helpers");

    // Benchmark bulk operations vs individual operations
    group.bench_function("bulk_add_100_batch", |b| {
        use ipfrs_transport::bulk_add_wants;

        let want_list = ConcurrentWantList::new(WantListConfig::default());
        let cids: Vec<_> = (0..100).map(dummy_cid).collect();

        b.iter(|| {
            let temp_list = want_list.clone();
            bulk_add_wants(&temp_list, black_box(&cids), black_box(100));
        });
    });

    group.bench_function("bulk_add_100_individual", |b| {
        let want_list = ConcurrentWantList::new(WantListConfig::default());
        let cids: Vec<_> = (0..100).map(dummy_cid).collect();

        b.iter(|| {
            let temp_list = want_list.clone();
            for cid in &cids {
                temp_list.add_simple(black_box(*cid), black_box(100));
            }
        });
    });

    // Benchmark bulk_remove
    group.bench_function("bulk_remove_100", |b| {
        use ipfrs_transport::bulk_remove_wants;

        let want_list = ConcurrentWantList::new(WantListConfig::default());
        let cids: Vec<_> = (0..100).map(dummy_cid).collect();
        for cid in &cids {
            want_list.add_simple(*cid, 100);
        }

        b.iter(|| {
            let temp_list = want_list.clone();
            bulk_remove_wants(&temp_list, black_box(&cids));
        });
    });

    // Benchmark bulk_update_priorities
    group.bench_function("bulk_update_priorities_100", |b| {
        use ipfrs_transport::bulk_update_priorities;

        let want_list = ConcurrentWantList::new(WantListConfig::default());
        let cids: Vec<_> = (0..100).map(dummy_cid).collect();
        for cid in &cids {
            want_list.add_simple(*cid, 100);
        }
        let updates: Vec<_> = cids
            .iter()
            .enumerate()
            .map(|(i, c)| (*c, 200 + i as i32))
            .collect();

        b.iter(|| {
            bulk_update_priorities(&want_list, black_box(&updates));
        });
    });

    // Benchmark all_wants_present
    group.bench_function("all_wants_present_100", |b| {
        use ipfrs_transport::all_wants_present;

        let want_list = ConcurrentWantList::new(WantListConfig::default());
        let cids: Vec<_> = (0..100).map(dummy_cid).collect();
        for cid in &cids {
            want_list.add_simple(*cid, 100);
        }

        b.iter(|| {
            let result = all_wants_present(&want_list, black_box(&cids));
            black_box(result);
        });
    });

    // Benchmark configuration validation
    group.bench_function("validate_want_list_config", |b| {
        use ipfrs_transport::validate_want_list_config;

        let config = WantListConfig::default();

        b.iter(|| {
            let _ = black_box(validate_want_list_config(black_box(&config)));
        });
    });

    group.bench_function("validate_peer_scoring_config", |b| {
        use ipfrs_transport::validate_peer_scoring_config;

        let config = PeerScoringConfig::default();

        b.iter(|| {
            let _ = black_box(validate_peer_scoring_config(black_box(&config)));
        });
    });

    // Benchmark calculate_optimal_concurrency
    group.bench_function("calculate_optimal_concurrency", |b| {
        use ipfrs_transport::calculate_optimal_concurrency;
        use std::time::Duration;

        b.iter(|| {
            let result = calculate_optimal_concurrency(
                black_box(10_000_000),
                black_box(Duration::from_millis(100)),
                black_box(256 * 1024),
            );
            black_box(result);
        });
    });

    // Benchmark preset creation
    group.bench_function("create_balanced_peer_scoring", |b| {
        use ipfrs_transport::create_balanced_peer_scoring;

        b.iter(|| {
            let config = create_balanced_peer_scoring();
            black_box(config);
        });
    });

    group.finish();
}

/// Benchmark configuration preset helpers
fn bench_config_presets(c: &mut Criterion) {
    let mut group = c.benchmark_group("config_presets");

    group.bench_function("create_high_throughput_want_list", |b| {
        use ipfrs_transport::create_high_throughput_want_list;

        b.iter(|| {
            let list = create_high_throughput_want_list();
            black_box(list);
        });
    });

    group.bench_function("create_low_latency_want_list", |b| {
        use ipfrs_transport::create_low_latency_want_list;

        b.iter(|| {
            let list = create_low_latency_want_list();
            black_box(list);
        });
    });

    group.bench_function("create_latency_optimized_peer_manager", |b| {
        use ipfrs_transport::create_latency_optimized_peer_manager;

        b.iter(|| {
            let manager = create_latency_optimized_peer_manager();
            black_box(manager);
        });
    });

    group.bench_function("create_bandwidth_optimized_peer_manager", |b| {
        use ipfrs_transport::create_bandwidth_optimized_peer_manager;

        b.iter(|| {
            let manager = create_bandwidth_optimized_peer_manager();
            black_box(manager);
        });
    });

    group.bench_function("create_bulk_transfer_session", |b| {
        use ipfrs_transport::create_bulk_transfer_session;

        b.iter(|| {
            let session = create_bulk_transfer_session(black_box(1));
            black_box(session);
        });
    });

    group.finish();
}

/// Benchmark request coalescing operations
fn bench_request_coalescing(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("request_coalescing");

    // Benchmark registering a new request
    group.bench_function("register_first_request", |b| {
        b.iter(|| {
            runtime.block_on(async {
                use ipfrs_transport::{CoalescerConfig, RequestCoalescer};
                let coalescer = RequestCoalescer::new(CoalescerConfig::default());
                let cid = dummy_cid(1);
                let result = coalescer.register_request(&cid).await;
                black_box(result)
            })
        });
    });

    // Benchmark completing a request
    group.bench_function("complete_request", |b| {
        b.iter(|| {
            runtime.block_on(async {
                use bytes::Bytes;
                use ipfrs_transport::{CoalescerConfig, RequestCoalescer};
                let coalescer = RequestCoalescer::new(CoalescerConfig::default());
                let cid = dummy_cid(1);
                coalescer.register_request(&cid).await.unwrap();
                let data = Bytes::from("test data");
                coalescer.complete_request(&cid, data).await;
                black_box(())
            })
        });
    });

    // Benchmark statistics collection
    group.bench_function("get_stats", |b| {
        b.iter(|| {
            runtime.block_on(async {
                use ipfrs_transport::{CoalescerConfig, RequestCoalescer};
                let coalescer = RequestCoalescer::new(CoalescerConfig::default());
                black_box(coalescer.stats().await)
            })
        });
    });

    group.finish();
}

/// Benchmark connection migration operations
fn bench_connection_migration(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("connection_migration");

    // Benchmark starting a migration
    group.bench_function("start_migration", |b| {
        b.iter(|| {
            runtime.block_on(async {
                use ipfrs_transport::{ConnectionMigration, MigrationConfig};
                let migration = ConnectionMigration::new(MigrationConfig::default());
                let old_addr = "127.0.0.1:8000".parse().unwrap();
                let new_addr = "127.0.0.1:8001".parse().unwrap();
                black_box(
                    migration
                        .start_migration("conn1".to_string(), old_addr, new_addr)
                        .await,
                )
            })
        });
    });

    // Benchmark completing a migration
    group.bench_function("complete_migration", |b| {
        b.iter(|| {
            runtime.block_on(async {
                use ipfrs_transport::{ConnectionMigration, MigrationConfig};
                let migration = ConnectionMigration::new(MigrationConfig::default());
                let old_addr = "127.0.0.1:8000".parse().unwrap();
                let new_addr = "127.0.0.1:8001".parse().unwrap();
                migration
                    .start_migration("conn1".to_string(), old_addr, new_addr)
                    .await
                    .unwrap();
                black_box(migration.complete_migration("conn1").await)
            })
        });
    });

    // Benchmark getting migration state
    group.bench_function("get_state", |b| {
        b.iter(|| {
            runtime.block_on(async {
                use ipfrs_transport::{ConnectionMigration, MigrationConfig};
                let migration = ConnectionMigration::new(MigrationConfig::default());
                black_box(migration.get_state("conn1"))
            })
        });
    });

    group.finish();
}

/// Benchmark advanced scheduling operations
fn bench_advanced_scheduling(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("advanced_scheduling");

    // Benchmark scheduling with FIFO
    group.bench_function("schedule_fifo", |b| {
        b.iter(|| {
            runtime.block_on(async {
                use ipfrs_transport::{
                    AdvancedScheduler, SchedulePriority, ScheduledRequest, SchedulingPolicy,
                };
                let scheduler = AdvancedScheduler::new(SchedulingPolicy::Fifo);
                let req = ScheduledRequest::new(dummy_cid(1), SchedulePriority::Normal);
                scheduler.schedule(req).await;
                black_box(())
            })
        });
    });

    // Benchmark scheduling with Earliest Deadline First
    group.bench_function("schedule_edf", |b| {
        b.iter(|| {
            runtime.block_on(async {
                use ipfrs_transport::{
                    AdvancedScheduler, SchedulePriority, ScheduledRequest, SchedulingPolicy,
                };
                let scheduler = AdvancedScheduler::new(SchedulingPolicy::EarliestDeadlineFirst);
                let req = ScheduledRequest::new(dummy_cid(1), SchedulePriority::Normal)
                    .with_deadline(Instant::now() + Duration::from_secs(10));
                scheduler.schedule(req).await;
                black_box(())
            })
        });
    });

    // Benchmark getting next request
    group.bench_function("get_next", |b| {
        b.iter(|| {
            runtime.block_on(async {
                use ipfrs_transport::{
                    AdvancedScheduler, SchedulePriority, ScheduledRequest, SchedulingPolicy,
                };
                let scheduler = AdvancedScheduler::new(SchedulingPolicy::Fifo);
                let req = ScheduledRequest::new(dummy_cid(1), SchedulePriority::Normal);
                scheduler.schedule(req).await;
                black_box(scheduler.next().await)
            })
        });
    });

    // Benchmark scheduling 100 requests
    group.bench_function("schedule_100_requests", |b| {
        b.iter(|| {
            runtime.block_on(async {
                use ipfrs_transport::{
                    AdvancedScheduler, SchedulePriority, ScheduledRequest, SchedulingPolicy,
                };
                let scheduler = AdvancedScheduler::new(SchedulingPolicy::WeightedFairQueueing);
                for i in 0..100 {
                    let priority = match i % 5 {
                        0 => SchedulePriority::Low,
                        1 => SchedulePriority::Normal,
                        2 => SchedulePriority::High,
                        3 => SchedulePriority::Urgent,
                        _ => SchedulePriority::Critical,
                    };
                    let req = ScheduledRequest::new(dummy_cid(i), priority);
                    scheduler.schedule(req).await;
                }
                black_box(scheduler.queue_size().await)
            })
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_want_list,
    bench_peer_manager,
    bench_messages,
    bench_erasure_coding,
    bench_multicast,
    bench_tensor_metadata,
    bench_cid_operations,
    bench_latency_tracking,
    bench_memory_tracking,
    bench_throughput_tracking,
    bench_cdn_edge,
    bench_with_latency_distribution,
    bench_session_management,
    bench_stream_queue,
    bench_concurrent_ops,
    bench_utility_helpers,
    bench_config_presets,
    bench_request_coalescing,
    bench_connection_migration,
    bench_advanced_scheduling,
);
criterion_main!(benches);
