//! Integration tests for performance validation
//!
//! Tests cover:
//! - Inference latency measurement
//! - Memory usage profiling
//! - Gradient tracking validation
//! - End-to-end performance scenarios

use ipfrs_tensorlogic::{
    AdaptiveBatchSizer, ArrowTensor, CacheManager, Constant, ConvergenceDetector,
    DeviceCapabilities, DifferentialPrivacy, GradientAggregator, GradientCompressor,
    InferenceEngine, KnowledgeBase, MemoizedInferenceEngine, ModelRepository, Predicate,
    QueryOptimizer, RemoteFactCache, Rule, SafetensorsWriter, SharedMemoryPool, SharedTensorBuffer,
    TabledInferenceEngine, Term,
};
use std::sync::Arc;
use std::time::Instant;

/// Measure the best-case latency of `op` over several iterations.
///
/// A warmup call (not timed) primes allocators, caches, and the branch
/// predictor; then `op` is timed `iters` times and the **minimum** is returned.
/// Scheduler jitter and parallel-test load can only *add* time to a sample, so
/// the minimum is the measurement least contaminated by environmental noise —
/// this keeps a tight latency bound meaningful while eliminating load-induced
/// flakiness. The final operation's result is returned for correctness asserts.
fn best_latency<T>(iters: usize, mut op: impl FnMut() -> T) -> (std::time::Duration, T) {
    let mut result = op(); // warmup (not timed)
    let mut best = std::time::Duration::MAX;
    for _ in 0..iters {
        let start = Instant::now();
        result = op();
        best = best.min(start.elapsed());
    }
    (best, result)
}

/// Test inference latency for simple fact lookup
#[test]
fn test_inference_latency_simple_facts() {
    let mut kb = KnowledgeBase::new();

    // Add 1000 facts
    for i in 0..1000 {
        kb.add_fact(Predicate::new(
            "data".to_string(),
            vec![
                Term::Const(Constant::String(format!("key_{}", i))),
                Term::Const(Constant::String(format!("value_{}", i))),
            ],
        ));
    }

    let engine = InferenceEngine::new();
    let query = Predicate::new(
        "data".to_string(),
        vec![
            Term::Const(Constant::String("key_500".to_string())),
            Term::Var("V".to_string()),
        ],
    );

    // Best-of-N latency: robust to scheduler jitter under parallel test load,
    // while still enforcing the < 1ms target for a simple fact lookup.
    let (latency, results) = best_latency(10, || {
        engine.query(&query, &kb).expect("query should succeed")
    });

    assert_eq!(results.len(), 1);
    println!("Simple fact lookup best-of-10 latency: {:?}", latency);

    // Target: < 1ms for simple fact lookup
    assert!(
        latency.as_micros() < 1000,
        "Latency too high: {:?}",
        latency
    );
}

/// Test inference latency with rules and backward chaining
#[test]
fn test_inference_latency_with_rules() {
    let mut kb = KnowledgeBase::new();

    // Add facts: parent relationships
    for i in 0..50 {
        kb.add_fact(Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String(format!("person_{}", i))),
                Term::Const(Constant::String(format!("person_{}", i + 1))),
            ],
        ));
    }

    // Add rule: ancestor(X, Y) :- parent(X, Y)
    kb.add_rule(Rule::new(
        Predicate::new(
            "ancestor".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        ),
        vec![Predicate::new(
            "parent".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        )],
    ));

    // Add rule: ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)
    kb.add_rule(Rule::new(
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
    ));

    let engine = InferenceEngine::new();
    let query = Predicate::new(
        "ancestor".to_string(),
        vec![
            Term::Const(Constant::String("person_0".to_string())),
            Term::Var("X".to_string()),
        ],
    );

    // Measure latency
    let start = Instant::now();
    let results = engine.query(&query, &kb).unwrap();
    let latency = start.elapsed();

    assert!(!results.is_empty());
    println!(
        "Rule-based inference latency (50 facts): {:?}, {} results",
        latency,
        results.len()
    );

    // Target: < 10000ms for moderate rule-based inference (debug build under parallel test load)
    assert!(
        latency.as_millis() < 10000,
        "Latency too high: {:?}",
        latency
    );
}

/// Test inference latency with query optimization
#[test]
fn test_inference_latency_with_optimization() {
    let mut kb = KnowledgeBase::new();

    // Add facts for join query
    for i in 0..100 {
        kb.add_fact(Predicate::new(
            "edge".to_string(),
            vec![
                Term::Const(Constant::String(format!("node_{}", i))),
                Term::Const(Constant::String(format!("node_{}", (i + 1) % 100))),
            ],
        ));
    }

    let goals = vec![
        Predicate::new(
            "edge".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        ),
        Predicate::new(
            "edge".to_string(),
            vec![
                Term::Var("Y".to_string()),
                Term::Const(Constant::String("node_50".to_string())),
            ],
        ),
    ];

    let optimizer = QueryOptimizer::new();

    // Best-of-N planning time: stable under parallel test load.
    let (planning_time, _plan) = best_latency(10, || optimizer.plan_query(&goals, &kb));

    println!(
        "Query planning best-of-10 time (100 facts, 2 goals): {:?}",
        planning_time
    );

    // Target: < 1ms for query planning
    assert!(
        planning_time.as_micros() < 1000,
        "Planning time too high: {:?}",
        planning_time
    );
}

/// Test inference latency with memoization
#[test]
fn test_inference_latency_with_memoization() {
    let mut kb = KnowledgeBase::new();

    // Add recursive facts
    for i in 0..20 {
        kb.add_fact(Predicate::new(
            "edge".to_string(),
            vec![
                Term::Const(Constant::String(format!("n{}", i))),
                Term::Const(Constant::String(format!("n{}", i + 1))),
            ],
        ));
    }

    // Add transitive closure rule: path(X, Y) :- edge(X, Y)
    kb.add_rule(Rule::new(
        Predicate::new(
            "path".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        ),
        vec![Predicate::new(
            "edge".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        )],
    ));

    // path(X, Z) :- edge(X, Y), path(Y, Z)
    kb.add_rule(Rule::new(
        Predicate::new(
            "path".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Z".to_string())],
        ),
        vec![
            Predicate::new(
                "edge".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            Predicate::new(
                "path".to_string(),
                vec![Term::Var("Y".to_string()), Term::Var("Z".to_string())],
            ),
        ],
    ));

    let cache = Arc::new(CacheManager::new());
    let engine = MemoizedInferenceEngine::new(cache.clone());

    let query = Predicate::new(
        "path".to_string(),
        vec![
            Term::Const(Constant::String("n0".to_string())),
            Term::Var("X".to_string()),
        ],
    );

    // First query (cold cache)
    let start = Instant::now();
    let results1 = engine.query(&query, &kb).unwrap();
    let cold_latency = start.elapsed();

    // Second query (warm cache)
    let start = Instant::now();
    let results2 = engine.query(&query, &kb).unwrap();
    let warm_latency = start.elapsed();

    assert_eq!(results1.len(), results2.len());
    println!("Cold cache latency: {:?}", cold_latency);
    println!("Warm cache latency: {:?}", warm_latency);

    // Warm cache should be significantly faster
    assert!(
        warm_latency < cold_latency,
        "Cache didn't improve performance"
    );
}

/// Test inference latency with tabling (SLG resolution)
#[test]
fn test_inference_latency_with_tabling() {
    let mut kb = KnowledgeBase::new();

    // Create a graph with cycles
    kb.add_fact(Predicate::new(
        "edge".to_string(),
        vec![
            Term::Const(Constant::String("a".to_string())),
            Term::Const(Constant::String("b".to_string())),
        ],
    ));
    kb.add_fact(Predicate::new(
        "edge".to_string(),
        vec![
            Term::Const(Constant::String("b".to_string())),
            Term::Const(Constant::String("c".to_string())),
        ],
    ));
    kb.add_fact(Predicate::new(
        "edge".to_string(),
        vec![
            Term::Const(Constant::String("c".to_string())),
            Term::Const(Constant::String("a".to_string())),
        ],
    ));

    // path(X, Y) :- edge(X, Y)
    kb.add_rule(Rule::new(
        Predicate::new(
            "path".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        ),
        vec![Predicate::new(
            "edge".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        )],
    ));

    // path(X, Z) :- path(X, Y), edge(Y, Z)
    kb.add_rule(Rule::new(
        Predicate::new(
            "path".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Z".to_string())],
        ),
        vec![
            Predicate::new(
                "path".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            Predicate::new(
                "edge".to_string(),
                vec![Term::Var("Y".to_string()), Term::Var("Z".to_string())],
            ),
        ],
    ));

    let engine = TabledInferenceEngine::new();

    let query = Predicate::new(
        "path".to_string(),
        vec![
            Term::Const(Constant::String("a".to_string())),
            Term::Var("X".to_string()),
        ],
    );

    // Measure latency
    let start = Instant::now();
    let results = engine.query(&query, &kb).unwrap();
    let latency = start.elapsed();

    assert!(!results.is_empty());
    println!("Tabled inference latency: {:?}", latency);

    // Should complete quickly even with cycles
    assert!(latency.as_millis() < 100, "Tabling too slow: {:?}", latency);
}

/// Test memory usage with shared memory pools
#[test]
fn test_memory_usage_shared_buffers() {
    use tempfile::tempdir;

    let temp_dir = tempdir().unwrap();
    let _pool = SharedMemoryPool::new(temp_dir.path(), 100 * 1024 * 1024); // 100MB max

    // Create multiple shared buffers
    let mut buffers = Vec::new();
    for i in 0..5 {
        let path = temp_dir.path().join(format!("test_buffer_{}.bin", i));
        let size = 1024 * 1024; // 1MB each
        let buffer = SharedTensorBuffer::create(&path, size, &[]).unwrap();
        buffers.push(buffer);
    }

    println!("Created {} shared memory buffers", buffers.len());

    // Verify buffers were created
    assert_eq!(buffers.len(), 5);

    // Clean up happens automatically when buffers are dropped
    drop(buffers);
}

/// Test memory usage with Arrow tensors
#[test]
fn test_memory_usage_arrow_tensors() {
    use rand::RngExt;

    let mut rng = rand::rng();

    // Create tensors of different sizes
    let sizes = vec![1024, 4096, 16384, 65536];
    let mut total_bytes = 0;

    for size in sizes {
        let data: Vec<f32> = (0..size).map(|_| rng.random::<f32>()).collect();
        let tensor = ArrowTensor::from_slice_f32(&format!("tensor_{}", size), vec![size], &data);

        let tensor_bytes = size * std::mem::size_of::<f32>();
        total_bytes += tensor_bytes;

        // Verify zero-copy access doesn't increase memory
        let slice = tensor.as_slice_f32().unwrap();
        assert_eq!(slice.len(), size);
    }

    println!("Total Arrow tensor memory: {} bytes", total_bytes);
    assert_eq!(total_bytes, (1024 + 4096 + 16384 + 65536) * 4);
}

/// Test memory usage with remote fact caching
#[test]
fn test_memory_usage_remote_cache() {
    use std::time::Duration;
    let cache = RemoteFactCache::new(1000, Duration::from_secs(300));

    // Add facts to cache
    for i in 0..500 {
        let fact = Predicate::new(
            format!("pred_{}", i % 10),
            vec![Term::Const(Constant::String(format!("fact_{}", i)))],
        );
        cache.add_fact(fact, None);
    }

    // Verify cache operations work
    let cached_facts = cache.get_facts("pred_5");
    assert!(!cached_facts.is_empty());
}

/// Test gradient tracking correctness
#[test]
fn test_gradient_tracking_compression_correctness() {
    use rand::RngExt;

    let mut rng = rand::rng();
    let size = 1000;
    let gradient: Vec<f32> = (0..size).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();

    // Test top-k compression
    let k = 100; // Keep top 10%
    let sparse = GradientCompressor::top_k(&gradient, vec![size], k).unwrap();

    assert_eq!(sparse.nnz(), k);
    assert!(sparse.sparsity_ratio() > 0.8); // At least 80% sparse

    // Decompress and verify largest values are preserved
    let decompressed = sparse.to_dense();
    assert_eq!(decompressed.len(), size);

    // Count non-zero values in decompressed
    let non_zero_count = decompressed.iter().filter(|&&x| x != 0.0).count();
    assert_eq!(non_zero_count, k);

    println!(
        "Top-k compression: {} -> {} elements ({}% sparse)",
        size,
        k,
        sparse.sparsity_ratio() * 100.0
    );
}

/// Test gradient aggregation correctness
#[test]
fn test_gradient_aggregation_correctness() {
    // Create two gradients
    let grad1 = vec![1.0, 2.0, 3.0, 4.0];
    let grad2 = vec![2.0, 3.0, 4.0, 5.0];

    let gradients = vec![grad1, grad2];

    // Average gradients
    let aggregated = GradientAggregator::average(&gradients).unwrap();

    // Should be average: [(1+2)/2, (2+3)/2, (3+4)/2, (4+5)/2] = [1.5, 2.5, 3.5, 4.5]
    assert_eq!(aggregated.len(), 4);
    assert!((aggregated[0] - 1.5).abs() < 1e-6);
    assert!((aggregated[1] - 2.5).abs() < 1e-6);
    assert!((aggregated[2] - 3.5).abs() < 1e-6);
    assert!((aggregated[3] - 4.5).abs() < 1e-6);

    println!("Aggregated gradient: {:?}", aggregated);
}

/// Test gradient tracking with differential privacy
#[test]
fn test_gradient_tracking_with_privacy() {
    use ipfrs_tensorlogic::DPMechanism;

    let mut gradient = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let gradient_orig = gradient.clone();
    let epsilon = 1.0;
    let delta = 1e-5;
    let sensitivity = 1.0;

    let mut dp = DifferentialPrivacy::new(epsilon, delta, sensitivity, DPMechanism::Gaussian);

    // Apply Gaussian noise (modifies gradient in place)
    dp.add_gaussian_noise(&mut gradient).unwrap();

    assert_eq!(gradient.len(), gradient_orig.len());

    // Verify noise was added (values should differ)
    let has_noise = gradient_orig
        .iter()
        .zip(gradient.iter())
        .any(|(&orig, &noisy)| (orig - noisy).abs() > 1e-6);

    assert!(has_noise, "No noise was added");

    println!("Privacy-protected gradient: {:?}", gradient);
}

/// Test convergence detection
#[test]
fn test_convergence_detection() {
    let mut detector = ConvergenceDetector::new(3, 0.01);

    // Simulate converging losses
    let losses = vec![1.0, 0.5, 0.26, 0.255, 0.254, 0.253];

    let mut converged = false;
    for &loss in &losses {
        detector.add_loss(loss);
        if detector.has_converged() {
            converged = true;
            println!("Converged at loss: {}", loss);
            break;
        }
    }

    assert!(converged, "Should have detected convergence");
}

/// Test device-aware batch sizing
#[test]
fn test_device_aware_batch_sizing() {
    // Detect device capabilities
    let caps = DeviceCapabilities::detect().unwrap();
    println!(
        "Device type: {:?}, Memory: {} GB, CPUs: {}",
        caps.device_type,
        caps.memory.total_bytes / 1024 / 1024 / 1024,
        caps.cpu.logical_cores
    );

    // Create adaptive batch sizer
    let sizer = AdaptiveBatchSizer::new(Arc::new(caps))
        .with_min_batch_size(1)
        .with_max_batch_size(256);

    // Test different scenarios
    let scenarios = vec![
        (1024, 100 * 1024 * 1024),         // 1KB items, 100MB model
        (256 * 1024, 500 * 1024 * 1024),   // 256KB items, 500MB model
        (1024 * 1024, 1024 * 1024 * 1024), // 1MB items, 1GB model
    ];

    for (item_size, model_size) in scenarios {
        let batch_size = sizer.calculate(item_size, model_size);
        println!(
            "Item: {} KB, Model: {} MB => Batch: {}",
            item_size / 1024,
            model_size / 1024 / 1024,
            batch_size
        );

        assert!(batch_size >= 1);
        assert!(batch_size <= 256);
    }
}

/// Test end-to-end gradient tracking workflow
#[test]
fn test_gradient_workflow_end_to_end() {
    use ipfrs_tensorlogic::DPMechanism;
    use rand::RngExt;

    let mut rng = rand::rng();

    // Simulate federated learning with 3 clients
    let num_clients = 3;
    let layer_size = 1000;

    // Each client computes gradients
    let mut client_gradients = Vec::new();
    for _i in 0..num_clients {
        let grad: Vec<f32> = (0..layer_size).map(|_| rng.random::<f32>() * 0.1).collect();

        // Compress gradient
        let sparse = GradientCompressor::top_k(&grad, vec![layer_size], 100).unwrap();

        // Convert to dense for aggregation
        let dense = sparse.to_dense();
        client_gradients.push(dense);

        println!("Client: sparsity = {:.2}%", sparse.sparsity_ratio() * 100.0);
    }

    // Aggregate gradients
    let mut aggregated = GradientAggregator::average(&client_gradients).unwrap();

    // Apply differential privacy
    let mut dp = DifferentialPrivacy::new(1.0, 1e-5, 1.0, DPMechanism::Gaussian);
    dp.add_gaussian_noise(&mut aggregated).unwrap();

    println!(
        "Privacy-protected aggregated gradient: {} elements",
        aggregated.len()
    );

    // Verify workflow completed successfully
    assert_eq!(aggregated.len(), layer_size);
}

/// Test model versioning with gradients
#[test]
fn test_model_versioning_workflow() {
    use ipfrs_core::Cid;
    use rand::RngExt;

    let mut rng = rand::rng();

    // Create initial model
    let mut writer = SafetensorsWriter::new();
    let layer1: Vec<f32> = (0..100).map(|_| rng.random::<f32>()).collect();
    writer.add_f32("layer1", vec![10, 10], &layer1);

    let _model_bytes = writer.serialize().unwrap();

    // In a real scenario, we would store the model bytes and get a CID
    // For testing, use default CIDs
    let model_cid1 = Cid::default();
    let model_cid2 = Cid::default();

    // Create repository
    let mut repo = ModelRepository::new();

    // Commit initial model
    let commit1 = repo
        .commit(
            model_cid1,
            "Initial model".to_string(),
            "test_author".to_string(),
        )
        .unwrap();

    println!("Initial commit: {}", commit1.id);

    // Simulate training - apply gradient
    let gradient = vec![0.01f32; 100];
    let updated: Vec<f32> = layer1
        .iter()
        .zip(gradient.iter())
        .map(|(&w, &g)| w - g)
        .collect();

    let mut writer2 = SafetensorsWriter::new();
    writer2.add_f32("layer1", vec![10, 10], &updated);
    let _model_bytes2 = writer2.serialize().unwrap();

    // Commit updated model
    let commit2 = repo
        .commit(
            model_cid2,
            "After gradient update".to_string(),
            "test_author".to_string(),
        )
        .unwrap();

    println!("Second commit: {}", commit2.id);

    // Verify commits were created
    let retrieved1 = repo.get_commit(&commit1.id.to_string());
    let retrieved2 = repo.get_commit(&commit2.id.to_string());

    assert!(retrieved1.is_some());
    assert!(retrieved2.is_some());
}

/// Test integration of caching, optimization, and inference
#[test]
fn test_integrated_query_performance() {
    let mut kb = KnowledgeBase::new();

    // Build a realistic knowledge base
    for i in 0..200 {
        kb.add_fact(Predicate::new(
            "person".to_string(),
            vec![
                Term::Const(Constant::String(format!("p{}", i))),
                Term::Const(Constant::String(format!("age_{}", 20 + (i % 50)))),
            ],
        ));
    }

    for i in 0..150 {
        kb.add_fact(Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String(format!("p{}", i))),
                Term::Const(Constant::String(format!("p{}", i + 50))),
            ],
        ));
    }

    // Add rules
    kb.add_rule(Rule::new(
        Predicate::new(
            "ancestor".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        ),
        vec![Predicate::new(
            "parent".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        )],
    ));

    let cache = Arc::new(CacheManager::new());
    let engine = MemoizedInferenceEngine::new(cache);

    // Query with caching
    let query = Predicate::new(
        "ancestor".to_string(),
        vec![
            Term::Const(Constant::String("p0".to_string())),
            Term::Var("D".to_string()),
        ],
    );

    // First query
    let start = Instant::now();
    let results1 = engine.query(&query, &kb).unwrap();
    let time1 = start.elapsed();

    // Second query (should use cache)
    let start = Instant::now();
    let results2 = engine.query(&query, &kb).unwrap();
    let time2 = start.elapsed();

    println!("First query: {:?} ({} results)", time1, results1.len());
    println!("Second query: {:?} ({} results)", time2, results2.len());

    // Results should be the same
    assert_eq!(results1.len(), results2.len());

    // Second query should be faster or equal
    assert!(time2 <= time1 * 2); // Allow some variance
}
