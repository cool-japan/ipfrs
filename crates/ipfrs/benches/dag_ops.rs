//! Benchmarks for DAG operations

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ipfrs::{Ipld, Node, NodeConfig};
use std::collections::BTreeMap;
use std::hint::black_box;
use tokio::runtime::Runtime;

/// Create a simple DAG structure for testing
fn create_test_dag(depth: usize) -> Ipld {
    if depth == 0 {
        Ipld::String("leaf".to_string())
    } else {
        let mut map = BTreeMap::new();
        map.insert("child1".to_string(), create_test_dag(depth - 1));
        map.insert("child2".to_string(), create_test_dag(depth - 1));
        map.insert("data".to_string(), Ipld::String(format!("node_{}", depth)));
        Ipld::Map(map)
    }
}

/// Benchmark DAG put operations
fn bench_dag_put(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("dag_put");

    // Test different DAG depths
    for depth in [1, 3, 5].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, &depth| {
            let dag = create_test_dag(depth);

            b.to_async(&rt).iter(|| async {
                let mut node = Node::new(NodeConfig::default()).unwrap();
                node.start().await.unwrap();

                let cid = black_box(node.dag_put(dag.clone()).await.unwrap());

                node.stop().await.unwrap();
                cid
            });
        });
    }
    group.finish();
}

/// Benchmark DAG get operations
fn bench_dag_get(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("dag_get");

    for depth in [1, 3, 5].iter() {
        // Setup: Create and store DAG
        let cid = rt.block_on(async {
            let mut node = Node::new(NodeConfig::default()).unwrap();
            node.start().await.unwrap();
            let dag = create_test_dag(*depth);
            let cid = node.dag_put(dag).await.unwrap();
            node.stop().await.unwrap();
            cid
        });

        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, _depth| {
            b.to_async(&rt).iter(|| async {
                let mut node = Node::new(NodeConfig::default()).unwrap();
                node.start().await.unwrap();

                let dag = black_box(node.dag_get(&cid).await.unwrap());

                node.stop().await.unwrap();
                dag
            });
        });
    }
    group.finish();
}

/// Benchmark DAG resolve operations
fn bench_dag_resolve(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("dag_resolve");

    // Setup: Create and store a nested DAG
    let cid = rt.block_on(async {
        let mut node = Node::new(NodeConfig::default()).unwrap();
        node.start().await.unwrap();
        let dag = create_test_dag(3);
        let cid = node.dag_put(dag).await.unwrap();
        node.stop().await.unwrap();
        cid
    });

    let paths = vec!["/child1/data", "/child2/data", "/child1/child1/data"];

    for path in paths {
        group.bench_with_input(BenchmarkId::from_parameter(path), &path, |b, path| {
            b.to_async(&rt).iter(|| async {
                let mut node = Node::new(NodeConfig::default()).unwrap();
                node.start().await.unwrap();

                let result = black_box(node.dag_resolve(&cid, path).await.unwrap());

                node.stop().await.unwrap();
                result
            });
        });
    }
    group.finish();
}

/// Benchmark DAG traverse operations
fn bench_dag_traverse(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("dag_traverse");

    for depth in [1, 3, 5].iter() {
        // Setup: Create and store DAG
        let cid = rt.block_on(async {
            let mut node = Node::new(NodeConfig::default()).unwrap();
            node.start().await.unwrap();
            let dag = create_test_dag(*depth);
            let cid = node.dag_put(dag).await.unwrap();
            node.stop().await.unwrap();
            cid
        });

        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, _depth| {
            b.to_async(&rt).iter(|| async {
                let mut node = Node::new(NodeConfig::default()).unwrap();
                node.start().await.unwrap();

                let cids = node.dag_traverse(&cid, None).await.unwrap();
                let count = cids.len();

                node.stop().await.unwrap();
                black_box(count)
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_dag_put,
    bench_dag_get,
    bench_dag_resolve,
    bench_dag_traverse
);
criterion_main!(benches);
