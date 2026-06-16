//! Benchmarks for logic programming operations

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ipfrs::{Node, NodeConfig};
use ipfrs_tensorlogic::{Constant, Predicate, Rule, Term};
use std::hint::black_box;
use tokio::runtime::Runtime;

/// Setup a knowledge base with facts and rules
async fn setup_kb(node: &mut Node, fact_count: usize) {
    // Add facts: parent(X, Y)
    for i in 0..fact_count {
        let parent = format!("person{}", i);
        let child = format!("person{}", i + 1);
        let predicate = Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String(parent)),
                Term::Const(Constant::String(child)),
            ],
        );
        node.add_fact(predicate).unwrap();
    }

    // Add fact: male(X) and female(X)
    for i in 0..fact_count / 2 {
        let person = format!("person{}", i);
        let predicate = Predicate::new(
            "male".to_string(),
            vec![Term::Const(Constant::String(person))],
        );
        node.add_fact(predicate).unwrap();
    }

    for i in fact_count / 2..fact_count {
        let person = format!("person{}", i);
        let predicate = Predicate::new(
            "female".to_string(),
            vec![Term::Const(Constant::String(person))],
        );
        node.add_fact(predicate).unwrap();
    }

    // Add rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z)
    let grandparent_rule = Rule::new(
        Predicate::new(
            "grandparent".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Z".to_string())],
        ),
        vec![
            Predicate::new(
                "parent".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            Predicate::new(
                "parent".to_string(),
                vec![Term::Var("Y".to_string()), Term::Var("Z".to_string())],
            ),
        ],
    );
    node.add_rule(grandparent_rule).unwrap();

    // Add rule: father(X, Y) :- parent(X, Y), male(X)
    let father_rule = Rule::new(
        Predicate::new(
            "father".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        ),
        vec![
            Predicate::new(
                "parent".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            Predicate::new("male".to_string(), vec![Term::Var("X".to_string())]),
        ],
    );
    node.add_rule(father_rule).unwrap();
}

/// Benchmark adding facts
fn bench_add_fact(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("add_fact");

    for count in [10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, &count| {
            b.to_async(&rt).iter(|| async {
                let config = NodeConfig::default().with_tensorlogic();
                let mut node = Node::new(config).unwrap();
                node.start().await.unwrap();

                for i in 0..count {
                    let predicate = Predicate::new(
                        "test".to_string(),
                        vec![
                            Term::Const(Constant::Int(i as i64)),
                            Term::Const(Constant::String(format!("value{}", i))),
                        ],
                    );
                    node.add_fact(predicate).unwrap();
                }

                node.stop().await.unwrap();
                black_box(count)
            });
        });
    }
    group.finish();
}

/// Benchmark adding rules
fn bench_add_rule(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("add_rule");

    group.bench_function("add_rule", |b| {
        b.to_async(&rt).iter(|| async {
            let config = NodeConfig::default().with_tensorlogic();
            let mut node = Node::new(config).unwrap();
            node.start().await.unwrap();

            for i in 0..10 {
                let rule = Rule::new(
                    Predicate::new(format!("derived{}", i), vec![Term::Var("X".to_string())]),
                    vec![Predicate::new(
                        "base".to_string(),
                        vec![Term::Var("X".to_string())],
                    )],
                );
                node.add_rule(rule).unwrap();
            }

            node.stop().await.unwrap();
        });
    });
    group.finish();
}

/// Benchmark simple inference queries
fn bench_simple_inference(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("simple_inference");

    for kb_size in [10, 50, 100].iter() {
        let config = NodeConfig::default().with_tensorlogic();
        let mut node = rt.block_on(async {
            let mut node = Node::new(config).unwrap();
            node.start().await.unwrap();
            setup_kb(&mut node, *kb_size).await;
            node
        });

        group.bench_with_input(
            BenchmarkId::from_parameter(kb_size),
            kb_size,
            |b, _kb_size| {
                b.to_async(&rt).iter(|| async {
                    // Query: parent(X, person1) - find parent of person1
                    let goal = Predicate::new(
                        "parent".to_string(),
                        vec![
                            Term::Var("X".to_string()),
                            Term::Const(Constant::String("person1".to_string())),
                        ],
                    );

                    black_box(node.infer(&goal).unwrap())
                });
            },
        );

        rt.block_on(async {
            node.stop().await.unwrap();
        });
    }
    group.finish();
}

/// Benchmark complex inference with joins
fn bench_complex_inference(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("complex_inference");

    for kb_size in [10, 50].iter() {
        let config = NodeConfig::default().with_tensorlogic();
        let mut node = rt.block_on(async {
            let mut node = Node::new(config).unwrap();
            node.start().await.unwrap();
            setup_kb(&mut node, *kb_size).await;
            node
        });

        group.bench_with_input(
            BenchmarkId::from_parameter(kb_size),
            kb_size,
            |b, _kb_size| {
                b.to_async(&rt).iter(|| async {
                    // Query: grandparent(X, Y) - find all grandparent relationships
                    let goal = Predicate::new(
                        "grandparent".to_string(),
                        vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
                    );

                    black_box(node.infer(&goal).unwrap())
                });
            },
        );

        rt.block_on(async {
            node.stop().await.unwrap();
        });
    }
    group.finish();
}

/// Benchmark proof generation
fn bench_prove(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("prove");

    let config = NodeConfig::default().with_tensorlogic();
    let mut node = rt.block_on(async {
        let mut node = Node::new(config).unwrap();
        node.start().await.unwrap();
        setup_kb(&mut node, 20).await;
        node
    });

    group.bench_function("prove", |b| {
        b.to_async(&rt).iter(|| async {
            // Query: grandparent(person0, person2)
            let goal = Predicate::new(
                "grandparent".to_string(),
                vec![
                    Term::Const(Constant::String("person0".to_string())),
                    Term::Const(Constant::String("person2".to_string())),
                ],
            );

            black_box(node.prove(&goal).unwrap())
        });
    });

    rt.block_on(async {
        node.stop().await.unwrap();
    });

    group.finish();
}

/// Benchmark knowledge base statistics
fn bench_kb_stats(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("kb_stats");

    for kb_size in [10, 50, 100].iter() {
        let config = NodeConfig::default().with_tensorlogic();
        let mut node = rt.block_on(async {
            let mut node = Node::new(config).unwrap();
            node.start().await.unwrap();
            setup_kb(&mut node, *kb_size).await;
            node
        });

        group.bench_with_input(
            BenchmarkId::from_parameter(kb_size),
            kb_size,
            |b, _kb_size| {
                b.to_async(&rt)
                    .iter(|| async { black_box(node.kb_stats().unwrap()) });
            },
        );

        rt.block_on(async {
            node.stop().await.unwrap();
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_add_fact,
    bench_add_rule,
    bench_simple_inference,
    bench_complex_inference,
    bench_prove,
    bench_kb_stats
);
criterion_main!(benches);
