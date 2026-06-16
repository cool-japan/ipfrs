use criterion::{criterion_group, criterion_main, Criterion};
use ipfrs::{Node, NodeConfig};
use ipfrs_tensorlogic::ir::{Constant, Predicate, Rule, Term};
use std::hint::black_box;
use std::path::PathBuf;
use tokio::runtime::Runtime;

fn create_test_node() -> Node {
    let path = "/tmp/ipfrs-bench-logic";
    let _ = std::fs::remove_dir_all(path);

    let mut config = NodeConfig::default();
    config.storage.path = PathBuf::from(path);
    config.enable_semantic = false;
    config.enable_tensorlogic = true;

    Node::new(config).expect("Failed to create node")
}

fn bench_add_fact(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut node = create_test_node();
    rt.block_on(node.start()).unwrap();

    c.bench_function("logic_add_fact", |b| {
        let mut i = 0;
        b.iter(|| {
            let fact = Predicate::new(
                "person".to_string(),
                vec![
                    Term::Const(Constant::String(format!("Person{}", i))),
                    Term::Const(Constant::Int(i as i64)),
                ],
            );
            node.add_fact(black_box(fact)).unwrap();
            i += 1;
        });
    });

    rt.block_on(node.stop()).unwrap();
}

fn bench_add_rule(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut node = create_test_node();
    rt.block_on(node.start()).unwrap();

    c.bench_function("logic_add_rule", |b| {
        let mut i = 0;
        b.iter(|| {
            let rule = Rule::new(
                Predicate::new(format!("derived{}", i), vec![Term::Var("X".to_string())]),
                vec![Predicate::new(
                    "base".to_string(),
                    vec![Term::Var("X".to_string())],
                )],
            );
            node.add_rule(black_box(rule)).unwrap();
            i += 1;
        });
    });

    rt.block_on(node.stop()).unwrap();
}

fn bench_simple_inference(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut node = create_test_node();
    rt.block_on(node.start()).unwrap();

    // Add facts
    for i in 0..100 {
        let fact = Predicate::new(
            "likes".to_string(),
            vec![
                Term::Const(Constant::String(format!("Person{}", i))),
                Term::Const(Constant::String("Rust".to_string())),
            ],
        );
        node.add_fact(fact).unwrap();
    }

    c.bench_function("logic_simple_inference", |b| {
        b.iter(|| {
            let goal = Predicate::new(
                "likes".to_string(),
                vec![
                    Term::Var("X".to_string()),
                    Term::Const(Constant::String("Rust".to_string())),
                ],
            );
            node.infer(black_box(&goal)).unwrap();
        });
    });

    rt.block_on(node.stop()).unwrap();
}

fn bench_complex_inference(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut node = create_test_node();
    rt.block_on(node.start()).unwrap();

    // Add facts for family relationships
    let people = ["Alice", "Bob", "Charlie", "Diana", "Eve"];
    for i in 0..people.len() - 1 {
        let fact = Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String(people[i].to_string())),
                Term::Const(Constant::String(people[i + 1].to_string())),
            ],
        );
        node.add_fact(fact).unwrap();
    }

    // Add grandparent rule
    let rule = Rule::new(
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
    node.add_rule(rule).unwrap();

    c.bench_function("logic_complex_inference", |b| {
        b.iter(|| {
            let goal = Predicate::new(
                "grandparent".to_string(),
                vec![
                    Term::Var("X".to_string()),
                    Term::Const(Constant::String("Charlie".to_string())),
                ],
            );
            node.infer(black_box(&goal)).unwrap();
        });
    });

    rt.block_on(node.stop()).unwrap();
}

fn bench_prove(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut node = create_test_node();
    rt.block_on(node.start()).unwrap();

    // Add facts
    let fact = Predicate::new(
        "likes".to_string(),
        vec![
            Term::Const(Constant::String("Alice".to_string())),
            Term::Const(Constant::String("Rust".to_string())),
        ],
    );
    node.add_fact(fact.clone()).unwrap();

    c.bench_function("logic_prove", |b| {
        b.iter(|| {
            node.prove(black_box(&fact)).unwrap();
        });
    });

    rt.block_on(node.stop()).unwrap();
}

fn bench_kb_stats(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut node = create_test_node();
    rt.block_on(node.start()).unwrap();

    // Add many facts and rules
    for i in 0..1000 {
        let fact = Predicate::new(
            "data".to_string(),
            vec![
                Term::Const(Constant::Int(i as i64)),
                Term::Const(Constant::String(format!("value{}", i))),
            ],
        );
        node.add_fact(fact).unwrap();
    }

    for i in 0..10 {
        let rule = Rule::new(
            Predicate::new(format!("rule{}", i), vec![Term::Var("X".to_string())]),
            vec![Predicate::new(
                "data".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            )],
        );
        node.add_rule(rule).unwrap();
    }

    c.bench_function("logic_kb_stats", |b| {
        b.iter(|| {
            node.tensorlogic_stats().unwrap();
        });
    });

    rt.block_on(node.stop()).unwrap();
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
