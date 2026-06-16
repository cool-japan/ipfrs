use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ipfrs_core::Cid;
use ipfrs_semantic::{LearnedIndex, ModelType, RMIConfig};
use std::hint::black_box;

fn generate_random_vector(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim).map(|i| ((seed + i) as f32).sin()).collect()
}

fn bench_learned_index_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("learned_index_insert");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let config = RMIConfig {
                    num_models: 10,
                    model_type: ModelType::Linear,
                    ..Default::default()
                };
                let mut index = LearnedIndex::new(config);

                for i in 0..size {
                    let cid = Cid::default();
                    let embedding = generate_random_vector(128, i);
                    let _ = index.add(cid, embedding);
                }
                black_box(index)
            });
        });
    }

    group.finish();
}

fn bench_learned_index_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("learned_index_search");

    for size in [100, 1000, 10000].iter() {
        let config = RMIConfig {
            num_models: 10,
            model_type: ModelType::Linear,
            ..Default::default()
        };
        let mut index = LearnedIndex::new(config);

        // Populate index
        for i in 0..*size {
            let cid = Cid::default();
            let embedding = generate_random_vector(128, i);
            let _ = index.add(cid, embedding);
        }
        let _ = index.rebuild();

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let query = generate_random_vector(128, 42);
                let results = index.search(&query, 10);
                black_box(results)
            });
        });
    }

    group.finish();
}

fn bench_model_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("learned_index_model_types");

    let model_types = vec![
        ("Linear", ModelType::Linear),
        ("Polynomial", ModelType::Polynomial),
        ("NeuralNetwork", ModelType::NeuralNetwork),
    ];

    for (name, model_type) in model_types {
        group.bench_function(name, |b| {
            b.iter(|| {
                let config = RMIConfig {
                    num_models: 10,
                    model_type,
                    training_iterations: 50,
                    ..Default::default()
                };
                let mut index = LearnedIndex::new(config);

                // Add data
                for i in 0..1000 {
                    let cid = Cid::default();
                    let embedding = generate_random_vector(128, i);
                    let _ = index.add(cid, embedding);
                }

                // Search
                let query = generate_random_vector(128, 42);
                let results = index.search(&query, 10);
                black_box(results)
            });
        });
    }

    group.finish();
}

fn bench_learned_vs_brute_force(c: &mut Criterion) {
    let mut group = c.benchmark_group("learned_vs_brute");

    let size = 10000;
    let dim = 128;

    // Setup learned index
    let config = RMIConfig {
        num_models: 10,
        model_type: ModelType::Linear,
        ..Default::default()
    };
    let mut learned_index = LearnedIndex::new(config);

    // Setup brute force data
    let mut brute_force_data: Vec<(Cid, Vec<f32>)> = Vec::new();

    for i in 0..size {
        let cid = Cid::default();
        let embedding = generate_random_vector(dim, i);
        let _ = learned_index.add(cid, embedding.clone());
        brute_force_data.push((cid, embedding));
    }

    let _ = learned_index.rebuild();
    let query = generate_random_vector(dim, 42);

    group.bench_function("learned_index", |b| {
        b.iter(|| {
            let results = learned_index.search(&query, 10);
            black_box(results)
        });
    });

    group.bench_function("brute_force", |b| {
        b.iter(|| {
            let mut distances: Vec<(Cid, f32)> = brute_force_data
                .iter()
                .map(|(cid, embedding)| {
                    let dist: f32 = query
                        .iter()
                        .zip(embedding.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f32>()
                        .sqrt();
                    (*cid, dist)
                })
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let results: Vec<_> = distances.into_iter().take(10).collect();
            black_box(results)
        });
    });

    group.finish();
}

fn bench_rebuild_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("learned_index_rebuild");

    for size in [1000, 5000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let config = RMIConfig {
                    num_models: 10,
                    model_type: ModelType::Linear,
                    ..Default::default()
                };
                let mut index = LearnedIndex::new(config);

                for i in 0..size {
                    let cid = Cid::default();
                    let embedding = generate_random_vector(128, i);
                    let _ = index.add(cid, embedding);
                }

                let _ = index.rebuild();
                black_box(index)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_learned_index_insert,
    bench_learned_index_search,
    bench_model_types,
    bench_learned_vs_brute_force,
    bench_rebuild_performance
);
criterion_main!(benches);
