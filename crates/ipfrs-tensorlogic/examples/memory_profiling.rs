//! Memory profiling example
//!
//! This example demonstrates how to use the MemoryProfiler to track memory usage
//! across different operations in the IPFRS TensorLogic system.
//!
//! Run with:
//! ```bash
//! cargo run --example memory_profiling
//! ```

use ipfrs_tensorlogic::{
    ArrowTensor, ArrowTensorStore, BufferPool, Constant, GradientCompressor, InferenceEngine,
    KnowledgeBase, MemoryProfiler, Predicate, Rule, SafetensorsWriter, Term,
};
use std::time::Duration;

fn main() {
    println!("=== IPFRS TensorLogic Memory Profiling Demo ===\n");

    let profiler = MemoryProfiler::new();

    // 1. Profile tensor creation
    println!("1. Profiling tensor creation...");
    {
        let _guard = profiler.start_tracking("tensor_creation");
        let mut store = ArrowTensorStore::new();

        for i in 0..10 {
            let data: Vec<f32> = (0..10000).map(|x| x as f32 * 0.1).collect();
            let tensor =
                ArrowTensor::from_slice_f32(&format!("tensor_{}", i), vec![100, 100], &data);
            store.insert(tensor);
        }

        std::thread::sleep(Duration::from_millis(50));
    }

    // 2. Profile buffer pooling
    println!("2. Profiling buffer pooling...");
    {
        let _guard = profiler.start_tracking("buffer_pooling");
        let pool = BufferPool::new(4096, 100);

        let mut buffers = Vec::new();
        for _ in 0..50 {
            let mut buf = pool.acquire();
            buf.as_mut().extend_from_slice(&[1u8; 1024]);
            buffers.push(buf);
        }

        std::thread::sleep(Duration::from_millis(30));
    }

    // 3. Profile Safetensors serialization
    println!("3. Profiling Safetensors serialization...");
    {
        let _guard = profiler.start_tracking("safetensors_serialization");
        let mut writer = SafetensorsWriter::new();

        // Add multiple tensors
        let weights: Vec<f32> = (0..10000).map(|x| x as f32 * 0.01).collect();
        writer.add_f32("layer1.weight", vec![100, 100], &weights);

        let bias: Vec<f32> = (0..100).map(|x| x as f32 * 0.1).collect();
        writer.add_f32("layer1.bias", vec![100], &bias);

        writer.add_f32("layer2.weight", vec![100, 50], &[0.5; 5000]);
        writer.add_f32("layer2.bias", vec![50], &[0.01; 50]);

        let _bytes = writer.serialize().unwrap();
        std::thread::sleep(Duration::from_millis(20));
    }

    // 4. Profile gradient compression
    println!("4. Profiling gradient compression...");
    {
        let _guard = profiler.start_tracking("gradient_compression");
        let gradient: Vec<f32> = (0..50000).map(|x| (x as f32 * 0.001).sin()).collect();

        // Top-k compression
        let _sparse = GradientCompressor::top_k(&gradient, vec![50000], 5000).unwrap();

        // Threshold compression
        let _sparse2 = GradientCompressor::threshold(&gradient, vec![50000], 0.1);

        // Quantization
        let _quantized = GradientCompressor::quantize(&gradient, vec![50000]);

        std::thread::sleep(Duration::from_millis(40));
    }

    // 5. Profile knowledge base operations
    println!("5. Profiling knowledge base operations...");
    {
        let _guard = profiler.start_tracking("knowledge_base_operations");
        let mut kb = KnowledgeBase::new();

        // Add many facts
        for i in 0..1000 {
            kb.add_fact(Predicate::new(
                "person".to_string(),
                vec![
                    Term::Const(Constant::String(format!("person_{}", i))),
                    Term::Const(Constant::String(format!("age_{}", i % 100))),
                ],
            ));
        }

        // Add rules
        kb.add_rule(Rule::new(
            Predicate::new("adult".to_string(), vec![Term::Var("X".to_string())]),
            vec![Predicate::new(
                "person".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Age".to_string())],
            )],
        ));

        std::thread::sleep(Duration::from_millis(25));
    }

    // 6. Profile inference operations
    println!("6. Profiling inference operations...");
    {
        let _guard = profiler.start_tracking("inference_operations");
        let mut kb = KnowledgeBase::new();

        // Add parent facts
        for i in 0..100 {
            kb.add_fact(Predicate::new(
                "parent".to_string(),
                vec![
                    Term::Const(Constant::String(format!("person_{}", i))),
                    Term::Const(Constant::String(format!("person_{}", i + 1))),
                ],
            ));
        }

        // Add ancestor rule
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

        // Perform queries
        let engine = InferenceEngine::new();
        for i in 0..10 {
            let query = Predicate::new(
                "ancestor".to_string(),
                vec![
                    Term::Const(Constant::String(format!("person_{}", i))),
                    Term::Var("Y".to_string()),
                ],
            );
            let _results = engine.query(&query, &kb).unwrap();
        }

        std::thread::sleep(Duration::from_millis(60));
    }

    // 7. Profile repeated operations
    println!("7. Profiling repeated operations...");
    for _ in 0..5 {
        let _guard = profiler.start_tracking("repeated_small_alloc");
        let _data: Vec<u8> = vec![0; 1024];
        std::thread::sleep(Duration::from_millis(5));
    }

    // Generate and print the report
    println!("\n=== Final Memory Profiling Report ===\n");
    let report = profiler.generate_report();
    report.print();

    // Print individual statistics
    println!("\n=== Detailed Statistics ===\n");
    let all_stats = profiler.get_all_stats();
    for (name, stats) in all_stats.iter() {
        println!("Operation: {}", name);
        println!("  Track count: {}", stats.track_count);
        println!("  Peak memory: {:.2} KB", stats.peak_bytes as f64 / 1024.0);
        println!("  Avg memory: {:.2} KB", stats.avg_bytes as f64 / 1024.0);
        println!("  Avg duration: {:?}", stats.avg_duration);
        println!();
    }

    println!("=== Memory Profiling Demo Complete ===");
}
