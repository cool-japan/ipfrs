//! Distributed Training Example - Federated Learning
//!
//! This example demonstrates a realistic distributed training scenario using:
//! - Gradient exchange between workers
//! - Gradient aggregation strategies (FederatedAvg)
//! - Tensor streaming with backpressure
//! - Progress tracking and performance monitoring
//! - Session management for coordinated training
//!
//! Run with: cargo run --example distributed_training

use bytes::Bytes;
use ipfrs_core::Cid;
use ipfrs_transport::{
    AggregationStrategy, GradientAggregator, GradientMessage, Priority, Session, SessionConfig,
    TensorMetadata, TensorStream,
};
use multihash::Multihash;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;

/// Create a dummy CID for demonstration
fn create_cid(seed: u64) -> Cid {
    let data = seed.to_le_bytes();
    let hash = Multihash::wrap(0x12, &data).unwrap();
    Cid::new_v1(0x55, hash)
}

/// Simulated gradient data for a layer
fn create_gradient_data(_layer_id: &str, worker_id: u8, size: usize) -> Vec<u8> {
    // In real scenario, this would be actual gradient data
    // For demo, create synthetic data
    let mut data = vec![0u8; size];
    for (i, byte) in data.iter_mut().enumerate() {
        *byte = ((i as u64 + worker_id as u64) % 256) as u8;
    }
    data
}

/// Simulate a worker node in federated learning
struct Worker {
    id: u8,
    local_data_size: usize,
}

impl Worker {
    fn new(id: u8, local_data_size: usize) -> Self {
        Worker {
            id,
            local_data_size,
        }
    }

    /// Simulate local training and gradient computation
    fn train_local_epoch(&mut self, epoch: u32) -> Vec<GradientMessage> {
        println!(
            "  Worker {} training epoch {} with {} samples",
            self.id, epoch, self.local_data_size
        );

        // Simulate computing gradients for multiple layers
        let layer_names = ["layer1", "layer2", "layer3", "output"];
        let mut gradients = Vec::new();

        for (idx, layer_name) in layer_names.iter().enumerate() {
            // Use layer name as ID for aggregation
            let data = create_gradient_data(layer_name, self.id, 1024 * (idx + 1)); // Different sizes

            let message = GradientMessage::new(
                layer_name.to_string(),
                data,
                vec![32, 32 * (idx + 1)], // Shape
                "float32".to_string(),
            )
            .with_metadata("layer".to_string(), layer_name.to_string())
            .with_metadata("epoch".to_string(), epoch.to_string())
            .with_metadata("worker".to_string(), self.id.to_string())
            .with_metadata("samples".to_string(), self.local_data_size.to_string());

            gradients.push(message);
        }

        gradients
    }
}

/// Simulate a parameter server in federated learning
struct ParameterServer {
    aggregator: GradientAggregator,
    aggregation_strategy: AggregationStrategy,
    runtime: Runtime,
}

impl ParameterServer {
    fn new(total_workers: usize, strategy: AggregationStrategy) -> Self {
        let aggregator = GradientAggregator::new(strategy, total_workers);
        let runtime = Runtime::new().unwrap();

        ParameterServer {
            aggregator,
            aggregation_strategy: strategy,
            runtime,
        }
    }

    /// Receive gradients from a worker
    fn receive_gradients(
        &mut self,
        worker_id: u8,
        gradients: Vec<GradientMessage>,
    ) -> Result<(), String> {
        println!(
            "  Parameter server receiving {} gradients from worker {}",
            gradients.len(),
            worker_id
        );

        for gradient in gradients {
            self.runtime
                .block_on(self.aggregator.add_gradient(gradient))
                .map_err(|e| format!("Failed to add gradient: {:?}", e))?;
        }

        Ok(())
    }

    /// Check if all workers have contributed for a layer
    fn is_layer_ready(&self, layer_name: &str) -> bool {
        self.runtime.block_on(self.aggregator.is_ready(layer_name))
    }

    /// Aggregate gradients for a layer
    fn aggregate_layer(&mut self, layer_name: &str) -> Result<Option<GradientMessage>, String> {
        self.runtime
            .block_on(self.aggregator.aggregate(layer_name))
            .map(Some)
            .map_err(|e| format!("Failed to aggregate: {:?}", e))
    }

    /// Get aggregation statistics
    fn stats(&self) -> String {
        let stats = self.runtime.block_on(self.aggregator.stats());
        format!(
            "Layers tracked: {}, Total gradients: {}, Expected contributors: {}",
            stats.layers_count, stats.total_gradients, stats.expected_contributors
        )
    }
}

/// Simulate a distributed training coordinator
struct TrainingCoordinator {
    workers: Vec<Worker>,
    parameter_server: ParameterServer,
    session: Session,
}

impl TrainingCoordinator {
    fn new(num_workers: usize, aggregation_strategy: AggregationStrategy) -> Self {
        // Create workers with different data sizes (simulating non-IID data)
        let mut workers = Vec::new();
        for i in 0..num_workers {
            let data_size = 1000 + i * 500; // Varying dataset sizes
            workers.push(Worker::new(i as u8, data_size));
        }

        // Create parameter server
        let parameter_server = ParameterServer::new(num_workers, aggregation_strategy);

        // Create a session for this training run
        let session_config = SessionConfig {
            timeout: Duration::from_secs(600), // 10 minutes per epoch
            default_priority: Priority::High,
            max_concurrent_blocks: 1000,
            progress_notifications: true,
        };
        let session = Session::new(1, session_config, None);

        TrainingCoordinator {
            workers,
            parameter_server,
            session,
        }
    }

    /// Run one training epoch
    fn run_epoch(&mut self, epoch: u32) -> Result<(), String> {
        println!("\n=== Epoch {} ===", epoch);
        let epoch_start = Instant::now();

        // Phase 1: Local training
        println!("\nPhase 1: Local Training");
        let mut all_gradients = HashMap::new();

        for worker in &mut self.workers {
            let gradients = worker.train_local_epoch(epoch);
            all_gradients.insert(worker.id, gradients);
        }

        // Phase 2: Gradient exchange
        println!("\nPhase 2: Gradient Exchange");
        for (worker_id, gradients) in all_gradients {
            self.parameter_server
                .receive_gradients(worker_id, gradients)?;
        }

        // Phase 3: Aggregation
        println!("\nPhase 3: Gradient Aggregation");
        let layer_names = vec!["layer1", "layer2", "layer3", "output"];
        let mut aggregated_gradients = Vec::new();

        for layer_name in &layer_names {
            if self.parameter_server.is_layer_ready(layer_name) {
                if let Some(aggregated) = self.parameter_server.aggregate_layer(layer_name)? {
                    println!("  Aggregated gradients for {}", layer_name);
                    println!(
                        "    Shape: {:?}, Size: {} bytes",
                        aggregated.shape,
                        aggregated.data.len()
                    );
                    aggregated_gradients.push(aggregated);
                }
            }
        }

        // Phase 4: Broadcast updated parameters (simulated)
        println!("\nPhase 4: Parameter Broadcast");
        println!(
            "  Broadcasting {} updated parameters to {} workers",
            aggregated_gradients.len(),
            self.workers.len()
        );

        // Update session statistics
        let blocks_transferred = aggregated_gradients.len() * self.workers.len();
        let cid = create_cid(epoch as u64);
        let data = Bytes::from(vec![0u8; 1024]);
        for _ in 0..blocks_transferred {
            let _ = self.session.mark_received(&cid, &data);
        }

        let epoch_duration = epoch_start.elapsed();
        println!("\nEpoch {} completed in {:?}", epoch, epoch_duration);
        println!(
            "  Parameter server stats: {}",
            self.parameter_server.stats()
        );

        // Show session progress
        let stats = self.session.stats();
        if stats.total_blocks > 0 {
            println!(
                "  Session progress: {:.1}% ({}/{})",
                stats.progress(),
                stats.blocks_received,
                stats.total_blocks
            );
        }

        Ok(())
    }

    /// Run full training
    fn run_training(&mut self, num_epochs: u32) -> Result<(), String> {
        println!("=== Distributed Training Started ===");
        println!("Workers: {}", self.workers.len());
        println!(
            "Aggregation strategy: {:?}",
            self.parameter_server.aggregation_strategy
        );

        let training_start = Instant::now();

        for epoch in 1..=num_epochs {
            self.run_epoch(epoch)?;
        }

        let total_duration = training_start.elapsed();
        println!("\n=== Training Completed ===");
        println!("Total epochs: {}", num_epochs);
        println!("Total time: {:?}", total_duration);
        println!("Average time per epoch: {:?}", total_duration / num_epochs);

        // Final statistics
        let session_stats = self.session.stats();
        println!("\nFinal Session Statistics:");
        println!("  Total blocks: {}", session_stats.total_blocks);
        println!("  Received blocks: {}", session_stats.blocks_received);
        println!("  Total bytes: {}", session_stats.bytes_transferred);
        println!("  Progress: {:.1}%", session_stats.progress());

        Ok(())
    }
}

fn main() {
    println!("=== Distributed Training Example - Federated Learning ===\n");

    // Scenario 1: FederatedAvg with 3 workers
    println!("--- Scenario 1: FederatedAvg (3 workers) ---");
    let mut coordinator = TrainingCoordinator::new(3, AggregationStrategy::FederatedAvg);

    if let Err(e) = coordinator.run_training(3) {
        eprintln!("Training failed: {}", e);
    }

    // Scenario 2: WeightedAverage with 5 workers
    println!("\n\n--- Scenario 2: WeightedAverage (5 workers) ---");
    let mut coordinator2 = TrainingCoordinator::new(5, AggregationStrategy::WeightedAverage);

    if let Err(e) = coordinator2.run_training(2) {
        eprintln!("Training failed: {}", e);
    }

    // Demonstrate tensor streaming
    println!("\n\n--- Scenario 3: Large Model Tensor Streaming ---");
    demonstrate_tensor_streaming();

    println!("\n=== Example Completed ===");
}

/// Demonstrate tensor streaming for large models
fn demonstrate_tensor_streaming() {
    println!("Simulating large model parameter streaming...");

    // Create a large tensor metadata (e.g., for a large language model layer)
    let root_cid = create_cid(9999);
    let chunk_cids: Vec<Cid> = (0..10).map(|i| create_cid(1000 + i)).collect();

    let metadata = TensorMetadata::new(root_cid)
        .with_shape(vec![4096, 4096]) // Large matrix
        .with_dtype("float32".to_string())
        .with_size(4096 * 4096 * 4) // 64 MB
        .with_chunks(chunk_cids.clone())
        .with_priority_hint(1000) // High priority value
        .with_deadline(Instant::now() + Duration::from_secs(30));

    let stream = TensorStream::new(metadata);

    println!("  Tensor shape: {:?}", stream.metadata.shape);
    if let Some(size) = stream.metadata.size_bytes {
        println!("  Total size: {} MB", size / 1024 / 1024);
    }
    println!("  Chunks: {}", chunk_cids.len());
    println!("  Initial progress: {:.1}%", stream.progress() * 100.0);

    // In a real scenario, chunks would be received asynchronously via mark_received()
    // For this demo, we just show the stream setup
    println!("  Stream initialized and ready to receive chunks");
    println!("  (In production, use async mark_received() to track chunk reception)")
}
