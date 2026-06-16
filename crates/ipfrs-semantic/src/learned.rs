//! Learned index structures using ML models for data indexing.
//!
//! This module implements learned indices, which use machine learning models
//! to predict the position of data in the index, replacing traditional index
//! structures like B-trees with neural networks or linear models.
//!
//! # Architecture
//!
//! The implementation uses a Recursive Model Index (RMI) architecture:
//! - Stage 0: Root model that routes to second-stage models
//! - Stage 1: Multiple specialized models for different data ranges
//! - Each model learns to predict positions in the sorted data
//!
//! # Example
//!
//! ```
//! use ipfrs_semantic::learned::{LearnedIndex, RMIConfig};
//! use ipfrs_core::cid::Cid;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a learned index with default configuration
//! let mut index = LearnedIndex::new(RMIConfig::default());
//!
//! // Add embeddings with their CIDs
//! let cid = Cid::default();
//! let embedding = vec![0.1, 0.2, 0.3, 0.4];
//! index.add(cid.clone(), embedding.clone())?;
//!
//! // Search for nearest neighbors
//! let query = vec![0.15, 0.25, 0.35, 0.45];
//! let results = index.search(&query, 5)?;
//! # Ok(())
//! # }
//! ```

use ipfrs_core::{Cid, Error, Result};
use serde::{Deserialize, Serialize};

/// Configuration for Recursive Model Index (RMI)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RMIConfig {
    /// Number of models in the second stage
    pub num_models: usize,
    /// Model type to use
    pub model_type: ModelType,
    /// Training iterations for neural models
    pub training_iterations: usize,
    /// Learning rate for neural models
    pub learning_rate: f32,
    /// Error threshold for adaptive model selection
    pub error_threshold: f32,
}

impl Default for RMIConfig {
    fn default() -> Self {
        Self {
            num_models: 10,
            model_type: ModelType::Linear,
            training_iterations: 100,
            learning_rate: 0.01,
            error_threshold: 0.05,
        }
    }
}

/// Type of model to use in the learned index
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    /// Linear regression model
    Linear,
    /// Simple neural network (single hidden layer)
    NeuralNetwork,
    /// Polynomial regression (degree 2)
    Polynomial,
}

/// A single learned model that predicts positions
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Model {
    /// Model type
    model_type: ModelType,
    /// Model weights (interpretation depends on model_type)
    weights: Vec<f32>,
    /// Bias term
    bias: f32,
    /// Input dimension
    input_dim: usize,
}

impl Model {
    /// Create a new model with random initialization
    fn new(model_type: ModelType, input_dim: usize) -> Self {
        let weight_count = match model_type {
            ModelType::Linear => input_dim,
            ModelType::Polynomial => input_dim * 2, // Linear + quadratic terms
            ModelType::NeuralNetwork => input_dim * 8 + 8, // Hidden layer + output layer
        };

        Self {
            model_type,
            weights: vec![0.01; weight_count],
            bias: 0.0,
            input_dim,
        }
    }

    /// Predict position for given input (normalized 0-1)
    fn predict(&self, input: &[f32]) -> f32 {
        match self.model_type {
            ModelType::Linear => self.predict_linear(input),
            ModelType::Polynomial => self.predict_polynomial(input),
            ModelType::NeuralNetwork => self.predict_neural(input),
        }
    }

    fn predict_linear(&self, input: &[f32]) -> f32 {
        let mut sum = self.bias;
        for (i, &val) in input.iter().enumerate() {
            if i < self.weights.len() {
                sum += self.weights[i] * val;
            }
        }
        sum.clamp(0.0, 1.0)
    }

    fn predict_polynomial(&self, input: &[f32]) -> f32 {
        let mut sum = self.bias;
        let half = self.weights.len() / 2;

        // Linear terms
        for (i, &val) in input.iter().enumerate() {
            if i < half {
                sum += self.weights[i] * val;
            }
        }

        // Quadratic terms
        for (i, &val) in input.iter().enumerate() {
            if half + i < self.weights.len() {
                sum += self.weights[half + i] * val * val;
            }
        }

        sum.clamp(0.0, 1.0)
    }

    fn predict_neural(&self, input: &[f32]) -> f32 {
        let hidden_size = 8;
        let input_weights = &self.weights[0..self.input_dim * hidden_size];
        let output_weights = &self.weights[self.input_dim * hidden_size..];

        // Hidden layer with ReLU activation
        let mut hidden = vec![0.0; hidden_size];
        for h in 0..hidden_size {
            let mut sum = 0.0;
            for (i, &val) in input.iter().enumerate() {
                if h * self.input_dim + i < input_weights.len() {
                    sum += input_weights[h * self.input_dim + i] * val;
                }
            }
            hidden[h] = sum.max(0.0); // ReLU
        }

        // Output layer with sigmoid
        let mut output = self.bias;
        for (h, &val) in hidden.iter().enumerate() {
            if h < output_weights.len() {
                output += output_weights[h] * val;
            }
        }

        // Sigmoid activation
        1.0 / (1.0 + (-output).exp())
    }

    /// Train the model on data (simple gradient descent)
    #[allow(dead_code)]
    fn train(&mut self, data: &[(Vec<f32>, f32)], learning_rate: f32, iterations: usize) {
        for _ in 0..iterations {
            for (input, target) in data {
                let prediction = self.predict(input);
                let error = target - prediction;

                // Update weights (simplified gradient descent)
                match self.model_type {
                    ModelType::Linear => {
                        for (i, &val) in input.iter().enumerate() {
                            if i < self.weights.len() {
                                self.weights[i] += learning_rate * error * val;
                            }
                        }
                        self.bias += learning_rate * error;
                    }
                    ModelType::Polynomial => {
                        let half = self.weights.len() / 2;
                        for (i, &val) in input.iter().enumerate() {
                            if i < half {
                                self.weights[i] += learning_rate * error * val;
                            }
                            if half + i < self.weights.len() {
                                self.weights[half + i] += learning_rate * error * val * val;
                            }
                        }
                        self.bias += learning_rate * error;
                    }
                    ModelType::NeuralNetwork => {
                        // Simplified backprop (full implementation would be more complex)
                        for i in 0..self.weights.len() {
                            self.weights[i] += learning_rate * error * 0.01;
                        }
                        self.bias += learning_rate * error;
                    }
                }
            }
        }
    }
}

/// Recursive Model Index (RMI) for learned indexing
pub struct LearnedIndex {
    /// Configuration
    config: RMIConfig,
    /// Root model (stage 0)
    root_model: Option<Model>,
    /// Second stage models
    stage1_models: Vec<Model>,
    /// Sorted data storage (CID, embedding, position)
    data: Vec<(Cid, Vec<f32>)>,
    /// Dimension of embeddings
    dimension: Option<usize>,
    /// Statistics
    stats: IndexStats,
}

#[derive(Debug, Default)]
struct IndexStats {
    /// Number of searches performed
    searches: usize,
    /// Total prediction error
    total_error: f32,
    /// Number of data points
    data_points: usize,
}

impl LearnedIndex {
    /// Create a new learned index
    pub fn new(config: RMIConfig) -> Self {
        Self {
            config,
            root_model: None,
            stage1_models: Vec::new(),
            data: Vec::new(),
            dimension: None,
            stats: IndexStats::default(),
        }
    }

    /// Add an embedding to the index
    pub fn add(&mut self, cid: Cid, embedding: Vec<f32>) -> Result<()> {
        if let Some(dim) = self.dimension {
            if embedding.len() != dim {
                return Err(Error::InvalidInput(format!(
                    "Dimension mismatch: expected {}, got {}",
                    dim,
                    embedding.len()
                )));
            }
        } else {
            self.dimension = Some(embedding.len());
        }

        self.data.push((cid, embedding));
        self.stats.data_points += 1;

        // Rebuild index if we have enough data
        if self.data.len().is_multiple_of(100) {
            self.rebuild()?;
        }

        Ok(())
    }

    /// Rebuild the learned index from scratch
    pub fn rebuild(&mut self) -> Result<()> {
        if self.data.is_empty() {
            return Ok(());
        }

        let dim = self
            .dimension
            .ok_or_else(|| Error::InvalidInput("No dimension set".to_string()))?;

        // Sort data by first dimension (simple heuristic)
        self.data.sort_by(|a, b| {
            a.1[0]
                .partial_cmp(&b.1[0])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Initialize models
        self.root_model = Some(Model::new(self.config.model_type, dim));
        self.stage1_models = (0..self.config.num_models)
            .map(|_| Model::new(self.config.model_type, dim))
            .collect();

        // Train models (simplified - real implementation would use proper training)
        self.train_models()?;

        Ok(())
    }

    fn train_models(&mut self) -> Result<()> {
        if self.data.is_empty() {
            return Ok(());
        }

        let n = self.data.len();

        // Prepare training data for root model
        let mut root_training_data = Vec::new();
        for (i, (_cid, embedding)) in self.data.iter().enumerate() {
            let normalized_pos = i as f32 / n as f32;
            let normalized_embedding = self.normalize_embedding(embedding);
            root_training_data.push((normalized_embedding, normalized_pos));
        }

        // Train root model
        if let Some(ref mut root) = self.root_model {
            root.train(
                &root_training_data,
                self.config.learning_rate,
                self.config.training_iterations,
            );
        }

        // Train stage 1 models (each responsible for a range)
        let chunk_size = n / self.config.num_models;

        // First, collect all training data for all models
        let mut all_model_training_data = Vec::new();
        for model_idx in 0..self.config.num_models {
            let start = model_idx * chunk_size;
            let end = if model_idx == self.config.num_models - 1 {
                n
            } else {
                (model_idx + 1) * chunk_size
            };

            let mut model_training_data = Vec::new();
            for i in start..end {
                if let Some((_cid, embedding)) = self.data.get(i) {
                    let local_pos = (i - start) as f32 / (end - start) as f32;
                    let normalized_embedding = self.normalize_embedding(embedding);
                    model_training_data.push((normalized_embedding, local_pos));
                }
            }
            all_model_training_data.push(model_training_data);
        }

        // Now train all models with their respective data
        for (model, training_data) in self
            .stage1_models
            .iter_mut()
            .zip(all_model_training_data.iter())
        {
            if !training_data.is_empty() {
                model.train(
                    training_data,
                    self.config.learning_rate,
                    self.config.training_iterations,
                );
            }
        }

        Ok(())
    }

    fn normalize_embedding(&self, embedding: &[f32]) -> Vec<f32> {
        // Simple min-max normalization to [0, 1]
        let min = embedding.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = embedding.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;

        if range > 1e-6 {
            embedding.iter().map(|&x| (x - min) / range).collect()
        } else {
            vec![0.5; embedding.len()]
        }
    }

    /// Search for k nearest neighbors
    pub fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<(Cid, f32)>> {
        if self.data.is_empty() {
            return Ok(Vec::new());
        }

        let dim = self
            .dimension
            .ok_or_else(|| Error::InvalidInput("No dimension set".to_string()))?;

        if query.len() != dim {
            return Err(Error::InvalidInput(format!(
                "Dimension mismatch: expected {}, got {}",
                dim,
                query.len()
            )));
        }

        // Rebuild index if not built yet
        if self.root_model.is_none() {
            self.rebuild()?;
        }

        self.stats.searches += 1;

        // Use learned index to predict position
        let predicted_pos = self.predict_position(query)?;
        let n = self.data.len();
        let start_idx = (predicted_pos * n as f32) as usize;

        // Search around predicted position (adaptive window)
        let window_size = (n as f32 * self.config.error_threshold).max(k as f32 * 2.0) as usize;
        let search_start = start_idx.saturating_sub(window_size / 2);
        let search_end = (start_idx + window_size / 2).min(n);

        // Find k nearest neighbors in the search window
        let mut candidates = Vec::new();
        for i in search_start..search_end {
            if let Some((cid, embedding)) = self.data.get(i) {
                let distance = self.compute_distance(query, embedding);
                candidates.push((*cid, distance));
            }
        }

        // Sort by distance and return top k
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(candidates.into_iter().take(k).collect())
    }

    fn predict_position(&mut self, query: &[f32]) -> Result<f32> {
        let normalized_query = self.normalize_embedding(query);

        // Stage 0: Root model predicts which stage 1 model to use
        let root_prediction = if let Some(ref root) = self.root_model {
            root.predict(&normalized_query)
        } else {
            return Err(Error::InvalidInput("No root model".to_string()));
        };

        // Select stage 1 model
        let model_idx = ((root_prediction * self.config.num_models as f32) as usize)
            .min(self.config.num_models - 1);

        // Stage 1: Selected model predicts position within its range
        let local_prediction = if let Some(model) = self.stage1_models.get(model_idx) {
            model.predict(&normalized_query)
        } else {
            0.5
        };

        // Combine predictions
        let chunk_size = 1.0 / self.config.num_models as f32;
        let final_prediction = model_idx as f32 * chunk_size + local_prediction * chunk_size;

        Ok(final_prediction.clamp(0.0, 1.0))
    }

    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        // L2 distance
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    /// Get index statistics
    pub fn stats(&self) -> LearnedIndexStats {
        LearnedIndexStats {
            data_points: self.stats.data_points,
            searches: self.stats.searches,
            num_models: self.stage1_models.len() + 1,
            avg_error: if self.stats.searches > 0 {
                self.stats.total_error / self.stats.searches as f32
            } else {
                0.0
            },
        }
    }

    /// Get number of data points
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Clear the index
    pub fn clear(&mut self) {
        self.data.clear();
        self.root_model = None;
        self.stage1_models.clear();
        self.stats = IndexStats::default();
    }
}

/// Statistics for the learned index
#[derive(Debug, Clone)]
pub struct LearnedIndexStats {
    /// Number of data points indexed
    pub data_points: usize,
    /// Number of searches performed
    pub searches: usize,
    /// Total number of models (root + stage 1)
    pub num_models: usize,
    /// Average prediction error
    pub avg_error: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learned_index_creation() {
        let index = LearnedIndex::new(RMIConfig::default());
        assert_eq!(index.size(), 0);
    }

    #[test]
    fn test_add_and_search() {
        let mut index = LearnedIndex::new(RMIConfig::default());

        // Add some embeddings
        for i in 0..100 {
            let cid = Cid::default();
            let embedding = vec![i as f32 / 100.0, 0.5, 0.5, 0.5];
            index
                .add(cid, embedding)
                .expect("test: add embedding to learned index");
        }

        assert_eq!(index.size(), 100);

        // Search
        let query = vec![0.5, 0.5, 0.5, 0.5];
        let results = index.search(&query, 5).expect("test: search learned index");
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_model_prediction() {
        let model = Model::new(ModelType::Linear, 4);
        let input = vec![0.1, 0.2, 0.3, 0.4];
        let prediction = model.predict(&input);
        assert!((0.0..=1.0).contains(&prediction));
    }

    #[test]
    fn test_polynomial_model() {
        let model = Model::new(ModelType::Polynomial, 4);
        let input = vec![0.5, 0.5, 0.5, 0.5];
        let prediction = model.predict(&input);
        assert!((0.0..=1.0).contains(&prediction));
    }

    #[test]
    fn test_neural_model() {
        let model = Model::new(ModelType::NeuralNetwork, 4);
        let input = vec![0.3, 0.4, 0.5, 0.6];
        let prediction = model.predict(&input);
        assert!((0.0..=1.0).contains(&prediction));
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut index = LearnedIndex::new(RMIConfig::default());

        let cid1 = Cid::default();
        index
            .add(cid1, vec![1.0, 2.0, 3.0])
            .expect("test: add first embedding");

        let cid2 = Cid::default();
        let result = index.add(cid2, vec![1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_rebuild_index() {
        let mut index = LearnedIndex::new(RMIConfig::default());

        for i in 0..50 {
            let cid = Cid::default();
            let embedding = vec![i as f32, 0.0, 0.0];
            index
                .add(cid, embedding)
                .expect("test: add embedding for rebuild");
        }

        index.rebuild().expect("test: rebuild index");

        let query = vec![25.0, 0.0, 0.0];
        let results = index.search(&query, 3).expect("test: search after rebuild");
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_stats() {
        let mut index = LearnedIndex::new(RMIConfig::default());

        for i in 0..10 {
            let cid = Cid::default();
            index
                .add(cid, vec![i as f32, 0.0])
                .expect("test: add embedding for stats");
        }

        let query = vec![5.0, 0.0];
        let _ = index.search(&query, 3).expect("test: search for stats");

        let stats = index.stats();
        assert_eq!(stats.data_points, 10);
        assert_eq!(stats.searches, 1);
    }

    #[test]
    fn test_clear() {
        let mut index = LearnedIndex::new(RMIConfig::default());

        let cid = Cid::default();
        index
            .add(cid, vec![1.0, 2.0, 3.0])
            .expect("test: add embedding for clear");
        assert_eq!(index.size(), 1);

        index.clear();
        assert_eq!(index.size(), 0);
    }

    #[test]
    fn test_config_variants() {
        let configs = vec![
            RMIConfig {
                model_type: ModelType::Linear,
                ..Default::default()
            },
            RMIConfig {
                model_type: ModelType::Polynomial,
                ..Default::default()
            },
            RMIConfig {
                model_type: ModelType::NeuralNetwork,
                ..Default::default()
            },
        ];

        for config in configs {
            let mut index = LearnedIndex::new(config);
            for i in 0..20 {
                let cid = Cid::default();
                index
                    .add(cid, vec![i as f32, 0.0, 0.0])
                    .expect("test: add embedding for config variant");
            }

            let query = vec![10.0, 0.0, 0.0];
            let results = index
                .search(&query, 5)
                .expect("test: search for config variant");
            assert!(!results.is_empty());
        }
    }
}
