//! Index optimization utilities
//!
//! This module provides tools for optimizing index performance,
//! including automatic parameter tuning, query optimization, and resource management.

use std::time::Duration;

/// Optimization goal for index tuning
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationGoal {
    /// Minimize query latency
    MinimizeLatency,
    /// Maximize recall/accuracy
    MaximizeRecall,
    /// Minimize memory usage
    MinimizeMemory,
    /// Balance between all factors
    Balanced,
}

/// Result of optimization analysis
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Current configuration quality score (0.0 - 1.0)
    pub current_score: f32,
    /// Recommended M parameter
    pub recommended_m: usize,
    /// Recommended ef_construction
    pub recommended_ef_construction: usize,
    /// Recommended ef_search
    pub recommended_ef_search: usize,
    /// Estimated improvement (0.0 - 1.0)
    pub estimated_improvement: f32,
    /// Reasoning for recommendations
    pub reasoning: Vec<String>,
}

/// Analyze index and provide optimization recommendations
pub fn analyze_optimization(
    index_size: usize,
    dimension: usize,
    current_m: usize,
    current_ef_construction: usize,
    goal: OptimizationGoal,
) -> OptimizationResult {
    let mut reasoning = Vec::new();

    // Compute recommended parameters based on goal
    let (recommended_m, recommended_ef_construction, recommended_ef_search) = match goal {
        OptimizationGoal::MinimizeLatency => {
            reasoning
                .push("Optimizing for low latency with reduced graph connectivity".to_string());
            let m = if index_size < 10_000 {
                12
            } else if index_size < 100_000 {
                16
            } else {
                20
            };
            (m, 150, 32)
        }
        OptimizationGoal::MaximizeRecall => {
            reasoning
                .push("Optimizing for high recall with increased graph connectivity".to_string());
            let m = if index_size < 10_000 {
                32
            } else if index_size < 100_000 {
                48
            } else {
                64
            };
            (m, 400, 200)
        }
        OptimizationGoal::MinimizeMemory => {
            reasoning.push("Optimizing for low memory with minimal graph connectivity".to_string());
            (8, 100, 50)
        }
        OptimizationGoal::Balanced => {
            reasoning.push("Balanced optimization for general use cases".to_string());
            let m = if index_size < 10_000 {
                16
            } else if index_size < 100_000 {
                24
            } else {
                32
            };
            let ef_c = if index_size < 10_000 { 200 } else { 300 };
            (m, ef_c, 100)
        }
    };

    // Evaluate current configuration
    let current_score =
        evaluate_config_quality(current_m, current_ef_construction, index_size, goal);

    // Evaluate recommended configuration
    let recommended_score =
        evaluate_config_quality(recommended_m, recommended_ef_construction, index_size, goal);

    let estimated_improvement = (recommended_score - current_score).max(0.0);

    // Add specific recommendations
    if current_m < recommended_m {
        reasoning.push(format!(
            "Increase M from {} to {} for better connectivity",
            current_m, recommended_m
        ));
    } else if current_m > recommended_m {
        reasoning.push(format!(
            "Decrease M from {} to {} to reduce memory usage",
            current_m, recommended_m
        ));
    }

    if dimension > 1024 {
        reasoning
            .push("High dimensionality detected. Consider dimensionality reduction.".to_string());
    }

    if index_size > 1_000_000 {
        reasoning.push("Large index detected. Consider using DiskANN or partitioning.".to_string());
    }

    OptimizationResult {
        current_score,
        recommended_m,
        recommended_ef_construction,
        recommended_ef_search,
        estimated_improvement,
        reasoning,
    }
}

/// Evaluate configuration quality for a given goal
fn evaluate_config_quality(
    m: usize,
    ef_construction: usize,
    index_size: usize,
    goal: OptimizationGoal,
) -> f32 {
    let optimal_m = match index_size {
        0..=10_000 => 16,
        10_001..=100_000 => 24,
        _ => 32,
    };

    let optimal_ef_c = match index_size {
        0..=10_000 => 200,
        10_001..=100_000 => 300,
        _ => 400,
    };

    // Compute distance from optimal for this index size
    let m_score = 1.0 - ((m as f32 - optimal_m as f32).abs() / optimal_m as f32).min(1.0);
    let ef_score =
        1.0 - ((ef_construction as f32 - optimal_ef_c as f32).abs() / optimal_ef_c as f32).min(1.0);

    // Weight scores based on goal
    match goal {
        OptimizationGoal::MinimizeLatency => {
            // Prefer lower M and ef_construction
            let latency_penalty = (m as f32 / 32.0).min(1.0) * 0.5;
            (m_score * 0.3 + ef_score * 0.7) * (1.0 - latency_penalty)
        }
        OptimizationGoal::MaximizeRecall => {
            // Prefer higher M and ef_construction
            let recall_bonus = (m as f32 / 64.0).min(1.0) * 0.3;
            (m_score * 0.7 + ef_score * 0.3) * (1.0 + recall_bonus)
        }
        OptimizationGoal::MinimizeMemory => {
            // Strongly prefer lower M
            let memory_penalty = (m as f32 / 16.0).min(1.0) * 0.7;
            m_score * (1.0 - memory_penalty)
        }
        OptimizationGoal::Balanced => {
            // Equal weight
            (m_score + ef_score) / 2.0
        }
    }
}

/// Query optimizer for adaptive ef_search selection
pub struct QueryOptimizer {
    /// Query performance history
    latency_samples: Vec<Duration>,
    /// Maximum samples to keep
    max_samples: usize,
    /// Current ef_search value
    current_ef_search: usize,
    /// Minimum ef_search
    min_ef_search: usize,
    /// Maximum ef_search
    max_ef_search: usize,
    /// Target latency
    target_latency: Duration,
}

impl QueryOptimizer {
    /// Create a new query optimizer
    pub fn new(initial_ef_search: usize, target_latency: Duration) -> Self {
        Self {
            latency_samples: Vec::new(),
            max_samples: 100,
            current_ef_search: initial_ef_search,
            min_ef_search: 16,
            max_ef_search: 512,
            target_latency,
        }
    }

    /// Record a query latency and adjust ef_search if needed
    pub fn record_query(&mut self, latency: Duration) {
        self.latency_samples.push(latency);
        if self.latency_samples.len() > self.max_samples {
            self.latency_samples.remove(0);
        }

        // Adjust ef_search based on recent performance
        if self.latency_samples.len() >= 10 {
            self.adjust_ef_search();
        }
    }

    /// Get current recommended ef_search
    pub fn get_ef_search(&self) -> usize {
        self.current_ef_search
    }

    /// Adjust ef_search based on observed latency
    fn adjust_ef_search(&mut self) {
        let avg_latency =
            self.latency_samples.iter().sum::<Duration>() / self.latency_samples.len() as u32;

        if avg_latency > self.target_latency {
            // Too slow, decrease ef_search
            self.current_ef_search = (self.current_ef_search * 9 / 10).max(self.min_ef_search);
        } else if avg_latency < self.target_latency / 2 {
            // Too fast, we can afford to increase ef_search for better recall
            self.current_ef_search = (self.current_ef_search * 11 / 10).min(self.max_ef_search);
        }
    }

    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.latency_samples.clear();
    }

    /// Get average latency from recent queries
    pub fn avg_latency(&self) -> Option<Duration> {
        if self.latency_samples.is_empty() {
            None
        } else {
            Some(self.latency_samples.iter().sum::<Duration>() / self.latency_samples.len() as u32)
        }
    }
}

/// Memory optimizer for managing index memory usage
pub struct MemoryOptimizer {
    /// Target memory budget in bytes
    target_memory: usize,
    /// Estimated memory per vector
    memory_per_vector: usize,
}

impl MemoryOptimizer {
    /// Create a new memory optimizer
    pub fn new(target_memory: usize) -> Self {
        Self {
            target_memory,
            memory_per_vector: 0,
        }
    }

    /// Estimate memory usage for an index configuration
    pub fn estimate_memory(&mut self, num_vectors: usize, dimension: usize, m: usize) -> usize {
        // Vector storage: dimension * 4 bytes per f32
        let vector_memory = num_vectors * dimension * 4;

        // HNSW graph: approximately (M * 2) * num_vectors * 8 bytes for node IDs
        let graph_memory = num_vectors * m * 2 * 8;

        // Metadata overhead (mappings, etc.)
        let overhead = num_vectors * 100;

        let total = vector_memory + graph_memory + overhead;

        self.memory_per_vector = total.checked_div(num_vectors).unwrap_or(0);

        total
    }

    /// Check if adding more vectors would exceed budget
    pub fn can_add_vectors(&self, num_new_vectors: usize) -> bool {
        let estimated_additional = num_new_vectors * self.memory_per_vector;
        estimated_additional <= self.target_memory
    }

    /// Get maximum vectors that can fit in budget
    pub fn max_vectors(&self, dimension: usize, m: usize) -> usize {
        self.target_memory
            .checked_div(self.memory_per_vector)
            .unwrap_or_else(|| {
                // memory_per_vector is 0: fall back to first estimate
                let bytes_per_vector = dimension * 4 + m * 2 * 8 + 100;
                self.target_memory / bytes_per_vector
            })
    }

    /// Recommend configuration for memory budget
    pub fn recommend_config(&self, dimension: usize) -> (usize, usize, usize) {
        // Try different M values to maximize vectors within budget
        for m in [8, 12, 16, 24, 32, 48, 64].iter().rev() {
            let bytes_per_vector = dimension * 4 + m * 2 * 8 + 100;
            let max_vectors = self.target_memory / bytes_per_vector;

            if max_vectors >= 1000 {
                // Found a viable configuration
                let ef_construction = if max_vectors < 10_000 {
                    200
                } else if max_vectors < 100_000 {
                    300
                } else {
                    400
                };
                return (*m, ef_construction, max_vectors);
            }
        }

        // Minimum configuration
        (
            8,
            100,
            self.target_memory / (dimension * 4 + 8 * 2 * 8 + 100),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_optimization_latency() {
        let result = analyze_optimization(10_000, 768, 16, 200, OptimizationGoal::MinimizeLatency);

        assert!(result.recommended_m <= 16);
        assert!(result.recommended_ef_search <= 50);
        assert!(!result.reasoning.is_empty());
    }

    #[test]
    fn test_analyze_optimization_recall() {
        let result = analyze_optimization(10_000, 768, 16, 200, OptimizationGoal::MaximizeRecall);

        assert!(result.recommended_m >= 16);
        assert!(result.recommended_ef_construction >= 200);
        assert!(!result.reasoning.is_empty());
    }

    #[test]
    fn test_query_optimizer() {
        let mut optimizer = QueryOptimizer::new(50, Duration::from_millis(10));

        // Record some fast queries
        for _ in 0..15 {
            optimizer.record_query(Duration::from_millis(2));
        }

        // Should increase ef_search since we're under target
        assert!(optimizer.get_ef_search() > 50);
    }

    #[test]
    fn test_query_optimizer_slow_queries() {
        let mut optimizer = QueryOptimizer::new(50, Duration::from_millis(10));

        // Record some slow queries
        for _ in 0..15 {
            optimizer.record_query(Duration::from_millis(20));
        }

        // Should decrease ef_search since we're over target
        assert!(optimizer.get_ef_search() < 50);
    }

    #[test]
    fn test_memory_optimizer() {
        let mut optimizer = MemoryOptimizer::new(1024 * 1024 * 1024); // 1GB

        let memory = optimizer.estimate_memory(10_000, 768, 16);
        assert!(memory > 0);
        assert!(memory <= 1024 * 1024 * 1024);
    }

    #[test]
    fn test_memory_optimizer_recommend() {
        let optimizer = MemoryOptimizer::new(1024 * 1024 * 1024); // 1GB

        let (m, ef_c, max_vecs) = optimizer.recommend_config(768);

        assert!(m >= 8);
        assert!(ef_c >= 100);
        assert!(max_vecs > 0);
    }
}
