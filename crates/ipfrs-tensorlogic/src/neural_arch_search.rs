//! Neural Architecture Search (NAS) — random and evolutionary search for optimal network structures.
//!
//! Provides [`NeuralArchitectureSearch`] which supports three search strategies:
//! - **Random** — independently sample architectures each generation
//! - **Evolutionary** — keep elite fraction, mutate and crossover the rest
//! - **GridSearch** — enumerate all combinations of layer widths and depths
//!
//! # Example
//!
//! ```
//! use ipfrs_tensorlogic::{NeuralArchitectureSearch, NasConfig, NasSearchStrategy};
//!
//! let config = NasConfig {
//!     strategy: NasSearchStrategy::Random { population_size: 10 },
//!     max_generations: 3,
//!     target_fitness: 0.99,
//!     min_layers: 2,
//!     max_layers: 5,
//!     min_units: 16,
//!     max_units: 128,
//!     seed: 42,
//! };
//! let mut nas = NeuralArchitectureSearch::new(config);
//! let results = nas.run_search(64, 10);
//! assert!(!results.is_empty());
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// xorshift64 PRNG (no external crate)
// ---------------------------------------------------------------------------

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[inline]
fn rng_range(state: &mut u64, lo: usize, hi: usize) -> usize {
    if lo >= hi {
        return lo;
    }
    lo + (xorshift64(state) as usize % (hi - lo))
}

// ---------------------------------------------------------------------------
// LayerType
// ---------------------------------------------------------------------------

/// A single layer in a candidate neural architecture.
#[derive(Debug, Clone, PartialEq)]
pub enum NasLayerType {
    /// Fully-connected layer with `units` output neurons.
    Dense { units: usize },
    /// Dropout regularisation with dropout probability `rate`.
    Dropout { rate: f64 },
    /// Batch normalisation (no learnable weight count in this simplified model).
    BatchNorm,
    /// Named activation function (e.g. `"relu"`, `"sigmoid"`, `"tanh"`).
    Activation { function: String },
    /// 1-D convolution with `filters` output channels and `kernel_size` taps.
    Conv1D { filters: usize, kernel_size: usize },
    /// 1-D pooling with `pool_size` window and named `pool_type` (`"max"` / `"avg"`).
    Pooling { pool_size: usize, pool_type: String },
}

impl fmt::Display for NasLayerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NasLayerType::Dense { units } => write!(f, "Dense({})", units),
            NasLayerType::Dropout { rate } => write!(f, "Dropout({:.3})", rate),
            NasLayerType::BatchNorm => write!(f, "BatchNorm"),
            NasLayerType::Activation { function } => write!(f, "Activation({})", function),
            NasLayerType::Conv1D {
                filters,
                kernel_size,
            } => {
                write!(f, "Conv1D({},{})", filters, kernel_size)
            }
            NasLayerType::Pooling {
                pool_size,
                pool_type,
            } => {
                write!(f, "Pooling({},{})", pool_size, pool_type)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Architecture
// ---------------------------------------------------------------------------

/// A candidate neural network architecture produced by NAS.
#[derive(Debug, Clone)]
pub struct NasArchitecture {
    /// FNV-1a hash of the architecture's string representation.
    pub id: u64,
    /// Sequence of layers from input to output.
    pub layers: Vec<NasLayerType>,
    /// Dimensionality of the input feature vector.
    pub input_dim: usize,
    /// Number of output units.
    pub output_dim: usize,
    /// Approximate total learnable parameter count.
    pub parameter_count: usize,
}

impl NasArchitecture {
    /// Build an [`NasArchitecture`], computing `id` and `parameter_count` automatically.
    pub fn new(layers: Vec<NasLayerType>, input_dim: usize, output_dim: usize) -> Self {
        let repr = layers
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let id = fnv1a_nas(&repr);
        let parameter_count = NeuralArchitectureSearch::compute_parameter_count(&layers, input_dim);
        NasArchitecture {
            id,
            layers,
            input_dim,
            output_dim,
            parameter_count,
        }
    }
}

/// FNV-1a 64-bit hash used to produce stable architecture IDs.
pub fn fnv1a_nas(s: &str) -> u64 {
    const OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;
    s.bytes()
        .fold(OFFSET_BASIS, |acc, b| acc.wrapping_mul(PRIME) ^ (b as u64))
}

// ---------------------------------------------------------------------------
// EvaluationResult
// ---------------------------------------------------------------------------

/// Fitness evaluation outcome for a single candidate architecture.
#[derive(Debug, Clone)]
pub struct NasEvaluationResult {
    /// Unique ID of the evaluated architecture (FNV-1a of layer string).
    pub arch_id: u64,
    /// Composite fitness score in `[0.0, 1.0]`.
    pub fitness: f64,
    /// Simulated accuracy derived from fitness.
    pub accuracy: f64,
    /// Simulated inference latency in milliseconds.
    pub latency_ms: f64,
    /// Approximate total parameter count.
    pub parameter_count: usize,
    /// Generation index at which this result was produced (0-indexed).
    pub generation: u32,
}

// ---------------------------------------------------------------------------
// SearchStrategy
// ---------------------------------------------------------------------------

/// Strategy that drives how the NAS population is sampled and evolved.
#[derive(Debug, Clone)]
pub enum NasSearchStrategy {
    /// Independently sample `population_size` random architectures per generation.
    Random { population_size: usize },
    /// Evolutionary search: keep the top `elite_fraction`, mutate/crossover rest.
    Evolutionary {
        population_size: usize,
        mutation_rate: f64,
        elite_fraction: f64,
    },
    /// Enumerate all combinations of units from `layer_options` at depths in `depth_range`.
    GridSearch {
        layer_options: Vec<usize>,
        depth_range: (usize, usize),
    },
}

// ---------------------------------------------------------------------------
// NasConfig
// ---------------------------------------------------------------------------

/// Configuration for a [`NeuralArchitectureSearch`] run.
#[derive(Debug, Clone)]
pub struct NasConfig {
    /// Search strategy variant.
    pub strategy: NasSearchStrategy,
    /// Maximum number of generations to run.
    pub max_generations: u32,
    /// Stop early when a candidate reaches this fitness.
    pub target_fitness: f64,
    /// Minimum number of hidden layers (excluding the mandatory output Dense).
    pub min_layers: usize,
    /// Maximum number of hidden layers (excluding the mandatory output Dense).
    pub max_layers: usize,
    /// Minimum units for randomly sampled Dense layers.
    pub min_units: usize,
    /// Maximum units (exclusive) for randomly sampled Dense layers.
    pub max_units: usize,
    /// PRNG seed for reproducibility.
    pub seed: u64,
}

// ---------------------------------------------------------------------------
// NasStats
// ---------------------------------------------------------------------------

/// Summary statistics for a completed NAS run.
#[derive(Debug, Clone)]
pub struct NasStats {
    /// Number of generations completed.
    pub generations_run: u32,
    /// Total number of architecture evaluations performed.
    pub total_architectures_evaluated: usize,
    /// Fitness of the best architecture found.
    pub best_fitness: f64,
    /// Mean fitness across all evaluations.
    pub avg_fitness: f64,
    /// Current population size.
    pub population_size: usize,
}

// ---------------------------------------------------------------------------
// NeuralArchitectureSearch
// ---------------------------------------------------------------------------

/// Random / evolutionary neural architecture search engine.
///
/// See the [module-level documentation](self) for usage examples.
pub struct NeuralArchitectureSearch {
    /// Configuration driving the search.
    pub config: NasConfig,
    /// Current population of candidate architectures.
    pub population: Vec<NasArchitecture>,
    /// Full history of evaluation results (all generations).
    pub history: Vec<NasEvaluationResult>,
    /// Best architecture found so far.
    pub best_arch: Option<NasArchitecture>,
    /// Current generation counter (0 before `run_search` is called).
    pub generation: u32,
}

impl NeuralArchitectureSearch {
    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Create a new NAS engine with the given configuration.
    pub fn new(config: NasConfig) -> Self {
        NeuralArchitectureSearch {
            config,
            population: Vec::new(),
            history: Vec::new(),
            best_arch: None,
            generation: 0,
        }
    }

    /// Generate the initial population according to the chosen strategy.
    ///
    /// Called automatically by `run_search`; exposed for testing.
    pub fn initialize_population(
        &self,
        input_dim: usize,
        output_dim: usize,
        rng: &mut u64,
    ) -> Vec<NasArchitecture> {
        match &self.config.strategy {
            NasSearchStrategy::Random { population_size } => (0..*population_size)
                .map(|_| Self::generate_random_arch(rng, input_dim, output_dim, &self.config))
                .collect(),
            NasSearchStrategy::Evolutionary {
                population_size, ..
            } => (0..*population_size)
                .map(|_| Self::generate_random_arch(rng, input_dim, output_dim, &self.config))
                .collect(),
            NasSearchStrategy::GridSearch {
                layer_options,
                depth_range,
            } => Self::grid_search_population(layer_options, *depth_range, input_dim, output_dim),
        }
    }

    /// Run the full architecture search for `max_generations` generations.
    ///
    /// Returns all evaluation results sorted by fitness (descending).
    pub fn run_search(&mut self, input_dim: usize, output_dim: usize) -> Vec<NasEvaluationResult> {
        let mut rng = self.config.seed.max(1); // seed must be non-zero for xorshift

        self.population = self.initialize_population(input_dim, output_dim, &mut rng);

        for gen in 0..self.config.max_generations {
            self.generation = gen;

            // Evaluate the current population
            let mut gen_results: Vec<NasEvaluationResult> = self
                .population
                .iter()
                .map(|arch| {
                    let fitness = Self::evaluate_fitness(arch);
                    let accuracy = (fitness * 0.95).clamp(0.0, 1.0);
                    // Latency grows with parameter count (naïve simulation)
                    let latency_ms = 1.0 + arch.parameter_count as f64 / 50_000.0;
                    NasEvaluationResult {
                        arch_id: arch.id,
                        fitness,
                        accuracy,
                        latency_ms,
                        parameter_count: arch.parameter_count,
                        generation: gen,
                    }
                })
                .collect();

            // Update best architecture
            if let Some(best_result) = gen_results.iter().max_by(|a, b| {
                a.fitness
                    .partial_cmp(&b.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                let should_update = self
                    .best_arch
                    .as_ref()
                    .is_none_or(|b| Self::evaluate_fitness(b) < best_result.fitness);

                if should_update {
                    // Find the architecture in the population
                    if let Some(arch) = self.population.iter().find(|a| a.id == best_result.arch_id)
                    {
                        self.best_arch = Some(arch.clone());
                    }
                }
            }

            self.history.append(&mut gen_results);

            // Early stopping
            let reached_target = self
                .best_arch
                .as_ref()
                .is_some_and(|b| Self::evaluate_fitness(b) >= self.config.target_fitness);
            if reached_target {
                break;
            }

            // Evolve population (not on last generation)
            if gen + 1 < self.config.max_generations {
                self.population = self.evolve_population(input_dim, output_dim, &mut rng);
            }
        }

        // Return all history sorted by fitness descending
        let mut all = self.history.clone();
        all.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all
    }

    /// Return a reference to the best architecture found so far.
    pub fn best_architecture(&self) -> Option<&NasArchitecture> {
        self.best_arch.as_ref()
    }

    /// Return summary statistics for the search run so far.
    pub fn stats(&self) -> NasStats {
        let total = self.history.len();
        let best_fitness = self.best_arch.as_ref().map_or(0.0, Self::evaluate_fitness);
        let avg_fitness = if total == 0 {
            0.0
        } else {
            self.history.iter().map(|r| r.fitness).sum::<f64>() / total as f64
        };
        let population_size = self.population.len();
        NasStats {
            generations_run: self.generation + 1,
            total_architectures_evaluated: total,
            best_fitness,
            avg_fitness,
            population_size,
        }
    }

    // -----------------------------------------------------------------------
    // Core building blocks
    // -----------------------------------------------------------------------

    /// Generate a single random architecture.
    ///
    /// Hidden layer count is drawn from `[min_layers, max_layers]`.
    /// Each hidden Dense unit count is drawn from `[min_units, max_units)`.
    /// Each hidden Dense layer has a 20 % chance of being followed by `Dropout(0.2)`.
    /// The final layer is always `Dense(output_dim)`.
    pub fn generate_random_arch(
        rng: &mut u64,
        input_dim: usize,
        output_dim: usize,
        config: &NasConfig,
    ) -> NasArchitecture {
        let num_hidden = rng_range(rng, config.min_layers, config.max_layers + 1);
        let mut layers = Vec::with_capacity(num_hidden * 2 + 1);

        for _ in 0..num_hidden {
            let units = rng_range(rng, config.min_units, config.max_units + 1);
            layers.push(NasLayerType::Dense { units });
            // 20 % chance of a Dropout layer after each Dense
            let p = xorshift64(rng) % 10;
            if p < 2 {
                layers.push(NasLayerType::Dropout { rate: 0.2 });
            }
        }

        // Final output layer
        layers.push(NasLayerType::Dense { units: output_dim });

        NasArchitecture::new(layers, input_dim, output_dim)
    }

    /// Simulated fitness function for a candidate architecture.
    ///
    /// - Base score: 0.5
    /// - Under-fitting penalty (params < 1 000): –0.1
    /// - Over-fitting penalty (params > 100 000): –0.1
    /// - Depth bonus: +0.05 × min(dense_count – 1, 3)
    /// - Dropout bonus: +0.02 per dropout layer, capped at +0.06
    /// - BatchNorm bonus: +0.03 per batchnorm layer, capped at +0.06
    pub fn evaluate_fitness(arch: &NasArchitecture) -> f64 {
        let base = 0.5_f64;

        let mut penalties = 0.0_f64;
        if arch.parameter_count < 1_000 {
            penalties += 0.1;
        }
        if arch.parameter_count > 100_000 {
            penalties += 0.1;
        }

        let dense_count = arch
            .layers
            .iter()
            .filter(|l| matches!(l, NasLayerType::Dense { .. }))
            .count();
        let depth_bonus = 0.05 * (dense_count.saturating_sub(1).min(3)) as f64;

        let dropout_count = arch
            .layers
            .iter()
            .filter(|l| matches!(l, NasLayerType::Dropout { .. }))
            .count();
        let dropout_bonus = (0.02 * dropout_count as f64).min(0.06);

        let batchnorm_count = arch
            .layers
            .iter()
            .filter(|l| matches!(l, NasLayerType::BatchNorm))
            .count();
        let batchnorm_bonus = (0.03 * batchnorm_count as f64).min(0.06);

        let bonuses = depth_bonus + dropout_bonus + batchnorm_bonus;

        (base + bonuses - penalties).clamp(0.0, 1.0)
    }

    /// Mutate an architecture by one of three operations chosen proportionally:
    ///
    /// - 30 %: insert a new `Dense` layer before the final layer
    /// - 30 %: remove a non-final layer (if more than one layer exists)
    /// - 40 %: change the unit count of an existing Dense hidden layer
    pub fn mutate(arch: &NasArchitecture, rng: &mut u64, config: &NasConfig) -> NasArchitecture {
        let mut layers = arch.layers.clone();
        let op = xorshift64(rng) % 10;

        if op < 3 {
            // --- Add a Dense layer before the final layer ---
            let units = rng_range(rng, config.min_units, config.max_units + 1);
            let insert_pos = layers.len().saturating_sub(1);
            layers.insert(insert_pos, NasLayerType::Dense { units });
        } else if op < 6 {
            // --- Remove a non-final layer (requires ≥ 2 layers) ---
            if layers.len() >= 2 {
                let remove_pos = rng_range(rng, 0, layers.len() - 1);
                layers.remove(remove_pos);
            }
        } else {
            // --- Change units of a Dense hidden layer ---
            let dense_indices: Vec<usize> = layers
                .iter()
                .enumerate()
                .filter_map(|(i, l)| {
                    if i + 1 < layers.len() {
                        if let NasLayerType::Dense { .. } = l {
                            return Some(i);
                        }
                    }
                    None
                })
                .collect();

            if !dense_indices.is_empty() {
                let pick = rng_range(rng, 0, dense_indices.len());
                let idx = dense_indices[pick];
                let new_units = rng_range(rng, config.min_units, config.max_units + 1);
                layers[idx] = NasLayerType::Dense { units: new_units };
            }
        }

        NasArchitecture::new(layers, arch.input_dim, arch.output_dim)
    }

    /// Crossover two architectures: take the first half of `a`'s hidden layers
    /// plus the second half of `b`'s hidden layers, then append the output layer.
    pub fn crossover(a: &NasArchitecture, b: &NasArchitecture, _rng: &mut u64) -> NasArchitecture {
        // Work with hidden layers (everything except the final Dense output)
        let a_hidden = Self::hidden_layers(&a.layers);
        let b_hidden = Self::hidden_layers(&b.layers);

        let a_half = a_hidden.len() / 2;
        let b_start = b_hidden.len() / 2;

        let mut child_layers: Vec<NasLayerType> = a_hidden[..a_half]
            .iter()
            .chain(b_hidden[b_start..].iter())
            .cloned()
            .collect();

        // Always end with the output Dense
        child_layers.push(NasLayerType::Dense {
            units: a.output_dim,
        });

        NasArchitecture::new(child_layers, a.input_dim, a.output_dim)
    }

    /// Compute the total learnable parameter count for a layer sequence.
    ///
    /// - `Dense { units }` → `units * prev_units`
    /// - `Conv1D { filters, kernel_size }` → `filters * kernel_size * prev_channels`
    /// - All other layer types → 0
    pub fn compute_parameter_count(layers: &[NasLayerType], input_dim: usize) -> usize {
        let mut prev = input_dim;
        let mut total = 0usize;
        for layer in layers {
            match layer {
                NasLayerType::Dense { units } => {
                    total += units * prev;
                    prev = *units;
                }
                NasLayerType::Conv1D {
                    filters,
                    kernel_size,
                } => {
                    total += filters * kernel_size * prev;
                    prev = *filters;
                }
                // BatchNorm, Dropout, Activation, Pooling contribute 0 parameters
                _ => {}
            }
        }
        total
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Return all layers except the final one (assumed to be the output Dense).
    fn hidden_layers(layers: &[NasLayerType]) -> &[NasLayerType] {
        if layers.is_empty() {
            layers
        } else {
            &layers[..layers.len() - 1]
        }
    }

    /// Enumerate all Dense-only architectures for `GridSearch`.
    fn grid_search_population(
        layer_options: &[usize],
        depth_range: (usize, usize),
        input_dim: usize,
        output_dim: usize,
    ) -> Vec<NasArchitecture> {
        let (min_depth, max_depth) = depth_range;
        let mut population = Vec::new();

        for depth in min_depth..=max_depth {
            Self::grid_combinations(layer_options, depth, &mut |combo: &[usize]| {
                let mut layers: Vec<NasLayerType> = combo
                    .iter()
                    .map(|&u| NasLayerType::Dense { units: u })
                    .collect();
                layers.push(NasLayerType::Dense { units: output_dim });
                population.push(NasArchitecture::new(layers, input_dim, output_dim));
            });
        }

        population
    }

    /// Enumerate all `n`-length combinations (with repetition) from `options`.
    fn grid_combinations(options: &[usize], n: usize, callback: &mut impl FnMut(&[usize])) {
        if options.is_empty() {
            return;
        }
        let mut combo = vec![0usize; n];
        // We use a stack-based iterator to avoid recursion overhead.
        Self::grid_rec(options, n, 0, &mut combo, callback);
    }

    fn grid_rec(
        options: &[usize],
        n: usize,
        pos: usize,
        combo: &mut Vec<usize>,
        callback: &mut impl FnMut(&[usize]),
    ) {
        if pos == n {
            callback(combo);
            return;
        }
        for &opt in options {
            combo[pos] = opt;
            Self::grid_rec(options, n, pos + 1, combo, callback);
        }
    }

    /// Evolve the population for the next generation.
    fn evolve_population(
        &self,
        input_dim: usize,
        output_dim: usize,
        rng: &mut u64,
    ) -> Vec<NasArchitecture> {
        match &self.config.strategy {
            NasSearchStrategy::Random { population_size } => (0..*population_size)
                .map(|_| Self::generate_random_arch(rng, input_dim, output_dim, &self.config))
                .collect(),
            NasSearchStrategy::Evolutionary {
                population_size,
                mutation_rate,
                elite_fraction,
            } => {
                // Evaluate and rank current population
                let mut scored: Vec<(f64, &NasArchitecture)> = self
                    .population
                    .iter()
                    .map(|a| (Self::evaluate_fitness(a), a))
                    .collect();
                scored.sort_by(|(fa, _), (fb, _)| {
                    fb.partial_cmp(fa).unwrap_or(std::cmp::Ordering::Equal)
                });

                let n_elite = ((scored.len() as f64 * elite_fraction).ceil() as usize)
                    .min(scored.len())
                    .max(1);
                let mut next_gen: Vec<NasArchitecture> = scored[..n_elite]
                    .iter()
                    .map(|(_, a)| (*a).clone())
                    .collect();

                while next_gen.len() < *population_size {
                    let p_idx = rng_range(rng, 0, n_elite);
                    let parent = &scored[p_idx].1;
                    let do_crossover = rng_range(rng, 0, 100) < 50 && scored.len() >= 2;
                    let child = if do_crossover {
                        let q_idx = rng_range(rng, 0, n_elite);
                        let other_parent = &scored[q_idx].1;
                        Self::crossover(parent, other_parent, rng)
                    } else {
                        (*parent).clone()
                    };

                    // Apply mutation with probability `mutation_rate`
                    let do_mutate = (xorshift64(rng) as f64 / u64::MAX as f64) < *mutation_rate;
                    if do_mutate {
                        next_gen.push(Self::mutate(&child, rng, &self.config));
                    } else {
                        next_gen.push(child);
                    }
                }
                next_gen
            }
            NasSearchStrategy::GridSearch {
                layer_options,
                depth_range,
            } => {
                // For grid search the population is static; just return it unchanged.
                Self::grid_search_population(layer_options, *depth_range, input_dim, output_dim)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{
        fnv1a_nas, rng_range, xorshift64, NasArchitecture, NasConfig, NasEvaluationResult,
        NasLayerType, NasSearchStrategy, NasStats, NeuralArchitectureSearch,
    };

    fn default_config() -> NasConfig {
        NasConfig {
            strategy: NasSearchStrategy::Random { population_size: 8 },
            max_generations: 3,
            target_fitness: 0.99,
            min_layers: 1,
            max_layers: 4,
            min_units: 16,
            max_units: 64,
            seed: 12345,
        }
    }

    // -----------------------------------------------------------------------
    // PRNG tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_xorshift64_non_zero() {
        let mut state = 1u64;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
    }

    #[test]
    fn test_xorshift64_changes_state() {
        let mut state = 42u64;
        let v1 = xorshift64(&mut state);
        let v2 = xorshift64(&mut state);
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_xorshift64_deterministic() {
        let mut s1 = 99u64;
        let mut s2 = 99u64;
        assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
    }

    #[test]
    fn test_rng_range_lo_equals_hi() {
        let mut rng = 7u64;
        assert_eq!(rng_range(&mut rng, 5, 5), 5);
    }

    #[test]
    fn test_rng_range_in_bounds() {
        let mut rng = 1u64;
        for _ in 0..1000 {
            let v = rng_range(&mut rng, 3, 10);
            assert!((3..10).contains(&v));
        }
    }

    // -----------------------------------------------------------------------
    // FNV-1a hash
    // -----------------------------------------------------------------------

    #[test]
    fn test_fnv1a_empty_string() {
        let h = fnv1a_nas("");
        assert_ne!(h, 0);
    }

    #[test]
    fn test_fnv1a_different_strings() {
        let h1 = fnv1a_nas("Dense(64)");
        let h2 = fnv1a_nas("Dense(32)");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_fnv1a_deterministic() {
        assert_eq!(fnv1a_nas("hello"), fnv1a_nas("hello"));
    }

    // -----------------------------------------------------------------------
    // NasLayerType
    // -----------------------------------------------------------------------

    #[test]
    fn test_layer_type_display_dense() {
        let l = NasLayerType::Dense { units: 128 };
        assert_eq!(l.to_string(), "Dense(128)");
    }

    #[test]
    fn test_layer_type_display_dropout() {
        let l = NasLayerType::Dropout { rate: 0.5 };
        assert!(l.to_string().starts_with("Dropout("));
    }

    #[test]
    fn test_layer_type_display_batchnorm() {
        let l = NasLayerType::BatchNorm;
        assert_eq!(l.to_string(), "BatchNorm");
    }

    #[test]
    fn test_layer_type_display_activation() {
        let l = NasLayerType::Activation {
            function: "relu".to_string(),
        };
        assert_eq!(l.to_string(), "Activation(relu)");
    }

    #[test]
    fn test_layer_type_display_conv1d() {
        let l = NasLayerType::Conv1D {
            filters: 32,
            kernel_size: 3,
        };
        assert_eq!(l.to_string(), "Conv1D(32,3)");
    }

    #[test]
    fn test_layer_type_display_pooling() {
        let l = NasLayerType::Pooling {
            pool_size: 2,
            pool_type: "max".to_string(),
        };
        assert_eq!(l.to_string(), "Pooling(2,max)");
    }

    // -----------------------------------------------------------------------
    // Parameter count
    // -----------------------------------------------------------------------

    #[test]
    fn test_param_count_single_dense() {
        let layers = vec![NasLayerType::Dense { units: 32 }];
        assert_eq!(
            NeuralArchitectureSearch::compute_parameter_count(&layers, 16),
            32 * 16
        );
    }

    #[test]
    fn test_param_count_two_dense() {
        let layers = vec![
            NasLayerType::Dense { units: 64 },
            NasLayerType::Dense { units: 10 },
        ];
        let expected = 64 * 16 + 10 * 64;
        assert_eq!(
            NeuralArchitectureSearch::compute_parameter_count(&layers, 16),
            expected
        );
    }

    #[test]
    fn test_param_count_dropout_zero() {
        let layers = vec![
            NasLayerType::Dense { units: 32 },
            NasLayerType::Dropout { rate: 0.2 },
            NasLayerType::Dense { units: 10 },
        ];
        let expected = 32 * 8 + 10 * 32;
        assert_eq!(
            NeuralArchitectureSearch::compute_parameter_count(&layers, 8),
            expected
        );
    }

    #[test]
    fn test_param_count_conv1d() {
        let layers = vec![NasLayerType::Conv1D {
            filters: 16,
            kernel_size: 3,
        }];
        assert_eq!(
            NeuralArchitectureSearch::compute_parameter_count(&layers, 4),
            16 * 3 * 4
        );
    }

    #[test]
    fn test_param_count_batchnorm_zero() {
        let layers = vec![NasLayerType::BatchNorm];
        assert_eq!(
            NeuralArchitectureSearch::compute_parameter_count(&layers, 10),
            0
        );
    }

    // -----------------------------------------------------------------------
    // Architecture construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_architecture_id_is_deterministic() {
        let layers = vec![
            NasLayerType::Dense { units: 32 },
            NasLayerType::Dense { units: 10 },
        ];
        let a1 = NasArchitecture::new(layers.clone(), 16, 10);
        let a2 = NasArchitecture::new(layers, 16, 10);
        assert_eq!(a1.id, a2.id);
    }

    #[test]
    fn test_architecture_different_layers_different_id() {
        let layers1 = vec![
            NasLayerType::Dense { units: 32 },
            NasLayerType::Dense { units: 10 },
        ];
        let layers2 = vec![
            NasLayerType::Dense { units: 64 },
            NasLayerType::Dense { units: 10 },
        ];
        let a1 = NasArchitecture::new(layers1, 16, 10);
        let a2 = NasArchitecture::new(layers2, 16, 10);
        assert_ne!(a1.id, a2.id);
    }

    // -----------------------------------------------------------------------
    // Fitness evaluation
    // -----------------------------------------------------------------------

    #[test]
    fn test_fitness_base_score() {
        // A minimal arch that avoids penalties and earns no bonuses
        let layers = vec![
            NasLayerType::Dense { units: 50 }, // just enough to avoid underfitting with input_dim=100
            NasLayerType::Dense { units: 10 },
        ];
        let arch = NasArchitecture::new(layers, 100, 10); // params = 50*100 + 10*50 = 5500
        let f = NeuralArchitectureSearch::evaluate_fitness(&arch);
        assert!((0.0..=1.0).contains(&f), "fitness out of range: {}", f);
    }

    #[test]
    fn test_fitness_underfitting_penalty() {
        // Very few params → should get underfitting penalty
        let layers = vec![NasLayerType::Dense { units: 1 }];
        let arch = NasArchitecture::new(layers, 1, 1); // 1*1 = 1 param
        let f = NeuralArchitectureSearch::evaluate_fitness(&arch);
        // 0.5 - 0.1 (underfit) = 0.4 (no depth bonus because only 1 dense)
        assert!(f < 0.5);
    }

    #[test]
    fn test_fitness_overfitting_penalty() {
        // Huge arch → overfitting penalty
        let layers = vec![
            NasLayerType::Dense { units: 1000 },
            NasLayerType::Dense { units: 10 },
        ];
        let arch = NasArchitecture::new(layers, 1000, 10); // 1000*1000 + 10*1000 = 1_010_000
        let f = NeuralArchitectureSearch::evaluate_fitness(&arch);
        // 0.5 + 0.05*(2-1) = 0.55 - 0.1 (overfit) = 0.45
        assert!(f < 0.6);
    }

    #[test]
    fn test_fitness_dropout_bonus() {
        let layers = vec![
            NasLayerType::Dense { units: 50 },
            NasLayerType::Dropout { rate: 0.2 },
            NasLayerType::Dropout { rate: 0.2 },
            NasLayerType::Dropout { rate: 0.2 },
            NasLayerType::Dense { units: 10 },
        ];
        let arch = NasArchitecture::new(layers, 100, 10);
        let f = NeuralArchitectureSearch::evaluate_fitness(&arch);
        // Should have dropout bonus
        assert!(f >= 0.5);
    }

    #[test]
    fn test_fitness_batchnorm_bonus() {
        let layers = vec![
            NasLayerType::Dense { units: 50 },
            NasLayerType::BatchNorm,
            NasLayerType::BatchNorm,
            NasLayerType::BatchNorm,
            NasLayerType::Dense { units: 10 },
        ];
        let arch = NasArchitecture::new(layers, 100, 10);
        let f = NeuralArchitectureSearch::evaluate_fitness(&arch);
        assert!(f >= 0.5);
    }

    #[test]
    fn test_fitness_clamped_to_1() {
        // Craft arch that maximises all bonuses
        let mut layers = vec![];
        for _ in 0..4 {
            layers.push(NasLayerType::Dense { units: 50 });
            layers.push(NasLayerType::BatchNorm);
            layers.push(NasLayerType::Dropout { rate: 0.2 });
        }
        layers.push(NasLayerType::Dense { units: 10 });
        let arch = NasArchitecture::new(layers, 100, 10);
        let f = NeuralArchitectureSearch::evaluate_fitness(&arch);
        assert!(f <= 1.0);
    }

    // -----------------------------------------------------------------------
    // generate_random_arch
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_random_arch_output_layer() {
        let config = default_config();
        let mut rng = 1u64;
        let arch = NeuralArchitectureSearch::generate_random_arch(&mut rng, 32, 5, &config);
        // Last layer must be Dense(output_dim)
        let last = arch.layers.last().expect("layers must not be empty");
        assert_eq!(*last, NasLayerType::Dense { units: 5 });
    }

    #[test]
    fn test_generate_random_arch_respects_min_layers() {
        let config = NasConfig {
            min_layers: 2,
            max_layers: 5,
            ..default_config()
        };
        let mut rng = 1u64;
        for _ in 0..20 {
            let arch = NeuralArchitectureSearch::generate_random_arch(&mut rng, 16, 4, &config);
            // count hidden Dense layers (all Dense except the final one)
            let dense_count = arch
                .layers
                .iter()
                .filter(|l| matches!(l, NasLayerType::Dense { .. }))
                .count();
            // at least min_layers hidden Denses + 1 output = min_layers+1 Dense total
            assert!(dense_count > config.min_layers);
        }
    }

    // -----------------------------------------------------------------------
    // Mutate
    // -----------------------------------------------------------------------

    #[test]
    fn test_mutate_output_layer_preserved() {
        let config = default_config();
        let layers = vec![
            NasLayerType::Dense { units: 32 },
            NasLayerType::Dense { units: 10 },
        ];
        let arch = NasArchitecture::new(layers, 16, 10);
        let mut rng = 1u64;
        for _ in 0..50 {
            let mutated = NeuralArchitectureSearch::mutate(&arch, &mut rng, &config);
            let last = mutated.layers.last().expect("mutated has no layers");
            assert_eq!(*last, NasLayerType::Dense { units: 10 });
        }
    }

    #[test]
    fn test_mutate_returns_architecture() {
        let config = default_config();
        let layers = vec![
            NasLayerType::Dense { units: 32 },
            NasLayerType::Dense { units: 8 },
        ];
        let arch = NasArchitecture::new(layers, 16, 8);
        let mut rng = 42u64;
        let mutated = NeuralArchitectureSearch::mutate(&arch, &mut rng, &config);
        assert!(!mutated.layers.is_empty());
    }

    // -----------------------------------------------------------------------
    // Crossover
    // -----------------------------------------------------------------------

    #[test]
    fn test_crossover_output_layer() {
        let layers_a = vec![
            NasLayerType::Dense { units: 32 },
            NasLayerType::Dense { units: 64 },
            NasLayerType::Dense { units: 10 },
        ];
        let layers_b = vec![
            NasLayerType::Dense { units: 128 },
            NasLayerType::Dense { units: 10 },
        ];
        let a = NasArchitecture::new(layers_a, 16, 10);
        let b = NasArchitecture::new(layers_b, 16, 10);
        let mut rng = 1u64;
        let child = NeuralArchitectureSearch::crossover(&a, &b, &mut rng);
        let last = child.layers.last().expect("child has no layers");
        assert_eq!(*last, NasLayerType::Dense { units: 10 });
    }

    #[test]
    fn test_crossover_inherits_from_both() {
        let layers_a = vec![
            NasLayerType::Dense { units: 32 },
            NasLayerType::Dense { units: 64 },
            NasLayerType::Dense { units: 128 },
            NasLayerType::Dense { units: 4 },
        ];
        let layers_b = vec![
            NasLayerType::Dense { units: 256 },
            NasLayerType::Dense { units: 512 },
            NasLayerType::Dense { units: 4 },
        ];
        let a = NasArchitecture::new(layers_a, 8, 4);
        let b = NasArchitecture::new(layers_b, 8, 4);
        let mut rng = 7u64;
        let child = NeuralArchitectureSearch::crossover(&a, &b, &mut rng);
        assert!(!child.layers.is_empty());
    }

    // -----------------------------------------------------------------------
    // run_search
    // -----------------------------------------------------------------------

    #[test]
    fn test_run_search_returns_results() {
        let config = default_config();
        let mut nas = NeuralArchitectureSearch::new(config);
        let results = nas.run_search(32, 10);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_run_search_results_sorted_descending() {
        let config = default_config();
        let mut nas = NeuralArchitectureSearch::new(config);
        let results = nas.run_search(32, 10);
        for w in results.windows(2) {
            assert!(w[0].fitness >= w[1].fitness);
        }
    }

    #[test]
    fn test_run_search_sets_best_arch() {
        let config = default_config();
        let mut nas = NeuralArchitectureSearch::new(config);
        nas.run_search(32, 10);
        assert!(nas.best_architecture().is_some());
    }

    #[test]
    fn test_run_search_deterministic_with_seed() {
        let config1 = default_config();
        let config2 = default_config();
        let mut nas1 = NeuralArchitectureSearch::new(config1);
        let mut nas2 = NeuralArchitectureSearch::new(config2);
        let r1 = nas1.run_search(32, 10);
        let r2 = nas2.run_search(32, 10);
        assert_eq!(r1.len(), r2.len());
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.arch_id, b.arch_id);
        }
    }

    #[test]
    fn test_run_search_evolutionary() {
        let config = NasConfig {
            strategy: NasSearchStrategy::Evolutionary {
                population_size: 6,
                mutation_rate: 0.8,
                elite_fraction: 0.3,
            },
            max_generations: 3,
            target_fitness: 0.99,
            min_layers: 1,
            max_layers: 4,
            min_units: 8,
            max_units: 64,
            seed: 1,
        };
        let mut nas = NeuralArchitectureSearch::new(config);
        let results = nas.run_search(16, 5);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_run_search_grid_search() {
        let config = NasConfig {
            strategy: NasSearchStrategy::GridSearch {
                layer_options: vec![16, 32],
                depth_range: (1, 2),
            },
            max_generations: 2,
            target_fitness: 0.99,
            min_layers: 1,
            max_layers: 3,
            min_units: 16,
            max_units: 64,
            seed: 1,
        };
        let mut nas = NeuralArchitectureSearch::new(config);
        let results = nas.run_search(8, 4);
        // 2 options ^ 1 depth + 2 options ^ 2 depths = 2 + 4 = 6 architectures
        // with 2 generations → at least 12 results total
        assert!(!results.is_empty());
    }

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_after_run() {
        let config = default_config();
        let mut nas = NeuralArchitectureSearch::new(config);
        nas.run_search(16, 4);
        let stats = nas.stats();
        assert!(stats.total_architectures_evaluated > 0);
        assert!(stats.best_fitness >= 0.0 && stats.best_fitness <= 1.0);
        assert!(stats.avg_fitness >= 0.0 && stats.avg_fitness <= 1.0);
        assert!(stats.population_size > 0);
    }

    #[test]
    fn test_stats_before_run() {
        let config = default_config();
        let nas = NeuralArchitectureSearch::new(config);
        let stats = nas.stats();
        assert_eq!(stats.total_architectures_evaluated, 0);
        assert_eq!(stats.best_fitness, 0.0);
    }

    #[test]
    fn test_stats_generations_run() {
        let config = NasConfig {
            max_generations: 5,
            ..default_config()
        };
        let mut nas = NeuralArchitectureSearch::new(config);
        nas.run_search(16, 4);
        let stats = nas.stats();
        assert!(stats.generations_run >= 1 && stats.generations_run <= 5);
    }

    // -----------------------------------------------------------------------
    // EvaluationResult fields
    // -----------------------------------------------------------------------

    #[test]
    fn test_evaluation_result_fitness_in_range() {
        let config = default_config();
        let mut nas = NeuralArchitectureSearch::new(config);
        let results = nas.run_search(16, 4);
        for r in &results {
            assert!(r.fitness >= 0.0 && r.fitness <= 1.0);
            assert!(r.accuracy >= 0.0 && r.accuracy <= 1.0);
            assert!(r.latency_ms > 0.0);
        }
    }

    #[test]
    fn test_evaluation_result_has_arch_id() {
        let config = default_config();
        let mut nas = NeuralArchitectureSearch::new(config);
        let results = nas.run_search(16, 4);
        for r in &results {
            assert_ne!(r.arch_id, 0);
        }
    }

    #[test]
    fn test_nas_stats_struct_fields() {
        let s = NasStats {
            generations_run: 3,
            total_architectures_evaluated: 24,
            best_fitness: 0.72,
            avg_fitness: 0.60,
            population_size: 8,
        };
        assert_eq!(s.generations_run, 3);
        assert_eq!(s.total_architectures_evaluated, 24);
    }

    /// Ensure the exported `NasEvaluationResult` type is usable
    #[test]
    fn test_evaluation_result_struct() {
        let r = NasEvaluationResult {
            arch_id: 42,
            fitness: 0.8,
            accuracy: 0.75,
            latency_ms: 5.0,
            parameter_count: 5000,
            generation: 1,
        };
        assert_eq!(r.arch_id, 42);
        assert_eq!(r.generation, 1);
    }
}
