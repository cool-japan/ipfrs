//! Weight initialization strategies for tensor operations.
//!
//! Provides common initialization methods used in deep learning:
//! - Xavier/Glorot (uniform and normal)
//! - He/Kaiming (uniform and normal)
//! - LeCun (uniform and normal)
//! - Orthogonal initialization
//! - Sparse initialization
//! - Truncated normal
//! - Constant/zeros/ones
//!
//! Uses a custom xorshift64 PRNG to avoid external random number dependencies.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Fan mode for calculating fan-in/fan-out
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FanMode {
    /// Use fan-in (number of input units)
    FanIn,
    /// Use fan-out (number of output units)
    FanOut,
    /// Use average of fan-in and fan-out
    Average,
}

/// Distribution type for initialization
#[derive(Debug, Clone, PartialEq)]
pub enum InitDistribution {
    /// Uniform distribution
    Uniform,
    /// Normal (Gaussian) distribution
    Normal,
}

/// Weight initialization strategy
#[derive(Debug, Clone, PartialEq)]
pub enum InitStrategy {
    /// All weights set to zero
    Zeros,
    /// All weights set to one
    Ones,
    /// All weights set to a constant value
    Constant(f64),
    /// Xavier/Glorot initialization: Var = 2/(fan_in + fan_out)
    Xavier(InitDistribution),
    /// He initialization: Var = 2/fan
    He(FanMode, InitDistribution),
    /// Kaiming initialization (alias for He)
    Kaiming(FanMode, InitDistribution),
    /// LeCun initialization: Var = 1/fan_in
    LeCun(InitDistribution),
    /// Orthogonal initialization with gain
    Orthogonal(f64),
    /// Sparse initialization with sparsity ratio (fraction of zeros)
    Sparse(f64),
    /// Truncated normal distribution clamped to [a, b]
    TruncatedNormal {
        /// Mean of the normal distribution
        mean: f64,
        /// Standard deviation
        std: f64,
        /// Lower bound
        a: f64,
        /// Upper bound
        b: f64,
    },
}

// ---------------------------------------------------------------------------
// Config / Shape / Stats
// ---------------------------------------------------------------------------

/// Configuration for weight initialization.
#[derive(Debug, Clone)]
pub struct WeightInitConfig {
    /// The initialization strategy to use
    pub strategy: InitStrategy,
    /// Seed for the PRNG
    pub seed: u64,
    /// Gain factor (used by Xavier, He, etc.)
    pub gain: f64,
}

impl Default for WeightInitConfig {
    fn default() -> Self {
        Self {
            strategy: InitStrategy::Xavier(InitDistribution::Uniform),
            seed: 42,
            gain: 1.0,
        }
    }
}

/// Shape information for computing fan-in/fan-out.
#[derive(Debug, Clone)]
pub struct TensorShape {
    /// Dimensions of the tensor
    pub dims: Vec<usize>,
}

impl TensorShape {
    /// Create a new `TensorShape` from the given dimensions.
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    /// Total number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }
}

/// Statistics about initialised weights.
#[derive(Debug, Clone)]
pub struct InitStats {
    /// Total number of parameters
    pub total_params: u64,
    /// Mean of the weights
    pub mean: f64,
    /// Variance of the weights
    pub variance: f64,
    /// Minimum weight value
    pub min: f64,
    /// Maximum weight value
    pub max: f64,
}

// ---------------------------------------------------------------------------
// WeightInitializer
// ---------------------------------------------------------------------------

/// Weight initializer that generates tensors according to a chosen strategy.
pub struct WeightInitializer {
    config: WeightInitConfig,
    rng_state: u64,
    initialized_count: u64,
}

impl WeightInitializer {
    /// Create a new initializer from the given configuration.
    pub fn new(config: WeightInitConfig) -> Self {
        let seed = if config.seed == 0 { 1 } else { config.seed };
        Self {
            rng_state: seed,
            config,
            initialized_count: 0,
        }
    }

    // -- Public API ----------------------------------------------------------

    /// Generate weights for the given tensor shape.
    pub fn initialize(&mut self, shape: &TensorShape) -> Vec<f64> {
        let n = shape.numel();
        let (fan_in, fan_out) = Self::compute_fan(shape);
        let gain = self.config.gain;

        let strategy = self.config.strategy.clone();
        let weights = match strategy {
            InitStrategy::Zeros => vec![0.0; n],
            InitStrategy::Ones => vec![1.0; n],
            InitStrategy::Constant(v) => vec![v; n],
            InitStrategy::Xavier(ref dist) => self.xavier_init(n, fan_in, fan_out, gain, dist),
            InitStrategy::He(ref mode, ref dist) | InitStrategy::Kaiming(ref mode, ref dist) => {
                self.he_init(n, fan_in, fan_out, gain, mode, dist)
            }
            InitStrategy::LeCun(ref dist) => self.lecun_init(n, fan_in, gain, dist),
            InitStrategy::Orthogonal(g) => self.orthogonal_init(shape, g),
            InitStrategy::Sparse(sparsity) => self.sparse_init(n, sparsity),
            InitStrategy::TruncatedNormal { mean, std, a, b } => {
                self.truncated_normal_init(n, mean, std, a, b)
            }
        };

        self.initialized_count += n as u64;
        weights
    }

    /// Compute (fan_in, fan_out) from a tensor shape.
    ///
    /// - 1D: `(dims[0], dims[0])`
    /// - 2D: `(dims[1], dims[0])`
    /// - 3D+: `(dims[1] * receptive, dims[0] * receptive)` where
    ///   `receptive = product of dims[2..]`
    pub fn compute_fan(shape: &TensorShape) -> (usize, usize) {
        match shape.dims.len() {
            0 => (1, 1),
            1 => (shape.dims[0], shape.dims[0]),
            2 => (shape.dims[1], shape.dims[0]),
            _ => {
                let receptive: usize = shape.dims[2..].iter().product();
                let fan_in = shape.dims[1] * receptive;
                let fan_out = shape.dims[0] * receptive;
                (fan_in, fan_out)
            }
        }
    }

    /// Compute the Xavier/Glorot bound: `gain * sqrt(6 / (fan_in + fan_out))`.
    pub fn xavier_bound(fan_in: usize, fan_out: usize, gain: f64) -> f64 {
        gain * (6.0 / (fan_in + fan_out) as f64).sqrt()
    }

    /// Compute the He/Kaiming bound: `gain * sqrt(3 / fan)`.
    pub fn he_bound(fan: usize, gain: f64) -> f64 {
        gain * (3.0 / fan as f64).sqrt()
    }

    /// Generate a uniform random value in `[low, high)` using xorshift64.
    pub fn next_uniform(&mut self, low: f64, high: f64) -> f64 {
        let u = self.next_unit();
        low + u * (high - low)
    }

    /// Generate a normally distributed value via the Box-Muller transform.
    pub fn next_normal(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = self.next_unit().max(1e-300); // avoid log(0)
        let u2 = self.next_unit();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        mean + std * z
    }

    /// Compute basic statistics for a slice of weights.
    pub fn compute_stats(weights: &[f64]) -> InitStats {
        if weights.is_empty() {
            return InitStats {
                total_params: 0,
                mean: 0.0,
                variance: 0.0,
                min: 0.0,
                max: 0.0,
            };
        }

        let n = weights.len() as f64;
        let sum: f64 = weights.iter().sum();
        let mean = sum / n;

        let var_sum: f64 = weights.iter().map(|w| (w - mean).powi(2)).sum();
        let variance = var_sum / n;

        let min = weights.iter().copied().fold(f64::INFINITY, f64::min);
        let max = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        InitStats {
            total_params: weights.len() as u64,
            mean,
            variance,
            min,
            max,
        }
    }

    /// Reset the PRNG seed.
    pub fn reset_seed(&mut self, seed: u64) {
        self.rng_state = if seed == 0 { 1 } else { seed };
    }

    /// Return the total number of weights initialised so far.
    pub fn initialized_count(&self) -> u64 {
        self.initialized_count
    }

    // -- Private helpers -----------------------------------------------------

    /// xorshift64 — returns a value in [0, 1).
    fn next_unit(&mut self) -> f64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        // Map to [0, 1)
        (x as f64) / (u64::MAX as f64)
    }

    fn xavier_init(
        &mut self,
        n: usize,
        fan_in: usize,
        fan_out: usize,
        gain: f64,
        dist: &InitDistribution,
    ) -> Vec<f64> {
        match dist {
            InitDistribution::Uniform => {
                let bound = Self::xavier_bound(fan_in, fan_out, gain);
                (0..n).map(|_| self.next_uniform(-bound, bound)).collect()
            }
            InitDistribution::Normal => {
                let std = gain * (2.0 / (fan_in + fan_out) as f64).sqrt();
                (0..n).map(|_| self.next_normal(0.0, std)).collect()
            }
        }
    }

    fn he_init(
        &mut self,
        n: usize,
        fan_in: usize,
        fan_out: usize,
        gain: f64,
        mode: &FanMode,
        dist: &InitDistribution,
    ) -> Vec<f64> {
        let fan = match mode {
            FanMode::FanIn => fan_in,
            FanMode::FanOut => fan_out,
            FanMode::Average => (fan_in + fan_out) / 2,
        };
        match dist {
            InitDistribution::Uniform => {
                let bound = Self::he_bound(fan, gain);
                (0..n).map(|_| self.next_uniform(-bound, bound)).collect()
            }
            InitDistribution::Normal => {
                let std = gain * (2.0 / fan as f64).sqrt();
                (0..n).map(|_| self.next_normal(0.0, std)).collect()
            }
        }
    }

    fn lecun_init(
        &mut self,
        n: usize,
        fan_in: usize,
        gain: f64,
        dist: &InitDistribution,
    ) -> Vec<f64> {
        match dist {
            InitDistribution::Uniform => {
                let bound = gain * (3.0 / fan_in as f64).sqrt();
                (0..n).map(|_| self.next_uniform(-bound, bound)).collect()
            }
            InitDistribution::Normal => {
                let std = gain * (1.0 / fan_in as f64).sqrt();
                (0..n).map(|_| self.next_normal(0.0, std)).collect()
            }
        }
    }

    /// Orthogonal initialization via QR-like decomposition of a random matrix.
    ///
    /// For a 2-D shape (rows, cols) we generate a random matrix and then
    /// produce an orthonormal basis via a simplified Gram-Schmidt process,
    /// scaled by `gain`.  For shapes with more than two dimensions we treat
    /// them as (dims[0], product-of-rest).
    fn orthogonal_init(&mut self, shape: &TensorShape, gain: f64) -> Vec<f64> {
        let (rows, cols) = match shape.dims.len() {
            0 => (1, 1),
            1 => (shape.dims[0], 1),
            2 => (shape.dims[0], shape.dims[1]),
            _ => {
                let rest: usize = shape.dims[1..].iter().product();
                (shape.dims[0], rest)
            }
        };

        let n = rows.max(cols);
        let m = rows.min(cols);

        // Generate random n x m matrix
        let mut mat: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..m).map(|_| self.next_normal(0.0, 1.0)).collect())
            .collect();

        // Modified Gram-Schmidt
        for j in 0..m {
            // Normalise column j
            let norm = col_norm(&mat, j, n);
            if norm > 1e-15 {
                for row in mat.iter_mut().take(n) {
                    row[j] /= norm;
                }
            }
            // Subtract projection from remaining columns
            for k in (j + 1)..m {
                let dot = col_dot(&mat, j, k, n);
                for row in mat.iter_mut().take(n) {
                    row[k] -= dot * row[j];
                }
            }
        }

        // Extract the result (rows x cols), applying gain
        let mut result = Vec::with_capacity(rows * cols);
        if rows <= cols {
            // Take first `rows` rows of the n x m matrix
            for row in mat.iter().take(rows) {
                for val in row.iter().take(cols.min(m)) {
                    result.push(val * gain);
                }
                // If cols > m pad with zeros (shouldn't happen with our setup)
                if cols > m {
                    result.extend(std::iter::repeat_n(0.0, cols - m));
                }
            }
        } else {
            // n == rows, m == cols
            for row in mat.iter().take(rows) {
                for val in row.iter().take(cols) {
                    result.push(val * gain);
                }
            }
        }

        result
    }

    /// Sparse initialisation: a fraction `sparsity` of the weights are zero,
    /// the rest are drawn from a standard normal.
    fn sparse_init(&mut self, n: usize, sparsity: f64) -> Vec<f64> {
        let sparsity = sparsity.clamp(0.0, 1.0);
        (0..n)
            .map(|_| {
                let u = self.next_unit();
                if u < sparsity {
                    0.0
                } else {
                    self.next_normal(0.0, 1.0)
                }
            })
            .collect()
    }

    /// Truncated normal: rejection-sample values in [a, b].
    fn truncated_normal_init(&mut self, n: usize, mean: f64, std: f64, a: f64, b: f64) -> Vec<f64> {
        (0..n)
            .map(|_| {
                // Rejection sampling — cap iterations to avoid infinite loop on
                // extremely narrow bounds.
                for _ in 0..1000 {
                    let v = self.next_normal(mean, std);
                    if v >= a && v <= b {
                        return v;
                    }
                }
                // Fallback: clamp
                self.next_normal(mean, std).clamp(a, b)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Free helpers for orthogonal init
// ---------------------------------------------------------------------------

/// L2 norm of column `j` across all rows.
fn col_norm(mat: &[Vec<f64>], j: usize, rows: usize) -> f64 {
    let s: f64 = mat.iter().take(rows).map(|row| row[j] * row[j]).sum();
    s.sqrt()
}

/// Dot product of columns `j` and `k`.
fn col_dot(mat: &[Vec<f64>], j: usize, k: usize, rows: usize) -> f64 {
    mat.iter().take(rows).map(|row| row[j] * row[k]).sum()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers -------------------------------------------------------------

    fn make_init(strategy: InitStrategy) -> WeightInitializer {
        WeightInitializer::new(WeightInitConfig {
            strategy,
            seed: 12345,
            gain: 1.0,
        })
    }

    fn shape2d(r: usize, c: usize) -> TensorShape {
        TensorShape::new(vec![r, c])
    }

    // -- Zeros / Ones / Constant --------------------------------------------

    #[test]
    fn test_zeros() {
        let mut init = make_init(InitStrategy::Zeros);
        let w = init.initialize(&shape2d(3, 4));
        assert_eq!(w.len(), 12);
        assert!(w.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_ones() {
        let mut init = make_init(InitStrategy::Ones);
        let w = init.initialize(&shape2d(3, 4));
        assert_eq!(w.len(), 12);
        assert!(w.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_constant() {
        let mut init = make_init(InitStrategy::Constant(std::f64::consts::PI));
        let w = init.initialize(&shape2d(2, 5));
        assert_eq!(w.len(), 10);
        assert!(w.iter().all(|&v| (v - std::f64::consts::PI).abs() < 1e-12));
    }

    // -- Fan computation ----------------------------------------------------

    #[test]
    fn test_fan_1d() {
        let s = TensorShape::new(vec![128]);
        let (fi, fo) = WeightInitializer::compute_fan(&s);
        assert_eq!(fi, 128);
        assert_eq!(fo, 128);
    }

    #[test]
    fn test_fan_2d() {
        let s = shape2d(64, 128);
        let (fi, fo) = WeightInitializer::compute_fan(&s);
        assert_eq!(fi, 128);
        assert_eq!(fo, 64);
    }

    #[test]
    fn test_fan_4d() {
        // Conv2D: [out_channels=32, in_channels=16, kH=3, kW=3]
        let s = TensorShape::new(vec![32, 16, 3, 3]);
        let (fi, fo) = WeightInitializer::compute_fan(&s);
        assert_eq!(fi, 16 * 9);
        assert_eq!(fo, 32 * 9);
    }

    #[test]
    fn test_fan_0d() {
        let s = TensorShape::new(vec![]);
        let (fi, fo) = WeightInitializer::compute_fan(&s);
        assert_eq!(fi, 1);
        assert_eq!(fo, 1);
    }

    #[test]
    fn test_fan_3d() {
        // [out, in, kernel]
        let s = TensorShape::new(vec![64, 32, 5]);
        let (fi, fo) = WeightInitializer::compute_fan(&s);
        assert_eq!(fi, 32 * 5);
        assert_eq!(fo, 64 * 5);
    }

    // -- Xavier -------------------------------------------------------------

    #[test]
    fn test_xavier_uniform_variance() {
        let mut init = WeightInitializer::new(WeightInitConfig {
            strategy: InitStrategy::Xavier(InitDistribution::Uniform),
            seed: 42,
            gain: 1.0,
        });
        let shape = shape2d(512, 256);
        let w = init.initialize(&shape);

        let stats = WeightInitializer::compute_stats(&w);
        // Theoretical variance for uniform[-a,a]: a^2/3
        // a = sqrt(6/(512+256)), Var = 6/(3*(512+256)) = 2/(512+256)
        let expected_var = 2.0 / (512.0 + 256.0);
        let rel_err = (stats.variance - expected_var).abs() / expected_var;
        assert!(
            rel_err < 0.1,
            "Xavier uniform var {:.6} vs expected {:.6}",
            stats.variance,
            expected_var
        );
    }

    #[test]
    fn test_xavier_normal_variance() {
        let mut init = WeightInitializer::new(WeightInitConfig {
            strategy: InitStrategy::Xavier(InitDistribution::Normal),
            seed: 42,
            gain: 1.0,
        });
        let shape = shape2d(512, 256);
        let w = init.initialize(&shape);

        let stats = WeightInitializer::compute_stats(&w);
        let expected_var = 2.0 / (512.0 + 256.0);
        let rel_err = (stats.variance - expected_var).abs() / expected_var;
        assert!(
            rel_err < 0.15,
            "Xavier normal var {:.6} vs expected {:.6}",
            stats.variance,
            expected_var
        );
    }

    // -- He / Kaiming -------------------------------------------------------

    #[test]
    fn test_he_fan_in_uniform() {
        let mut init = WeightInitializer::new(WeightInitConfig {
            strategy: InitStrategy::He(FanMode::FanIn, InitDistribution::Uniform),
            seed: 77,
            gain: 1.0,
        });
        let shape = shape2d(256, 512);
        let w = init.initialize(&shape);

        let stats = WeightInitializer::compute_stats(&w);
        // Var = bound^2 / 3 = (sqrt(3/fan))^2/3 = 3/(3*fan) = 1/fan
        // He: gain^2 * 2/fan  -> bound = gain*sqrt(3*2/fan) hmm
        // Actually he_bound = gain * sqrt(3/fan), Var(uniform[-b,b]) = b^2/3
        // For He fan_in: fan = 512, bound = sqrt(3/512)
        // Var = 3/(3*512) = 1/512
        // But He init uses gain * sqrt(3/fan) -> b^2 = 3/512 -> Var = b^2/3 = 1/512
        // However the theory for He is Var = 2/fan_in. The uniform variant
        // uses bound = sqrt(6/fan) so Var = 6/(3*fan) = 2/fan.
        // Our implementation uses he_bound = gain * sqrt(3/fan). Let me check:
        // That means Var = (3/fan)/3 = 1/fan. Let me re-verify against the code.
        // The uniform case: bound = he_bound(fan, gain) = gain * sqrt(3/fan)
        // Var(U[-b,b]) = b^2/3 = gain^2 * 3/(3*fan) = gain^2/fan
        // For gain=sqrt(2): Var = 2/fan. But we use gain=1 here.
        // So with gain=1, Var = 1/fan_in = 1/512.
        let expected_var = 1.0 / 512.0;
        let rel_err = (stats.variance - expected_var).abs() / expected_var;
        assert!(
            rel_err < 0.15,
            "He fan_in uniform var {:.6} vs expected {:.6}",
            stats.variance,
            expected_var
        );
    }

    #[test]
    fn test_he_fan_in_normal() {
        let mut init = WeightInitializer::new(WeightInitConfig {
            strategy: InitStrategy::He(FanMode::FanIn, InitDistribution::Normal),
            seed: 99,
            gain: 1.0,
        });
        let shape = shape2d(256, 512);
        let w = init.initialize(&shape);
        let stats = WeightInitializer::compute_stats(&w);
        // std = gain * sqrt(2/fan_in), Var = gain^2 * 2/fan_in
        let expected_var = 2.0 / 512.0;
        let rel_err = (stats.variance - expected_var).abs() / expected_var;
        assert!(
            rel_err < 0.15,
            "He fan_in normal var {:.6} vs expected {:.6}",
            stats.variance,
            expected_var
        );
    }

    #[test]
    fn test_kaiming_is_he_alias() {
        let seed = 555;
        let shape = shape2d(64, 128);

        let mut init_he = WeightInitializer::new(WeightInitConfig {
            strategy: InitStrategy::He(FanMode::FanIn, InitDistribution::Uniform),
            seed,
            gain: 1.0,
        });
        let mut init_kaiming = WeightInitializer::new(WeightInitConfig {
            strategy: InitStrategy::Kaiming(FanMode::FanIn, InitDistribution::Uniform),
            seed,
            gain: 1.0,
        });

        let w1 = init_he.initialize(&shape);
        let w2 = init_kaiming.initialize(&shape);
        assert_eq!(w1, w2, "Kaiming must produce the same weights as He");
    }

    #[test]
    fn test_he_fan_out() {
        let mut init = WeightInitializer::new(WeightInitConfig {
            strategy: InitStrategy::He(FanMode::FanOut, InitDistribution::Normal),
            seed: 42,
            gain: 1.0,
        });
        let shape = shape2d(256, 512);
        let w = init.initialize(&shape);
        let stats = WeightInitializer::compute_stats(&w);
        // fan_out = 256, Var = 2/256
        let expected_var = 2.0 / 256.0;
        let rel_err = (stats.variance - expected_var).abs() / expected_var;
        assert!(
            rel_err < 0.15,
            "He fan_out normal var {:.6} vs expected {:.6}",
            stats.variance,
            expected_var
        );
    }

    #[test]
    fn test_he_average_mode() {
        let mut init = WeightInitializer::new(WeightInitConfig {
            strategy: InitStrategy::He(FanMode::Average, InitDistribution::Normal),
            seed: 42,
            gain: 1.0,
        });
        let shape = shape2d(200, 400);
        let w = init.initialize(&shape);
        let stats = WeightInitializer::compute_stats(&w);
        // fan = (400+200)/2 = 300, Var = 2/300
        let expected_var = 2.0 / 300.0;
        let rel_err = (stats.variance - expected_var).abs() / expected_var;
        assert!(
            rel_err < 0.15,
            "He average normal var {:.6} vs expected {:.6}",
            stats.variance,
            expected_var
        );
    }

    // -- LeCun --------------------------------------------------------------

    #[test]
    fn test_lecun_normal_variance() {
        let mut init = WeightInitializer::new(WeightInitConfig {
            strategy: InitStrategy::LeCun(InitDistribution::Normal),
            seed: 42,
            gain: 1.0,
        });
        let shape = shape2d(256, 512);
        let w = init.initialize(&shape);
        let stats = WeightInitializer::compute_stats(&w);
        // std = gain * sqrt(1/fan_in), Var = 1/fan_in
        let expected_var = 1.0 / 512.0;
        let rel_err = (stats.variance - expected_var).abs() / expected_var;
        assert!(
            rel_err < 0.15,
            "LeCun normal var {:.6} vs expected {:.6}",
            stats.variance,
            expected_var
        );
    }

    #[test]
    fn test_lecun_uniform_variance() {
        let mut init = WeightInitializer::new(WeightInitConfig {
            strategy: InitStrategy::LeCun(InitDistribution::Uniform),
            seed: 42,
            gain: 1.0,
        });
        let shape = shape2d(256, 512);
        let w = init.initialize(&shape);
        let stats = WeightInitializer::compute_stats(&w);
        // bound = gain * sqrt(3/fan_in), Var = bound^2/3 = 1/fan_in
        let expected_var = 1.0 / 512.0;
        let rel_err = (stats.variance - expected_var).abs() / expected_var;
        assert!(
            rel_err < 0.15,
            "LeCun uniform var {:.6} vs expected {:.6}",
            stats.variance,
            expected_var
        );
    }

    // -- TruncatedNormal ----------------------------------------------------

    #[test]
    fn test_truncated_normal_bounds() {
        let mut init = make_init(InitStrategy::TruncatedNormal {
            mean: 0.0,
            std: 1.0,
            a: -1.5,
            b: 1.5,
        });
        let w = init.initialize(&TensorShape::new(vec![10000]));
        assert!(w.iter().all(|&v| (-1.5..=1.5).contains(&v)));
    }

    #[test]
    fn test_truncated_normal_mean_near_zero() {
        let mut init = make_init(InitStrategy::TruncatedNormal {
            mean: 0.0,
            std: 1.0,
            a: -2.0,
            b: 2.0,
        });
        let w = init.initialize(&TensorShape::new(vec![50000]));
        let stats = WeightInitializer::compute_stats(&w);
        assert!(
            stats.mean.abs() < 0.05,
            "truncated normal mean should be near 0: {}",
            stats.mean
        );
    }

    // -- Orthogonal ---------------------------------------------------------

    #[test]
    fn test_orthogonal_semi_orthogonal() {
        let mut init = make_init(InitStrategy::Orthogonal(1.0));
        let shape = shape2d(4, 4);
        let w = init.initialize(&shape);
        assert_eq!(w.len(), 16);

        // Check that Q^T Q ≈ I (columns are orthonormal)
        let rows = 4;
        let cols = 4;
        for i in 0..cols {
            for j in 0..cols {
                let dot: f64 = (0..rows).map(|r| w[r * cols + i] * w[r * cols + j]).sum();
                if i == j {
                    assert!(
                        (dot - 1.0).abs() < 0.1,
                        "diagonal ({},{}) should be ~1.0, got {}",
                        i,
                        j,
                        dot,
                    );
                } else {
                    assert!(
                        dot.abs() < 0.1,
                        "off-diagonal ({},{}) should be ~0.0, got {}",
                        i,
                        j,
                        dot,
                    );
                }
            }
        }
    }

    #[test]
    fn test_orthogonal_rectangular() {
        let mut init = make_init(InitStrategy::Orthogonal(1.0));
        let shape = shape2d(8, 4);
        let w = init.initialize(&shape);
        assert_eq!(w.len(), 32);

        // For tall matrix (rows > cols), Q^T Q ≈ I (cols x cols identity)
        let cols = 4;
        let rows = 8;
        for i in 0..cols {
            for j in 0..cols {
                let dot: f64 = (0..rows).map(|r| w[r * cols + i] * w[r * cols + j]).sum();
                if i == j {
                    assert!(
                        (dot - 1.0).abs() < 0.15,
                        "orthogonal rect diag ({},{}) = {}",
                        i,
                        j,
                        dot,
                    );
                } else {
                    assert!(
                        dot.abs() < 0.15,
                        "orthogonal rect off-diag ({},{}) = {}",
                        i,
                        j,
                        dot,
                    );
                }
            }
        }
    }

    #[test]
    fn test_orthogonal_with_gain() {
        let mut init = make_init(InitStrategy::Orthogonal(2.0));
        let shape = shape2d(4, 4);
        let w = init.initialize(&shape);

        // Column norms should be approximately `gain = 2.0`
        for j in 0..4 {
            let norm: f64 = (0..4).map(|i| w[i * 4 + j].powi(2)).sum::<f64>().sqrt();
            assert!(
                (norm - 2.0).abs() < 0.3,
                "column {} norm should be ~2.0, got {}",
                j,
                norm,
            );
        }
    }

    // -- Sparse -------------------------------------------------------------

    #[test]
    fn test_sparse_ratio() {
        let mut init = make_init(InitStrategy::Sparse(0.8));
        let w = init.initialize(&TensorShape::new(vec![10000]));
        let zeros = w.iter().filter(|&&v| v == 0.0).count();
        let ratio = zeros as f64 / w.len() as f64;
        assert!(
            (ratio - 0.8).abs() < 0.05,
            "sparse ratio {:.3} vs expected 0.8",
            ratio,
        );
    }

    #[test]
    fn test_sparse_zero_sparsity() {
        let mut init = make_init(InitStrategy::Sparse(0.0));
        let w = init.initialize(&TensorShape::new(vec![1000]));
        let zeros = w.iter().filter(|&&v| v == 0.0).count();
        assert_eq!(zeros, 0, "sparsity=0 should produce no zeros");
    }

    #[test]
    fn test_sparse_full_sparsity() {
        let mut init = make_init(InitStrategy::Sparse(1.0));
        let w = init.initialize(&TensorShape::new(vec![1000]));
        assert!(
            w.iter().all(|&v| v == 0.0),
            "sparsity=1 should be all zeros"
        );
    }

    // -- Stats --------------------------------------------------------------

    #[test]
    fn test_stats_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = WeightInitializer::compute_stats(&data);
        assert_eq!(stats.total_params, 5);
        assert!((stats.mean - 3.0).abs() < 1e-12);
        assert!((stats.variance - 2.0).abs() < 1e-12);
        assert!((stats.min - 1.0).abs() < 1e-12);
        assert!((stats.max - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_stats_empty() {
        let stats = WeightInitializer::compute_stats(&[]);
        assert_eq!(stats.total_params, 0);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.variance, 0.0);
    }

    #[test]
    fn test_stats_single() {
        let stats = WeightInitializer::compute_stats(&[7.0]);
        assert_eq!(stats.total_params, 1);
        assert!((stats.mean - 7.0).abs() < 1e-12);
        assert!((stats.variance).abs() < 1e-12);
        assert!((stats.min - 7.0).abs() < 1e-12);
        assert!((stats.max - 7.0).abs() < 1e-12);
    }

    // -- Seed reproducibility -----------------------------------------------

    #[test]
    fn test_same_seed_same_weights() {
        let shape = shape2d(32, 64);
        let mut a = WeightInitializer::new(WeightInitConfig {
            strategy: InitStrategy::Xavier(InitDistribution::Normal),
            seed: 999,
            gain: 1.0,
        });
        let mut b = WeightInitializer::new(WeightInitConfig {
            strategy: InitStrategy::Xavier(InitDistribution::Normal),
            seed: 999,
            gain: 1.0,
        });
        assert_eq!(a.initialize(&shape), b.initialize(&shape));
    }

    #[test]
    fn test_different_seed_different_weights() {
        let shape = shape2d(32, 64);
        let mut a = WeightInitializer::new(WeightInitConfig {
            strategy: InitStrategy::Xavier(InitDistribution::Normal),
            seed: 111,
            gain: 1.0,
        });
        let mut b = WeightInitializer::new(WeightInitConfig {
            strategy: InitStrategy::Xavier(InitDistribution::Normal),
            seed: 222,
            gain: 1.0,
        });
        assert_ne!(a.initialize(&shape), b.initialize(&shape));
    }

    #[test]
    fn test_reset_seed() {
        let shape = shape2d(16, 16);
        let mut init = WeightInitializer::new(WeightInitConfig {
            strategy: InitStrategy::He(FanMode::FanIn, InitDistribution::Uniform),
            seed: 42,
            gain: 1.0,
        });
        let w1 = init.initialize(&shape);
        init.reset_seed(42);
        let w2 = init.initialize(&shape);
        assert_eq!(w1, w2, "reset_seed should reproduce identical weights");
    }

    // -- Large tensors ------------------------------------------------------

    #[test]
    fn test_large_tensor() {
        let mut init = make_init(InitStrategy::Xavier(InitDistribution::Uniform));
        let shape = TensorShape::new(vec![100, 100]);
        let w = init.initialize(&shape);
        assert_eq!(w.len(), 10_000);
        let stats = WeightInitializer::compute_stats(&w);
        assert!(
            stats.mean.abs() < 0.05,
            "large Xavier mean should be near zero"
        );
    }

    #[test]
    fn test_very_large_tensor() {
        let mut init = make_init(InitStrategy::He(FanMode::FanIn, InitDistribution::Normal));
        let shape = TensorShape::new(vec![50, 50]);
        let w = init.initialize(&shape);
        assert_eq!(w.len(), 2500);
        assert!(w.iter().all(|v| v.is_finite()));
    }

    // -- initialized_count ---------------------------------------------------

    #[test]
    fn test_initialized_count() {
        let mut init = make_init(InitStrategy::Zeros);
        assert_eq!(init.initialized_count(), 0);
        init.initialize(&shape2d(3, 4));
        assert_eq!(init.initialized_count(), 12);
        init.initialize(&TensorShape::new(vec![10]));
        assert_eq!(init.initialized_count(), 22);
    }

    // -- Xavier bound / He bound helpers ------------------------------------

    #[test]
    fn test_xavier_bound_value() {
        let b = WeightInitializer::xavier_bound(256, 512, 1.0);
        let expected = (6.0 / 768.0_f64).sqrt();
        assert!((b - expected).abs() < 1e-12);
    }

    #[test]
    fn test_he_bound_value() {
        let b = WeightInitializer::he_bound(512, 1.0);
        let expected = (3.0 / 512.0_f64).sqrt();
        assert!((b - expected).abs() < 1e-12);
    }

    // -- Gain factor --------------------------------------------------------

    #[test]
    fn test_xavier_with_gain() {
        let mut init = WeightInitializer::new(WeightInitConfig {
            strategy: InitStrategy::Xavier(InitDistribution::Normal),
            seed: 42,
            gain: 2.0,
        });
        let shape = shape2d(256, 256);
        let w = init.initialize(&shape);
        let stats = WeightInitializer::compute_stats(&w);
        // Var = gain^2 * 2/(fan_in+fan_out) = 4 * 2/512 = 8/512
        let expected_var = 4.0 * 2.0 / 512.0;
        let rel_err = (stats.variance - expected_var).abs() / expected_var;
        assert!(
            rel_err < 0.15,
            "Xavier gain=2 var {:.6} vs {:.6}",
            stats.variance,
            expected_var
        );
    }

    // -- numel --------------------------------------------------------------

    #[test]
    fn test_numel() {
        assert_eq!(TensorShape::new(vec![3, 4, 5]).numel(), 60);
        assert_eq!(TensorShape::new(vec![]).numel(), 1); // empty product = 1
        assert_eq!(TensorShape::new(vec![7]).numel(), 7);
    }

    // -- Default config -----------------------------------------------------

    #[test]
    fn test_default_config() {
        let cfg = WeightInitConfig::default();
        assert_eq!(cfg.seed, 42);
        assert!((cfg.gain - 1.0).abs() < 1e-12);
        assert_eq!(
            cfg.strategy,
            InitStrategy::Xavier(InitDistribution::Uniform)
        );
    }
}
