//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::functions::{
    chi2_p_value, inverse_normal_cdf, sample_stats, t_critical, t_two_tailed, xorshift_normal,
    z_two_tailed,
};

/// Aggregate statistics tracked by the engine.
#[derive(Debug, Clone, Default)]
pub struct TestStats {
    /// Total number of tests run.
    pub tests_performed: u64,
    /// Tests where H₀ was rejected.
    pub nulls_rejected: u64,
    /// Fraction of tests where H₀ was rejected.
    pub rejection_rate: f64,
    /// Running average of all p-values.
    pub avg_p_value: f64,
}
/// Errors produced by the hypothesis test engine.
#[derive(Debug, Clone, PartialEq)]
pub enum TestError {
    /// Not enough data was supplied.
    InsufficientData { needed: usize, got: usize },
    /// Alpha (significance level) must be in (0, 1).
    InvalidAlpha(f64),
    /// A numerical computation failed.
    NumericalError(String),
    /// A hypothesis with the given id was not found.
    HypothesisNotFound(String),
    /// The contingency table supplied was invalid.
    InvalidContingency(String),
}
/// Descriptive statistics for a sample.
#[derive(Debug, Clone)]
pub struct SampleData {
    /// Raw observations.
    pub values: Vec<f64>,
    /// Optional label for this sample.
    pub label: String,
    /// Sample size n.
    pub n: usize,
    /// Sample mean x̄.
    pub mean: f64,
    /// Sample variance s² (unbiased, divides by n − 1).
    pub variance: f64,
    /// Sample standard deviation s.
    pub std_dev: f64,
}
/// Which statistical test to perform.
#[derive(Debug, Clone)]
pub enum TestType {
    /// One-sample z-test: known population mean μ₀ and σ.
    OneSampleZTest { mu0: f64, sigma: f64 },
    /// One-sample t-test: known population mean μ₀, unknown σ.
    OneSampleTTest { mu0: f64 },
    /// Two-sample t-test (Welch's or pooled).
    TwoSampleTTest { equal_variance: bool },
    /// Chi-square goodness-of-fit.
    ChiSquareGoodnessOfFit { expected: Vec<f64> },
    /// Chi-square test of independence via contingency table.
    ChiSquareIndependence { contingency: Vec<Vec<f64>> },
    /// One-sample proportion z-test.
    OneSampleProportion { p0: f64 },
    /// Two-sample proportion z-test.
    TwoSampleProportion,
}
/// Statistical hypothesis testing engine.
pub struct HypothesisTestEngine {
    pub(super) hypotheses: HashMap<String, Hypothesis>,
    pub(super) config: EngineConfig,
    pub(super) stats: TestStats,
    /// PRNG state for power simulations.
    pub(super) prng_state: u64,
}
impl HypothesisTestEngine {
    /// Create a new engine with default configuration.
    pub fn new() -> Self {
        Self::with_config(EngineConfig::default())
    }
    /// Create a new engine with a custom configuration.
    pub fn with_config(config: EngineConfig) -> Self {
        Self {
            hypotheses: HashMap::new(),
            config,
            stats: TestStats::default(),
            prng_state: 0x_dead_beef_cafe_babe_u64,
        }
    }
    /// Register a hypothesis. Returns an error if α is invalid.
    pub fn add_hypothesis(&mut self, h: Hypothesis) -> Result<(), TestError> {
        if h.alpha <= 0.0 || h.alpha >= 1.0 {
            return Err(TestError::InvalidAlpha(h.alpha));
        }
        self.hypotheses.insert(h.id.clone(), h);
        Ok(())
    }
    /// Retrieve a previously-registered hypothesis.
    pub fn get_hypothesis(&self, id: &str) -> Option<&Hypothesis> {
        self.hypotheses.get(id)
    }
    /// Run a test identified by `hypothesis_id`.
    pub fn test(
        &mut self,
        hypothesis_id: &str,
        test_type: TestType,
        samples: Vec<SampleData>,
    ) -> Result<TestResult, TestError> {
        let h = self
            .hypotheses
            .get(hypothesis_id)
            .ok_or_else(|| TestError::HypothesisNotFound(hypothesis_id.to_string()))?
            .clone();
        let alpha = h.alpha;
        self.validate_alpha(alpha)?;
        let result = match &test_type {
            TestType::OneSampleZTest { mu0, sigma } => {
                self.one_sample_z(hypothesis_id, &samples, *mu0, *sigma, alpha)?
            }
            TestType::OneSampleTTest { mu0 } => {
                self.one_sample_t(hypothesis_id, &samples, *mu0, alpha)?
            }
            TestType::TwoSampleTTest { equal_variance } => {
                self.two_sample_t(hypothesis_id, &samples, *equal_variance, alpha)?
            }
            TestType::ChiSquareGoodnessOfFit { expected } => {
                self.chi2_gof(hypothesis_id, &samples, expected, alpha)?
            }
            TestType::ChiSquareIndependence { contingency } => {
                self.chi2_independence(hypothesis_id, contingency, alpha)?
            }
            TestType::OneSampleProportion { p0 } => {
                self.one_sample_proportion(hypothesis_id, &samples, *p0, alpha)?
            }
            TestType::TwoSampleProportion => {
                self.two_sample_proportion(hypothesis_id, &samples, alpha)?
            }
        };
        self.stats.tests_performed += 1;
        if result.reject_null {
            self.stats.nulls_rejected += 1;
        }
        let n = self.stats.tests_performed as f64;
        self.stats.avg_p_value = self.stats.avg_p_value * (n - 1.0) / n + result.p_value / n;
        self.stats.rejection_rate =
            self.stats.nulls_rejected as f64 / self.stats.tests_performed as f64;
        Ok(result)
    }
    /// Compute a (1 − α) confidence interval for the mean using the t-distribution.
    pub fn confidence_interval(&self, data: &SampleData, alpha: f64) -> (f64, f64) {
        if data.n < 2 {
            return (data.mean, data.mean);
        }
        let df = (data.n - 1) as u32;
        let tc = t_critical(alpha, df);
        let margin = tc * data.std_dev / (data.n as f64).sqrt();
        (data.mean - margin, data.mean + margin)
    }
    /// Cohen's d effect size: d = (μ₁ − μ₂) / pooled_std.
    pub fn effect_size_cohens_d(s1: &SampleData, s2: &SampleData) -> f64 {
        let n1 = s1.n as f64;
        let n2 = s2.n as f64;
        if n1 < 2.0 || n2 < 2.0 {
            return 0.0;
        }
        let pooled_var = ((n1 - 1.0) * s1.variance + (n2 - 1.0) * s2.variance) / (n1 + n2 - 2.0);
        let pooled_std = pooled_var.sqrt();
        if pooled_std == 0.0 {
            return 0.0;
        }
        (s1.mean - s2.mean) / pooled_std
    }
    /// Monte Carlo power estimate: fraction of 1 000 simulations that reject H₀.
    pub fn power(&mut self, test_type: &TestType, alpha: f64, n: usize, effect_size: f64) -> f64 {
        if !self.config.enable_power_calculation || n == 0 {
            return 0.0;
        }
        let iterations = 1_000_usize;
        let mut rejections = 0u64;
        for _ in 0..iterations {
            let sample: Vec<f64> = (0..n)
                .map(|_| xorshift_normal(&mut self.prng_state) + effect_size)
                .collect();
            let sd = sample_stats(&sample);
            let reject = match test_type {
                TestType::OneSampleZTest { sigma, .. } => {
                    let z = sd.mean / (*sigma / (n as f64).sqrt());
                    z_two_tailed(z) < alpha
                }
                TestType::OneSampleTTest { .. } | TestType::TwoSampleTTest { .. } => {
                    if sd.std_dev == 0.0 || n < 2 {
                        false
                    } else {
                        let t = sd.mean / (sd.std_dev / (n as f64).sqrt());
                        let df = (n - 1) as u32;
                        t_two_tailed(t, df) < alpha
                    }
                }
                _ => {
                    let z = sd.mean * (n as f64).sqrt();
                    z_two_tailed(z) < alpha
                }
            };
            if reject {
                rejections += 1;
            }
        }
        rejections as f64 / iterations as f64
    }
    /// Current aggregate statistics.
    pub fn stats(&self) -> &TestStats {
        &self.stats
    }
    pub(super) fn validate_alpha(&self, alpha: f64) -> Result<(), TestError> {
        if alpha <= 0.0 || alpha >= 1.0 {
            Err(TestError::InvalidAlpha(alpha))
        } else {
            Ok(())
        }
    }
    pub(super) fn require_sample(
        samples: &[SampleData],
        idx: usize,
        min_n: usize,
    ) -> Result<&SampleData, TestError> {
        let s = samples.get(idx).ok_or(TestError::InsufficientData {
            needed: idx + 1,
            got: samples.len(),
        })?;
        if s.n < min_n {
            return Err(TestError::InsufficientData {
                needed: min_n,
                got: s.n,
            });
        }
        Ok(s)
    }
    pub(super) fn one_sample_z(
        &mut self,
        hyp_id: &str,
        samples: &[SampleData],
        mu0: f64,
        sigma: f64,
        alpha: f64,
    ) -> Result<TestResult, TestError> {
        let s = Self::require_sample(samples, 0, self.config.min_sample_size)?;
        if sigma <= 0.0 {
            return Err(TestError::NumericalError("sigma must be positive".into()));
        }
        let se = sigma / (s.n as f64).sqrt();
        let z = (s.mean - mu0) / se;
        let p_value = z_two_tailed(z);
        let reject_null = p_value < alpha;
        let ci = self.confidence_interval(s, alpha);
        let d = (s.mean - mu0) / sigma;
        let pwr = self.power(
            &TestType::OneSampleZTest { mu0, sigma },
            alpha,
            s.n,
            d.abs(),
        );
        Ok(TestResult {
            hypothesis_id: hyp_id.to_string(),
            statistic: TestStatistic::ZScore(z),
            p_value,
            reject_null,
            confidence_interval: Some(ci),
            effect_size: d,
            power: pwr,
        })
    }
    pub(super) fn one_sample_t(
        &mut self,
        hyp_id: &str,
        samples: &[SampleData],
        mu0: f64,
        alpha: f64,
    ) -> Result<TestResult, TestError> {
        let s = Self::require_sample(samples, 0, 2)?;
        if s.std_dev == 0.0 {
            return Err(TestError::NumericalError(
                "standard deviation is zero".into(),
            ));
        }
        let se = s.std_dev / (s.n as f64).sqrt();
        let t = (s.mean - mu0) / se;
        let df = (s.n - 1) as u32;
        let p_value = t_two_tailed(t, df);
        let reject_null = p_value < alpha;
        let ci = self.confidence_interval(s, alpha);
        let d = (s.mean - mu0) / s.std_dev;
        let pwr = self.power(&TestType::OneSampleTTest { mu0 }, alpha, s.n, d.abs());
        Ok(TestResult {
            hypothesis_id: hyp_id.to_string(),
            statistic: TestStatistic::TScore { t, df },
            p_value,
            reject_null,
            confidence_interval: Some(ci),
            effect_size: d,
            power: pwr,
        })
    }
    pub(super) fn two_sample_t(
        &mut self,
        hyp_id: &str,
        samples: &[SampleData],
        equal_variance: bool,
        alpha: f64,
    ) -> Result<TestResult, TestError> {
        let s1 = Self::require_sample(samples, 0, 2)?;
        let s2 = Self::require_sample(samples, 1, 2)?;
        let n1 = s1.n as f64;
        let n2 = s2.n as f64;
        let (t, df) = if equal_variance {
            let sp2 = ((n1 - 1.0) * s1.variance + (n2 - 1.0) * s2.variance) / (n1 + n2 - 2.0);
            let sp = sp2.sqrt();
            if sp == 0.0 {
                return Err(TestError::NumericalError("pooled std is zero".into()));
            }
            let se = sp * (1.0 / n1 + 1.0 / n2).sqrt();
            let t_val = (s1.mean - s2.mean) / se;
            let df_val = (n1 + n2 - 2.0).round() as u32;
            (t_val, df_val)
        } else {
            let se12 = s1.variance / n1;
            let se22 = s2.variance / n2;
            let se_total = (se12 + se22).sqrt();
            if se_total == 0.0 {
                return Err(TestError::NumericalError("Welch SE is zero".into()));
            }
            let t_val = (s1.mean - s2.mean) / se_total;
            let df_num = (se12 + se22).powi(2);
            let df_den = se12.powi(2) / (n1 - 1.0) + se22.powi(2) / (n2 - 1.0);
            let df_val = if df_den == 0.0 {
                1u32
            } else {
                (df_num / df_den).floor() as u32
            };
            (t_val, df_val)
        };
        let p_value = t_two_tailed(t, df);
        let reject_null = p_value < alpha;
        let combined = sample_stats(
            &samples[0]
                .values
                .iter()
                .chain(samples[1].values.iter())
                .copied()
                .collect::<Vec<_>>(),
        );
        let ci = self.confidence_interval(&combined, alpha);
        let d = Self::effect_size_cohens_d(s1, s2);
        let pwr = self.power(
            &TestType::TwoSampleTTest { equal_variance },
            alpha,
            s1.n.min(s2.n),
            d.abs(),
        );
        Ok(TestResult {
            hypothesis_id: hyp_id.to_string(),
            statistic: TestStatistic::TScore { t, df },
            p_value,
            reject_null,
            confidence_interval: Some(ci),
            effect_size: d,
            power: pwr,
        })
    }
    pub(super) fn chi2_gof(
        &mut self,
        hyp_id: &str,
        samples: &[SampleData],
        expected: &[f64],
        alpha: f64,
    ) -> Result<TestResult, TestError> {
        let s = Self::require_sample(samples, 0, 1)?;
        let observed = &s.values;
        if observed.len() != expected.len() {
            return Err(TestError::NumericalError(
                "observed and expected lengths differ".into(),
            ));
        }
        if observed.is_empty() {
            return Err(TestError::InsufficientData { needed: 1, got: 0 });
        }
        let chi2: f64 = observed
            .iter()
            .zip(expected.iter())
            .map(|(o, e)| if *e == 0.0 { 0.0 } else { (o - e).powi(2) / e })
            .sum();
        let df = (observed.len() - 1) as u32;
        let p_value = chi2_p_value(chi2, df);
        let reject_null = p_value < alpha;
        let n_total: f64 = observed.iter().sum();
        let k = observed.len() as f64;
        let effect_size = if n_total > 0.0 && k > 1.0 {
            (chi2 / (n_total * (k - 1.0))).sqrt()
        } else {
            0.0
        };
        let pwr = self.power(
            &TestType::ChiSquareGoodnessOfFit {
                expected: expected.to_vec(),
            },
            alpha,
            s.n,
            effect_size,
        );
        Ok(TestResult {
            hypothesis_id: hyp_id.to_string(),
            statistic: TestStatistic::ChiSquare { chi2, df },
            p_value,
            reject_null,
            confidence_interval: None,
            effect_size,
            power: pwr,
        })
    }
    pub(super) fn chi2_independence(
        &mut self,
        hyp_id: &str,
        contingency: &[Vec<f64>],
        alpha: f64,
    ) -> Result<TestResult, TestError> {
        if contingency.is_empty() {
            return Err(TestError::InvalidContingency("empty table".into()));
        }
        let rows = contingency.len();
        let cols = contingency[0].len();
        if cols == 0 {
            return Err(TestError::InvalidContingency("zero columns".into()));
        }
        for row in contingency {
            if row.len() != cols {
                return Err(TestError::InvalidContingency(
                    "rows have unequal lengths".into(),
                ));
            }
        }
        let row_sums: Vec<f64> = contingency.iter().map(|r| r.iter().sum::<f64>()).collect();
        let col_sums: Vec<f64> = (0..cols)
            .map(|c| contingency.iter().map(|r| r[c]).sum::<f64>())
            .collect();
        let n_total: f64 = row_sums.iter().sum();
        if n_total == 0.0 {
            return Err(TestError::InvalidContingency("all cells are zero".into()));
        }
        let mut chi2 = 0.0_f64;
        for (i, row) in contingency.iter().enumerate() {
            for (j, &obs) in row.iter().enumerate() {
                let exp = row_sums[i] * col_sums[j] / n_total;
                if exp > 0.0 {
                    chi2 += (obs - exp).powi(2) / exp;
                }
            }
        }
        let df = ((rows - 1) * (cols - 1)) as u32;
        let p_value = chi2_p_value(chi2, df);
        let reject_null = p_value < alpha;
        let min_dim = rows.min(cols) as f64;
        let effect_size = if n_total > 0.0 && min_dim > 1.0 {
            (chi2 / (n_total * (min_dim - 1.0))).sqrt()
        } else {
            0.0
        };
        let pwr = self.power(
            &TestType::ChiSquareIndependence {
                contingency: contingency.to_vec(),
            },
            alpha,
            n_total as usize,
            effect_size,
        );
        Ok(TestResult {
            hypothesis_id: hyp_id.to_string(),
            statistic: TestStatistic::ChiSquare { chi2, df },
            p_value,
            reject_null,
            confidence_interval: None,
            effect_size,
            power: pwr,
        })
    }
    pub(super) fn one_sample_proportion(
        &mut self,
        hyp_id: &str,
        samples: &[SampleData],
        p0: f64,
        alpha: f64,
    ) -> Result<TestResult, TestError> {
        let s = Self::require_sample(samples, 0, 1)?;
        let n = s.n as f64;
        let p_hat = s.mean;
        let se = (p0 * (1.0 - p0) / n).sqrt();
        if se == 0.0 {
            return Err(TestError::NumericalError(
                "SE is zero: check p0 and n".into(),
            ));
        }
        let z = (p_hat - p0) / se;
        let p_value = z_two_tailed(z);
        let reject_null = p_value < alpha;
        let z_crit = inverse_normal_cdf(1.0 - alpha / 2.0);
        let se_hat = (p_hat * (1.0 - p_hat) / n).sqrt();
        let ci = (p_hat - z_crit * se_hat, p_hat + z_crit * se_hat);
        let effect_size = (p_hat - p0) / se;
        let pwr = self.power(
            &TestType::OneSampleProportion { p0 },
            alpha,
            s.n,
            effect_size.abs(),
        );
        Ok(TestResult {
            hypothesis_id: hyp_id.to_string(),
            statistic: TestStatistic::BinomialZ {
                z,
                n: s.n as u32,
                p: p0,
            },
            p_value,
            reject_null,
            confidence_interval: Some(ci),
            effect_size,
            power: pwr,
        })
    }
    pub(super) fn two_sample_proportion(
        &mut self,
        hyp_id: &str,
        samples: &[SampleData],
        alpha: f64,
    ) -> Result<TestResult, TestError> {
        let s1 = Self::require_sample(samples, 0, 1)?;
        let s2 = Self::require_sample(samples, 1, 1)?;
        let n1 = s1.n as f64;
        let n2 = s2.n as f64;
        let p1 = s1.mean;
        let p2 = s2.mean;
        let p_pool = (p1 * n1 + p2 * n2) / (n1 + n2);
        let se = (p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2)).sqrt();
        if se == 0.0 {
            return Err(TestError::NumericalError("pooled SE is zero".into()));
        }
        let z = (p1 - p2) / se;
        let p_value = z_two_tailed(z);
        let reject_null = p_value < alpha;
        let effect_size = (p1 - p2) / se;
        let pwr = self.power(
            &TestType::TwoSampleProportion,
            alpha,
            (n1 + n2) as usize / 2,
            effect_size.abs(),
        );
        Ok(TestResult {
            hypothesis_id: hyp_id.to_string(),
            statistic: TestStatistic::BinomialZ {
                z,
                n: (n1 + n2) as u32,
                p: p_pool,
            },
            p_value,
            reject_null,
            confidence_interval: None,
            effect_size,
            power: pwr,
        })
    }
}
/// The computed test statistic.
#[derive(Debug, Clone, PartialEq)]
pub enum TestStatistic {
    /// Standard normal Z-score.
    ZScore(f64),
    /// Student's t with degrees of freedom.
    TScore { t: f64, df: u32 },
    /// Chi-square with degrees of freedom.
    ChiSquare { chi2: f64, df: u32 },
    /// F-statistic with two degrees of freedom.
    FStatistic { f: f64, df1: u32, df2: u32 },
    /// Binomial z-score with sample size and null proportion.
    BinomialZ { z: f64, n: u32, p: f64 },
}
/// A named statistical hypothesis.
#[derive(Debug, Clone)]
pub struct Hypothesis {
    /// Unique identifier.
    pub id: String,
    /// Plain-English statement.
    pub statement: String,
    /// Null hypothesis (H₀).
    pub null_hypothesis: String,
    /// Alternative hypothesis (H₁).
    pub alternative: String,
    /// Significance level α (default 0.05).
    pub alpha: f64,
}
impl Hypothesis {
    /// Construct a new hypothesis with default α = 0.05.
    pub fn new(
        id: impl Into<String>,
        statement: impl Into<String>,
        null_hypothesis: impl Into<String>,
        alternative: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            statement: statement.into(),
            null_hypothesis: null_hypothesis.into(),
            alternative: alternative.into(),
            alpha: 0.05,
        }
    }
    /// Construct a new hypothesis with a custom α.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
}
/// The result returned after running a hypothesis test.
#[derive(Debug, Clone)]
pub struct TestResult {
    /// Which hypothesis was tested.
    pub hypothesis_id: String,
    /// The computed test statistic.
    pub statistic: TestStatistic,
    /// Two-tailed p-value.
    pub p_value: f64,
    /// True when p_value < α (null is rejected).
    pub reject_null: bool,
    /// Optional confidence interval for the mean (or difference of means).
    pub confidence_interval: Option<(f64, f64)>,
    /// Cohen's d or Cramér's V effect size.
    pub effect_size: f64,
    /// Statistical power (1 − β), estimated via Monte Carlo.
    pub power: f64,
}
/// Engine-level configuration.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Default significance level.
    pub default_alpha: f64,
    /// Minimum sample size accepted.
    pub min_sample_size: usize,
    /// Whether to compute power via Monte Carlo simulation.
    pub enable_power_calculation: bool,
}
