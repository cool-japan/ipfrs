//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::SampleData;

/// Compute descriptive statistics for a slice.
pub fn sample_stats(data: &[f64]) -> SampleData {
    let n = data.len();
    let mean = if n == 0 {
        0.0
    } else {
        data.iter().copied().sum::<f64>() / n as f64
    };
    let variance = if n < 2 {
        0.0
    } else {
        data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    };
    let std_dev = variance.sqrt();
    SampleData {
        values: data.to_vec(),
        label: String::new(),
        n,
        mean,
        variance,
        std_dev,
    }
}
/// Standard normal CDF Φ(z).
///
/// Uses the Abramowitz & Stegun rational approximation (formula 26.2.17),
/// accurate to |ε| ≤ 7.5 × 10⁻⁸.
pub fn normal_cdf(z: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let poly = t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let phi = ((-z * z / 2.0).exp()) / (2.0 * std::f64::consts::PI).sqrt() * poly;
    if z >= 0.0 {
        1.0 - phi
    } else {
        phi
    }
}
/// Two-tailed p-value from a z-score: p = 2 · (1 − Φ(|z|)).
#[inline]
pub(super) fn z_two_tailed(z: f64) -> f64 {
    2.0 * (1.0 - normal_cdf(z.abs()))
}
/// Regularised incomplete gamma P(a, x) via series expansion.
///
/// Used internally by chi2_p_value.
fn regularised_gamma_p(a: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return 0.0;
    }
    let ln_gamma_a = ln_gamma(a);
    let max_iter = 200;
    let mut term = 1.0 / a;
    let mut sum = term;
    let mut ap = a;
    for _ in 0..max_iter {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if term.abs() < sum.abs() * 1e-10 {
            break;
        }
    }
    let val = (-x + a * x.ln() - ln_gamma_a).exp() * sum;
    val.clamp(0.0, 1.0)
}
/// Regularised incomplete gamma Q(a, x) = 1 − P(a, x) via continued fraction.
fn regularised_gamma_q(a: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 1.0;
    }
    let ln_gamma_a = ln_gamma(a);
    let fpmin = 1e-300_f64;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d = 1.0 / b;
    let mut h = d;
    let max_iter = 200;
    for i in 1..=max_iter {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = b + an / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-10 {
            break;
        }
    }
    let val = (-x + a * x.ln() - ln_gamma_a).exp() * h;
    val.clamp(0.0, 1.0)
}
/// Natural log of the gamma function, Lanczos approximation (g=7).
fn ln_gamma(z: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_3,
        676.520_368_121_885_1,
        -1_259.139_216_722_403,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if z < 0.5 {
        std::f64::consts::PI.ln() - (std::f64::consts::PI * z).sin().ln() - ln_gamma(1.0 - z)
    } else {
        let x = z - 1.0;
        let mut a = C[0];
        for (i, &ci) in C[1..].iter().enumerate() {
            a += ci / (x + i as f64 + 1.0);
        }
        let t = x + G + 0.5;
        (2.0 * std::f64::consts::PI).sqrt().ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}
/// Upper-tail p-value for a chi-square statistic: P(χ² > chi2 | df).
pub fn chi2_p_value(chi2: f64, df: u32) -> f64 {
    if chi2 <= 0.0 {
        return 1.0;
    }
    let a = df as f64 / 2.0;
    let x = chi2 / 2.0;
    if x < a + 1.0 {
        1.0 - regularised_gamma_p(a, x)
    } else {
        regularised_gamma_q(a, x)
    }
}
/// Approximate t-distribution CDF via normal for large df;
/// for smaller df uses a beta-function-based approximation.
pub fn t_cdf_approx(t: f64, df: u32) -> f64 {
    if df == 0 {
        return 0.5;
    }
    if df >= 200 {
        return normal_cdf(t);
    }
    let df_f = df as f64;
    let t2 = t * t;
    let x = df_f / (df_f + t2);
    let ibeta = regularised_incomplete_beta(x, df_f / 2.0, 0.5);
    let p = 1.0 - 0.5 * ibeta;
    if t >= 0.0 {
        p
    } else {
        1.0 - p
    }
}
/// Regularised incomplete beta I_x(a, b) via continued fraction (Lentz).
fn regularised_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let symmetry = x > (a + 1.0) / (a + b + 2.0);
    let (xx, aa, bb) = if symmetry { (1.0 - x, b, a) } else { (x, a, b) };
    let ln_beta = ln_gamma(aa) + ln_gamma(bb) - ln_gamma(aa + bb);
    let front = (aa * xx.ln() + bb * (1.0 - xx).ln() - ln_beta).exp() / aa;
    let cf = beta_cf(xx, aa, bb);
    let result = front * cf;
    if symmetry {
        1.0 - result
    } else {
        result
    }
}
/// Continued fraction for the incomplete beta (modified Lentz algorithm).
fn beta_cf(x: f64, a: f64, b: f64) -> f64 {
    let fpmin = 1e-300_f64;
    let max_iter = 200;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0_f64;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < fpmin {
        d = fpmin;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=max_iter {
        let m_f = m as f64;
        let aa = m_f * (b - m_f) * x / ((qam + 2.0 * m_f) * (a + 2.0 * m_f));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa2 = -(a + m_f) * (qab + m_f) * x / ((a + 2.0 * m_f) * (qap + 2.0 * m_f));
        d = 1.0 + aa2 * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa2 / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-10 {
            break;
        }
    }
    h
}
/// Two-tailed p-value from a t-score: p = 2 · (1 − P(T ≤ |t|)).
#[inline]
pub(super) fn t_two_tailed(t: f64, df: u32) -> f64 {
    let p = t_cdf_approx(t.abs(), df);
    2.0 * (1.0 - p)
}
/// xorshift64 PRNG step.
#[inline]
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}
/// Draw a standard-normal variate via Box-Muller using xorshift64.
pub fn xorshift_normal(state: &mut u64) -> f64 {
    let u1 = (xorshift64(state) >> 11) as f64 / (1u64 << 53) as f64 + 1e-10;
    let u2 = (xorshift64(state) >> 11) as f64 / (1u64 << 53) as f64;
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
/// Approximate inverse normal CDF Φ⁻¹(p) via rational approximation
/// (Beasley-Springer-Moro, 1-6 digits accuracy).
pub(super) fn inverse_normal_cdf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e+01,
        2.209_460_984_245_205e+02,
        -2.759_285_104_469_687e+02,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e+01,
        2.506_628_277_459_239e+00,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e+01,
        1.615_858_368_580_409e+02,
        -1.556_989_798_598_866e+02,
        6.680_131_188_771_972e+01,
        -1.328_068_155_288_572e+01,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-03,
        -3.223_964_580_411_365e-01,
        -2.400_758_277_161_838e+00,
        -2.549_732_539_343_734e+00,
        4.374_664_141_464_968e+00,
        2.938_163_982_698_783e+00,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-03,
        3.224_671_290_700_398e-01,
        2.445_134_137_142_996e+00,
        3.754_408_661_907_416e+00,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}
/// Critical value t_{α/2, df} (two-tailed) via binary search on t_cdf_approx.
pub(super) fn t_critical(alpha: f64, df: u32) -> f64 {
    let target_p = 1.0 - alpha / 2.0;
    let mut lo = 0.0_f64;
    let mut hi = 20.0_f64;
    for _ in 0..60 {
        let mid = (lo + hi) / 2.0;
        if t_cdf_approx(mid, df) < target_p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}
