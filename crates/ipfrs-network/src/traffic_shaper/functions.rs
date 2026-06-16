//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// xorshift64 pseudo-random number generator — used for RED and tests.
#[inline]
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}
/// Returns a uniform float in [0, 1) using xorshift64.
#[inline]
pub fn xorshift_f64(state: &mut u64) -> f64 {
    (xorshift64(state) >> 11) as f64 / (1u64 << 53) as f64
}
/// Compute available tokens given elapsed microseconds and burst cap.
#[inline]
pub fn tokens_available(tokens: u64, rate_bps: u64, elapsed_us: u64, burst_bytes: u64) -> u64 {
    let new_tokens = (rate_bps as u128 * elapsed_us as u128 / 8_000_000) as u64;
    (tokens + new_tokens).min(burst_bytes)
}
