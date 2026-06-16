//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[inline]
pub(super) fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}
/// Ceiling division for signed integers.
pub(super) fn ceil_div(a: i64, b: i64) -> i64 {
    if b == 0 {
        return if a >= 0 { i64::MAX } else { i64::MIN };
    }
    if (a >= 0 && b > 0) || (a < 0 && b < 0) {
        (a + b - b.signum()) / b
    } else {
        a / b
    }
}
/// Floor division for signed integers.
pub(super) fn floor_div(a: i64, b: i64) -> i64 {
    if b == 0 {
        return if a >= 0 { i64::MAX } else { i64::MIN };
    }
    if (a >= 0 && b > 0) || (a < 0 && b < 0) {
        a / b
    } else {
        (a - b + b.signum()) / b
    }
}
