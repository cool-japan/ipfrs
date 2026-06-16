//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ProtocolFeature;

/// Return the complete set of [`ProtocolFeature`] variants.
pub(super) fn all_known_features() -> [ProtocolFeature; 6] {
    [
        ProtocolFeature::Compression,
        ProtocolFeature::Encryption,
        ProtocolFeature::Multiplexing,
        ProtocolFeature::PriorityQueuing,
        ProtocolFeature::FlowControl,
        ProtocolFeature::ArrowIpc,
    ]
}
#[inline]
pub(super) fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}
#[inline]
pub(super) fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}
