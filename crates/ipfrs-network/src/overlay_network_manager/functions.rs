//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, HashSet};

use super::types::{OverlayError, OverlayLink, VirtualRoute};

/// xorshift64 pseudo-random number generator (state must be non-zero).
#[inline]
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}
/// FNV-1a 64-bit hash of an arbitrary byte slice.
#[inline]
pub fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}
/// Canonical link key: always (smaller, larger) lexicographically.
#[inline]
pub(super) fn canonical_key(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_owned(), b.to_owned())
    } else {
        (b.to_owned(), a.to_owned())
    }
}
/// Create a default link between two nodes (10 ms, 1 Gbps, 100 % reliability).
pub(super) fn default_link(a: &str, b: &str) -> OverlayLink {
    OverlayLink {
        from_id: a.to_owned(),
        to_id: b.to_owned(),
        latency_ms: 10,
        bandwidth_bps: 1_000_000_000,
        reliability: 1.0,
        is_tunnel: false,
    }
}
/// Reconstruct the path from `from` to `to` using the `prev` map.
pub(super) fn reconstruct_path(
    prev: &HashMap<String, String>,
    from: &str,
    to: &str,
) -> Result<Vec<String>, OverlayError> {
    let mut path = Vec::new();
    let mut cur = to.to_owned();
    let mut visited: HashSet<String> = HashSet::new();
    loop {
        if visited.contains(&cur) {
            return Err(OverlayError::NoPathExists {
                from: from.to_owned(),
                to: to.to_owned(),
            });
        }
        visited.insert(cur.clone());
        path.push(cur.clone());
        if cur == from {
            break;
        }
        match prev.get(&cur) {
            Some(p) => cur = p.clone(),
            None => {
                return Err(OverlayError::NoPathExists {
                    from: from.to_owned(),
                    to: to.to_owned(),
                });
            }
        }
    }
    path.reverse();
    Ok(path)
}
/// Return the index of the best candidate in a list of virtual routes.
/// Ranking: smallest total_latency_ms first; ties broken by highest
/// path_reliability.
pub(super) fn best_candidate_index(candidates: &[VirtualRoute]) -> usize {
    let mut best = 0;
    for i in 1..candidates.len() {
        let a = &candidates[best];
        let b = &candidates[i];
        if b.total_latency_ms < a.total_latency_ms
            || (b.total_latency_ms == a.total_latency_ms && b.path_reliability > a.path_reliability)
        {
            best = i;
        }
    }
    best
}
