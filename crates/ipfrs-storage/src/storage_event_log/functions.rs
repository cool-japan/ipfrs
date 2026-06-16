//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{SelStorageEvent, StorageEventKind};

/// FNV-1a 64-bit hash over a byte slice.
pub fn sel_fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}
/// Compute the FNV-1a checksum for a [`SelStorageEvent`].
///
/// The checksum covers `id`, `object_id`, `user_id`, and `timestamp`.
pub fn event_checksum(e: &SelStorageEvent) -> u64 {
    let mut buf = Vec::with_capacity(8 + e.object_id.len() + e.user_id.len() + 8);
    buf.extend_from_slice(&e.id.0.to_le_bytes());
    buf.extend_from_slice(e.object_id.as_bytes());
    buf.extend_from_slice(e.user_id.as_bytes());
    buf.extend_from_slice(&e.timestamp.to_le_bytes());
    sel_fnv1a_64(&buf)
}
/// Extract the CID referenced by a [`StorageEventKind`], if any.
pub fn event_cid(kind: &StorageEventKind) -> Option<&str> {
    match kind {
        StorageEventKind::Put { cid, .. } => Some(cid),
        StorageEventKind::Get { cid, .. } => Some(cid),
        StorageEventKind::Delete { cid, .. } => Some(cid),
        StorageEventKind::Evict { cid, .. } => Some(cid),
        StorageEventKind::Replicate { cid, .. } => Some(cid),
        StorageEventKind::Verify { cid, .. } => Some(cid),
        StorageEventKind::Migrate { cid, .. } => Some(cid),
        StorageEventKind::Compact { .. } => None,
    }
}
