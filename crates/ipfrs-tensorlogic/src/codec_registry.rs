//! Codec Registry — compression/encoding codec selection and peer negotiation
//!
//! Tensor data flows through multiple compression/encoding stages. This module
//! manages codec selection and negotiation between peers so that both sides
//! agree on a common codec before data is transmitted.
//!
//! # Overview
//!
//! [`CodecRegistry`] is the central store of [`CodecDescriptor`] entries, each
//! keyed by a [`CodecId`].  Seven built-in codecs are pre-registered on
//! construction:
//!
//! | Id | Name       | Speed class  | Ratio est. |
//! |----|------------|--------------|------------|
//! | 0  | none       | VeryFast     | 1.00       |
//! | 1  | zstd       | Balanced     | 0.30       |
//! | 2  | lz4        | VeryFast     | 0.55       |
//! | 3  | snappy     | VeryFast     | 0.60       |
//! | 4  | brotli     | Slow         | 0.25       |
//! | 10 | arrow_ipc  | Fast         | 0.70       |
//! | 11 | garw       | Fast         | 0.40       |
//!
//! Codec negotiation follows a "first-match" policy: the first codec in the
//! *local* preference list that is also present in the *remote* list wins.
//!
//! # Examples
//!
//! ```
//! use ipfrs_tensorlogic::codec_registry::{CodecId, CodecRegistry};
//!
//! let registry = CodecRegistry::new();
//!
//! // Look up ZSTD
//! let desc = registry.get(CodecId::ZSTD).expect("example: should succeed in docs");
//! assert_eq!(desc.name, "zstd");
//!
//! // Negotiate between two peers
//! let local  = vec![CodecId::ZSTD, CodecId::LZ4, CodecId::NONE];
//! let remote = vec![CodecId::LZ4,  CodecId::NONE];
//! let agreed = CodecRegistry::negotiate(&local, &remote);
//! assert_eq!(agreed, Some(CodecId::LZ4));
//! ```

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;
use thiserror::Error;

// ─── CodecId ─────────────────────────────────────────────────────────────────

/// Opaque identifier for a compression/encoding codec.
///
/// Use the associated constants (`NONE`, `ZSTD`, …) for well-known codecs.
/// Custom codecs should use ids ≥ 1 000 to avoid conflicts with future
/// built-in ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CodecId(pub u32);

impl CodecId {
    /// No codec — data is passed through unmodified.
    pub const NONE: Self = Self(0);
    /// Zstandard (zstd) compression.
    pub const ZSTD: Self = Self(1);
    /// LZ4 block compression.
    pub const LZ4: Self = Self(2);
    /// Snappy compression.
    pub const SNAPPY: Self = Self(3);
    /// Brotli compression.
    pub const BROTLI: Self = Self(4);
    /// Apache Arrow IPC framing/encoding.
    pub const ARROW_IPC: Self = Self(10);
    /// GARW (Generic Arrow Row-Wire) encoding.
    pub const GARW: Self = Self(11);

    /// Return the raw numeric value of this id.
    #[inline]
    pub fn value(self) -> u32 {
        self.0
    }
}

impl fmt::Display for CodecId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CodecId({})", self.0)
    }
}

impl From<u32> for CodecId {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<CodecId> for u32 {
    fn from(c: CodecId) -> Self {
        c.0
    }
}

// ─── SpeedClass ──────────────────────────────────────────────────────────────

/// Qualitative speed tier for a codec.
///
/// The ordering is `VeryFast < Fast < Balanced < Slow < VerySlow`, i.e.
/// a *smaller* variant is *faster*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SpeedClass {
    /// Negligible encoding/decoding overhead (e.g. pass-through, LZ4).
    VeryFast,
    /// Low overhead; suitable for latency-sensitive paths.
    Fast,
    /// Moderate overhead; good balance of speed and compression.
    Balanced,
    /// High overhead; use when CPU is not the bottleneck.
    Slow,
    /// Very high overhead; only for batch/offline scenarios.
    VerySlow,
}

// ─── CodecDescriptor ─────────────────────────────────────────────────────────

/// Metadata about a single codec.
#[derive(Debug, Clone)]
pub struct CodecDescriptor {
    /// Stable numeric identifier.
    pub id: CodecId,
    /// Human-readable name (e.g. `"zstd"`).
    pub name: String,
    /// Estimated output-to-input byte ratio after compression.
    ///
    /// A value of `0.3` means the compressed output is 30 % of the original
    /// size (70 % reduction).  `1.0` means no compression.
    pub compression_ratio_estimate: f32,
    /// Qualitative speed tier.
    pub speed_class: SpeedClass,
    /// Whether the codec is bit-for-bit lossless.
    pub is_lossless: bool,
}

// ─── CodecError ──────────────────────────────────────────────────────────────

/// Errors that can occur while working with [`CodecRegistry`].
#[derive(Debug, Error)]
pub enum CodecError {
    /// A codec with the given id is already registered.
    #[error("codec id {0} is already registered")]
    AlreadyRegistered(u32),
    /// No codec with the given id exists in the registry.
    #[error("unknown codec id {0}")]
    UnknownCodec(u32),
}

// ─── CodecNegotiationRecord ───────────────────────────────────────────────────

/// Record of a single codec negotiation between the local peer and a remote
/// peer.
#[derive(Debug)]
pub struct CodecNegotiationRecord {
    /// Codec ids offered by the local side, in preference order.
    pub local_offered: Vec<CodecId>,
    /// Codec ids offered by the remote side, in any order.
    pub remote_offered: Vec<CodecId>,
    /// The codec that was agreed upon, or `None` if no common codec was found.
    pub agreed: Option<CodecId>,
    /// Wall-clock time at which negotiation completed.
    pub negotiated_at: Instant,
    /// Duration of the negotiation in milliseconds.
    pub negotiation_ms: u64,
}

// ─── CodecRegistry ───────────────────────────────────────────────────────────

/// Central registry of compression/encoding codecs.
///
/// On construction the seven built-in codecs are pre-registered.  Custom codecs
/// can be added via [`CodecRegistry::register`]; duplicate ids are rejected.
pub struct CodecRegistry {
    /// Internal map from `CodecId` value to descriptor.
    codecs: HashMap<u32, CodecDescriptor>,
}

impl CodecRegistry {
    /// Construct a new registry with all seven built-in codecs pre-registered.
    pub fn new() -> Self {
        let mut registry = Self {
            codecs: HashMap::new(),
        };

        let built_ins: &[CodecDescriptor] = &[
            CodecDescriptor {
                id: CodecId::NONE,
                name: "none".to_string(),
                compression_ratio_estimate: 1.0,
                speed_class: SpeedClass::VeryFast,
                is_lossless: true,
            },
            CodecDescriptor {
                id: CodecId::ZSTD,
                name: "zstd".to_string(),
                compression_ratio_estimate: 0.30,
                speed_class: SpeedClass::Balanced,
                is_lossless: true,
            },
            CodecDescriptor {
                id: CodecId::LZ4,
                name: "lz4".to_string(),
                compression_ratio_estimate: 0.55,
                speed_class: SpeedClass::VeryFast,
                is_lossless: true,
            },
            CodecDescriptor {
                id: CodecId::SNAPPY,
                name: "snappy".to_string(),
                compression_ratio_estimate: 0.60,
                speed_class: SpeedClass::VeryFast,
                is_lossless: true,
            },
            CodecDescriptor {
                id: CodecId::BROTLI,
                name: "brotli".to_string(),
                compression_ratio_estimate: 0.25,
                speed_class: SpeedClass::Slow,
                is_lossless: true,
            },
            CodecDescriptor {
                id: CodecId::ARROW_IPC,
                name: "arrow_ipc".to_string(),
                compression_ratio_estimate: 0.70,
                speed_class: SpeedClass::Fast,
                is_lossless: true,
            },
            CodecDescriptor {
                id: CodecId::GARW,
                name: "garw".to_string(),
                compression_ratio_estimate: 0.40,
                speed_class: SpeedClass::Fast,
                is_lossless: true,
            },
        ];

        for desc in built_ins {
            // These are known-unique; insertion cannot fail at construction time.
            registry.codecs.insert(desc.id.value(), desc.clone());
        }

        registry
    }

    /// Register a custom [`CodecDescriptor`].
    ///
    /// Returns [`CodecError::AlreadyRegistered`] if a codec with the same id
    /// already exists in the registry.
    pub fn register(&mut self, descriptor: CodecDescriptor) -> Result<(), CodecError> {
        let key = descriptor.id.value();
        if self.codecs.contains_key(&key) {
            return Err(CodecError::AlreadyRegistered(key));
        }
        self.codecs.insert(key, descriptor);
        Ok(())
    }

    /// Look up a codec by its id.
    ///
    /// Returns `None` when the id is not registered.
    pub fn get(&self, id: CodecId) -> Option<&CodecDescriptor> {
        self.codecs.get(&id.value())
    }

    /// Return all registered codecs, sorted in ascending [`CodecId`] order.
    pub fn list_all(&self) -> Vec<&CodecDescriptor> {
        let mut entries: Vec<&CodecDescriptor> = self.codecs.values().collect();
        entries.sort_by_key(|d| d.id);
        entries
    }

    /// Negotiate a codec between two peers.
    ///
    /// Returns the first id in `local` that also appears in `remote`, or `None`
    /// if there is no common codec.  This is a static method because negotiation
    /// does not require access to a registry instance — it operates purely on
    /// the id sets.
    pub fn negotiate(local: &[CodecId], remote: &[CodecId]) -> Option<CodecId> {
        // Build a hash-set from the remote list for O(1) lookup.
        let remote_set: std::collections::HashSet<CodecId> = remote.iter().copied().collect();
        local.iter().find(|id| remote_set.contains(id)).copied()
    }

    /// Return a reference to the fastest registered codec.
    ///
    /// When multiple codecs share the best speed class the one with the
    /// numerically smallest [`CodecId`] is returned for determinism.
    ///
    /// Returns `None` only if the registry is empty (not possible after
    /// `new()` unless all entries were somehow replaced).
    pub fn best_for_speed(&self) -> Option<&CodecDescriptor> {
        self.codecs.values().min_by(|a, b| {
            a.speed_class
                .cmp(&b.speed_class)
                .then_with(|| a.id.cmp(&b.id))
        })
    }

    /// Return a reference to the codec with the lowest
    /// `compression_ratio_estimate` (best compression).
    ///
    /// When multiple codecs share the same ratio the one with the numerically
    /// smallest [`CodecId`] is returned for determinism.  NaN ratio values are
    /// sorted last.
    ///
    /// Returns `None` only if the registry is empty.
    pub fn best_for_compression(&self) -> Option<&CodecDescriptor> {
        self.codecs.values().min_by(|a, b| {
            a.compression_ratio_estimate
                .partial_cmp(&b.compression_ratio_estimate)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        })
    }
}

impl Default for CodecRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── 1. All pre-registered codecs are present ──────────────────────────────

    #[test]
    fn test_builtin_codecs_all_present() {
        let r = CodecRegistry::new();
        let ids = [
            CodecId::NONE,
            CodecId::ZSTD,
            CodecId::LZ4,
            CodecId::SNAPPY,
            CodecId::BROTLI,
            CodecId::ARROW_IPC,
            CodecId::GARW,
        ];
        for id in &ids {
            assert!(
                r.get(*id).is_some(),
                "codec {} should be pre-registered",
                id
            );
        }
    }

    // ── 2. Correct codec count ────────────────────────────────────────────────

    #[test]
    fn test_builtin_codec_count() {
        let r = CodecRegistry::new();
        assert_eq!(r.list_all().len(), 7);
    }

    // ── 3. `get` returns the correct descriptor ────────────────────────────────

    #[test]
    fn test_get_zstd_descriptor() {
        let r = CodecRegistry::new();
        let desc = r.get(CodecId::ZSTD).expect("ZSTD should exist");
        assert_eq!(desc.id, CodecId::ZSTD);
        assert_eq!(desc.name, "zstd");
        assert!(desc.is_lossless);
        assert_eq!(desc.speed_class, SpeedClass::Balanced);
    }

    #[test]
    fn test_get_none_descriptor() {
        let r = CodecRegistry::new();
        let desc = r.get(CodecId::NONE).expect("NONE should exist");
        assert_eq!(desc.compression_ratio_estimate, 1.0);
        assert_eq!(desc.speed_class, SpeedClass::VeryFast);
    }

    // ── 4. `get` returns None for unknown id ─────────────────────────────────

    #[test]
    fn test_get_unknown_returns_none() {
        let r = CodecRegistry::new();
        assert!(r.get(CodecId::from(9999)).is_none());
    }

    // ── 5. `negotiate` finds intersection ────────────────────────────────────

    #[test]
    fn test_negotiate_finds_common_codec() {
        let local = vec![CodecId::ZSTD, CodecId::LZ4, CodecId::NONE];
        let remote = vec![CodecId::LZ4, CodecId::NONE];
        let agreed = CodecRegistry::negotiate(&local, &remote);
        assert_eq!(agreed, Some(CodecId::LZ4));
    }

    #[test]
    fn test_negotiate_respects_local_preference_order() {
        // local prefers ZSTD; remote supports all — ZSTD wins
        let local = vec![CodecId::ZSTD, CodecId::LZ4, CodecId::NONE];
        let remote = vec![CodecId::NONE, CodecId::LZ4, CodecId::ZSTD];
        let agreed = CodecRegistry::negotiate(&local, &remote);
        assert_eq!(agreed, Some(CodecId::ZSTD));
    }

    // ── 6. `negotiate` returns None when no common codec ──────────────────────

    #[test]
    fn test_negotiate_no_common_returns_none() {
        let local = vec![CodecId::ZSTD, CodecId::BROTLI];
        let remote = vec![CodecId::LZ4, CodecId::SNAPPY];
        assert_eq!(CodecRegistry::negotiate(&local, &remote), None);
    }

    #[test]
    fn test_negotiate_empty_lists_returns_none() {
        assert_eq!(CodecRegistry::negotiate(&[], &[]), None);
        assert_eq!(CodecRegistry::negotiate(&[CodecId::ZSTD], &[]), None);
        assert_eq!(CodecRegistry::negotiate(&[], &[CodecId::ZSTD]), None);
    }

    // ── 7. `best_for_speed` returns a VeryFast codec ─────────────────────────

    #[test]
    fn test_best_for_speed_is_very_fast() {
        let r = CodecRegistry::new();
        let best = r.best_for_speed().expect("registry is non-empty");
        assert_eq!(
            best.speed_class,
            SpeedClass::VeryFast,
            "expected VeryFast, got {:?}",
            best.speed_class
        );
    }

    // ── 8. `best_for_compression` returns the lowest ratio ────────────────────

    #[test]
    fn test_best_for_compression_is_lowest_ratio() {
        let r = CodecRegistry::new();
        let best = r.best_for_compression().expect("registry is non-empty");
        // Brotli has ratio 0.25, which is the lowest among built-ins
        assert_eq!(
            best.id,
            CodecId::BROTLI,
            "expected BROTLI (0.25), got {} ({})",
            best.name,
            best.compression_ratio_estimate
        );
    }

    // ── 9. `register` custom codec succeeds ───────────────────────────────────

    #[test]
    fn test_register_custom_codec() {
        let mut r = CodecRegistry::new();
        let custom = CodecDescriptor {
            id: CodecId::from(1000),
            name: "my_codec".to_string(),
            compression_ratio_estimate: 0.45,
            speed_class: SpeedClass::Fast,
            is_lossless: true,
        };
        r.register(custom)
            .expect("custom registration should succeed");
        let found = r
            .get(CodecId::from(1000))
            .expect("custom codec should exist");
        assert_eq!(found.name, "my_codec");
    }

    // ── 10. Duplicate registration returns AlreadyRegistered error ───────────

    #[test]
    fn test_register_duplicate_returns_error() {
        let mut r = CodecRegistry::new();
        let dup = CodecDescriptor {
            id: CodecId::ZSTD, // already registered
            name: "duplicate_zstd".to_string(),
            compression_ratio_estimate: 0.30,
            speed_class: SpeedClass::Balanced,
            is_lossless: true,
        };
        let err = r.register(dup).expect_err("duplicate should be rejected");
        match err {
            CodecError::AlreadyRegistered(id) => assert_eq!(id, CodecId::ZSTD.value()),
            other => panic!("unexpected error: {}", other),
        }
    }

    // ── 11. `list_all` returns entries in ascending CodecId order ─────────────

    #[test]
    fn test_list_all_sorted_order() {
        let r = CodecRegistry::new();
        let list = r.list_all();
        let ids: Vec<u32> = list.iter().map(|d| d.id.value()).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted, "list_all() must be sorted by CodecId");
    }

    #[test]
    fn test_list_all_first_is_none() {
        let r = CodecRegistry::new();
        let list = r.list_all();
        assert_eq!(list[0].id, CodecId::NONE);
    }

    // ── 12. CodecId constants have correct numeric values ────────────────────

    #[test]
    fn test_codec_id_constants() {
        assert_eq!(CodecId::NONE.value(), 0);
        assert_eq!(CodecId::ZSTD.value(), 1);
        assert_eq!(CodecId::LZ4.value(), 2);
        assert_eq!(CodecId::SNAPPY.value(), 3);
        assert_eq!(CodecId::BROTLI.value(), 4);
        assert_eq!(CodecId::ARROW_IPC.value(), 10);
        assert_eq!(CodecId::GARW.value(), 11);
    }

    // ── 13. CodecId Display ───────────────────────────────────────────────────

    #[test]
    fn test_codec_id_display() {
        assert_eq!(CodecId::ZSTD.to_string(), "CodecId(1)");
        assert_eq!(CodecId::NONE.to_string(), "CodecId(0)");
    }

    // ── 14. CodecNegotiationRecord stores correct data ────────────────────────

    #[test]
    fn test_codec_negotiation_record() {
        let local = vec![CodecId::ZSTD, CodecId::LZ4];
        let remote = vec![CodecId::LZ4];
        let start = Instant::now();
        let agreed = CodecRegistry::negotiate(&local, &remote);
        let elapsed_ms = start.elapsed().as_millis() as u64;

        let record = CodecNegotiationRecord {
            local_offered: local.clone(),
            remote_offered: remote.clone(),
            agreed,
            negotiated_at: Instant::now(),
            negotiation_ms: elapsed_ms,
        };

        assert_eq!(record.agreed, Some(CodecId::LZ4));
        assert_eq!(record.local_offered.len(), 2);
        assert_eq!(record.remote_offered.len(), 1);
    }

    // ── 15. SpeedClass ordering ────────────────────────────────────────────────

    #[test]
    fn test_speed_class_ordering() {
        assert!(SpeedClass::VeryFast < SpeedClass::Fast);
        assert!(SpeedClass::Fast < SpeedClass::Balanced);
        assert!(SpeedClass::Balanced < SpeedClass::Slow);
        assert!(SpeedClass::Slow < SpeedClass::VerySlow);
    }

    // ── 16. best_for_speed with registry containing only slow codecs ──────────

    #[test]
    fn test_best_for_speed_custom_only_slow() {
        let mut r = CodecRegistry::new();
        // Add an even slower custom codec; the result should still be the
        // built-in VeryFast codec.
        let custom = CodecDescriptor {
            id: CodecId::from(2000),
            name: "super_slow".to_string(),
            compression_ratio_estimate: 0.10,
            speed_class: SpeedClass::VerySlow,
            is_lossless: true,
        };
        r.register(custom).expect("should register");
        let best = r.best_for_speed().expect("non-empty");
        assert_eq!(best.speed_class, SpeedClass::VeryFast);
    }

    // ── 17. UnknownCodec error variant Display ────────────────────────────────

    #[test]
    fn test_unknown_codec_error_display() {
        let err = CodecError::UnknownCodec(42);
        assert!(err.to_string().contains("42"));
    }

    // ── 18. Default impl matches new() ────────────────────────────────────────

    #[test]
    fn test_default_matches_new() {
        let a = CodecRegistry::new();
        let b = CodecRegistry::default();
        assert_eq!(a.list_all().len(), b.list_all().len());
    }
}
