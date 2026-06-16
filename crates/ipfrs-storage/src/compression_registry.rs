//! `StorageCompressionRegistry` — tracks codec registrations, usage statistics,
//! and provides codec selection recommendations based on data characteristics.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// CompressionCodec
// ─────────────────────────────────────────────────────────────────────────────

/// Compression codecs supported by the registry.
///
/// Re-exported as `RegistryCompressionCodec` from `lib.rs` to avoid name
/// collision with `compression_advisor::CompressionCodec`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompressionCodec {
    /// Zstandard — balanced ratio and speed.
    Zstd,
    /// LZ4 — extremely fast, moderate ratio.
    Lz4,
    /// Snappy — fast with decent ratio.
    Snappy,
    /// Brotli — best ratio, slower speed.
    Brotli,
    /// No compression applied.
    None,
}

// ─────────────────────────────────────────────────────────────────────────────
// CodecProfile
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime profile for a single compression codec, updated via EMA on each use.
#[derive(Clone, Debug)]
pub struct CodecProfile {
    /// Which codec this profile belongs to.
    pub codec: CompressionCodec,
    /// Average compression ratio: compressed / original (lower = better).
    pub avg_ratio: f64,
    /// Average compression latency in microseconds.
    pub avg_compress_micros: u64,
    /// Average decompression latency in microseconds.
    pub avg_decompress_micros: u64,
    /// Total bytes processed (original sizes) across all recorded uses.
    pub total_bytes_processed: u64,
    /// Number of times this codec has been recorded.
    pub uses: u64,
}

impl CodecProfile {
    /// Efficiency score: `(1.0 - avg_ratio) / (avg_compress_micros + 1)`.
    ///
    /// Higher values indicate better efficiency (better ratio **and** faster).
    pub fn efficiency_score(&self) -> f64 {
        (1.0 - self.avg_ratio) / (self.avg_compress_micros as f64 + 1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DataCharacteristics
// ─────────────────────────────────────────────────────────────────────────────

/// Describes characteristics of a data block to inform codec selection.
#[derive(Clone, Debug)]
pub struct DataCharacteristics {
    /// Size of the data in bytes.
    pub size_bytes: u64,
    /// Whether the data is primarily text (benefits from Brotli).
    pub is_text: bool,
    /// Whether the data is already compressed (avoid re-compressing).
    pub is_already_compressed: bool,
    /// Whether low latency is preferred over compression ratio.
    pub latency_sensitive: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// CodecRecommendation
// ─────────────────────────────────────────────────────────────────────────────

/// A recommended codec together with a human-readable rationale.
#[derive(Clone, Debug)]
pub struct CodecRecommendation {
    /// The recommended codec.
    pub codec: CompressionCodec,
    /// Human-readable reason for this recommendation.
    pub reason: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// CompressionRegistryStats
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics across all registered codecs.
#[derive(Clone, Debug)]
pub struct CompressionRegistryStats {
    /// Number of codecs tracked by this registry.
    pub total_codecs: usize,
    /// Total bytes processed across all codecs.
    pub total_bytes_processed: u64,
    /// Codec with the lowest `avg_ratio` among those with at least one use.
    pub best_ratio_codec: Option<CompressionCodec>,
    /// Codec with the lowest `avg_compress_micros` among those with at least one use.
    pub fastest_codec: Option<CompressionCodec>,
}

// ─────────────────────────────────────────────────────────────────────────────
// StorageCompressionRegistry
// ─────────────────────────────────────────────────────────────────────────────

/// Registry that tracks compression codec profiles and recommends a codec based
/// on observed performance data and data characteristics.
///
/// # Example
/// ```
/// use ipfrs_storage::compression_registry::{
///     StorageCompressionRegistry, CompressionCodec, DataCharacteristics,
/// };
///
/// let mut registry = StorageCompressionRegistry::new();
/// registry.record_usage(CompressionCodec::Zstd, 1024, 400, 480, 90);
///
/// let chars = DataCharacteristics {
///     size_bytes: 4096,
///     is_text: false,
///     is_already_compressed: false,
///     latency_sensitive: false,
/// };
/// let rec = registry.recommend(&chars);
/// println!("Recommended: {:?} — {}", rec.codec, rec.reason);
/// ```
pub struct StorageCompressionRegistry {
    /// Per-codec runtime profiles.
    pub profiles: HashMap<CompressionCodec, CodecProfile>,
}

impl StorageCompressionRegistry {
    /// Create a new registry pre-populated with default profiles for all five codecs.
    ///
    /// Default values are based on typical real-world measurements:
    ///
    /// | Codec   | avg_ratio | compress µs | decompress µs |
    /// |---------|-----------|-------------|---------------|
    /// | Zstd    | 0.40      | 500         | 100           |
    /// | Lz4     | 0.60      | 50          | 30            |
    /// | Snappy  | 0.70      | 100         | 50            |
    /// | Brotli  | 0.35      | 2000        | 100           |
    /// | None    | 1.00      | 1           | 1             |
    pub fn new() -> Self {
        let defaults: &[(CompressionCodec, f64, u64, u64)] = &[
            (CompressionCodec::Zstd, 0.40, 500, 100),
            (CompressionCodec::Lz4, 0.60, 50, 30),
            (CompressionCodec::Snappy, 0.70, 100, 50),
            (CompressionCodec::Brotli, 0.35, 2000, 100),
            (CompressionCodec::None, 1.00, 1, 1),
        ];

        let profiles = defaults
            .iter()
            .map(
                |&(codec, avg_ratio, avg_compress_micros, avg_decompress_micros)| {
                    (
                        codec,
                        CodecProfile {
                            codec,
                            avg_ratio,
                            avg_compress_micros,
                            avg_decompress_micros,
                            total_bytes_processed: 0,
                            uses: 0,
                        },
                    )
                },
            )
            .collect();

        Self { profiles }
    }

    /// Record a compression event and update the codec profile.
    ///
    /// For the **first** use (`uses == 0` before the call) the measurements are
    /// stored directly.  For subsequent uses an exponential moving average
    /// (α = 0.1) is applied:
    ///
    /// ```text
    /// new_value = 0.9 * old_value + 0.1 * sample
    /// ```
    ///
    /// # Arguments
    /// * `codec` — the codec that was used.
    /// * `original_bytes` — uncompressed size.
    /// * `compressed_bytes` — size after compression.
    /// * `compress_micros` — time taken to compress.
    /// * `decompress_micros` — time taken to decompress.
    pub fn record_usage(
        &mut self,
        codec: CompressionCodec,
        original_bytes: u64,
        compressed_bytes: u64,
        compress_micros: u64,
        decompress_micros: u64,
    ) {
        let new_ratio = if original_bytes == 0 {
            1.0
        } else {
            compressed_bytes as f64 / original_bytes as f64
        };

        let profile = self.profiles.entry(codec).or_insert_with(|| CodecProfile {
            codec,
            avg_ratio: new_ratio,
            avg_compress_micros: compress_micros,
            avg_decompress_micros: decompress_micros,
            total_bytes_processed: 0,
            uses: 0,
        });

        if profile.uses == 0 {
            // First use: store directly without EMA.
            profile.avg_ratio = new_ratio;
            profile.avg_compress_micros = compress_micros;
            profile.avg_decompress_micros = decompress_micros;
        } else {
            // Subsequent uses: apply EMA with α = 0.1.
            profile.avg_ratio = 0.9 * profile.avg_ratio + 0.1 * new_ratio;
            profile.avg_compress_micros = (0.9 * profile.avg_compress_micros as f64
                + 0.1 * compress_micros as f64)
                .round() as u64;
            profile.avg_decompress_micros = (0.9 * profile.avg_decompress_micros as f64
                + 0.1 * decompress_micros as f64)
                .round() as u64;
        }

        profile.uses += 1;
        profile.total_bytes_processed += original_bytes;
    }

    /// Recommend a compression codec for the given data characteristics.
    ///
    /// Decision logic (in order of priority):
    /// 1. Already compressed → `None` ("already compressed").
    /// 2. Very small block (< 256 B) → `None` ("too small").
    /// 3. Latency-sensitive → `Lz4` ("latency optimized").
    /// 4. Text data → `Brotli` ("text compression").
    /// 5. Otherwise → codec with the best `efficiency_score` (excluding `None`)
    ///    ("best efficiency").
    pub fn recommend(&self, data: &DataCharacteristics) -> CodecRecommendation {
        if data.is_already_compressed {
            return CodecRecommendation {
                codec: CompressionCodec::None,
                reason: "already compressed".to_string(),
            };
        }

        if data.size_bytes < 256 {
            return CodecRecommendation {
                codec: CompressionCodec::None,
                reason: "too small".to_string(),
            };
        }

        if data.latency_sensitive {
            return CodecRecommendation {
                codec: CompressionCodec::Lz4,
                reason: "latency optimized".to_string(),
            };
        }

        if data.is_text {
            return CodecRecommendation {
                codec: CompressionCodec::Brotli,
                reason: "text compression".to_string(),
            };
        }

        // Select the codec with the highest efficiency_score, excluding None.
        let best = [
            CompressionCodec::Zstd,
            CompressionCodec::Lz4,
            CompressionCodec::Snappy,
            CompressionCodec::Brotli,
        ]
        .iter()
        .filter_map(|c| self.profiles.get(c).map(|p| (c, p.efficiency_score())))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        match best {
            Some((&codec, _)) => CodecRecommendation {
                codec,
                reason: "best efficiency".to_string(),
            },
            Option::None => CodecRecommendation {
                codec: CompressionCodec::None,
                reason: "best efficiency".to_string(),
            },
        }
    }

    /// Return a reference to the profile for the given codec, if present.
    pub fn get_profile(&self, codec: CompressionCodec) -> Option<&CodecProfile> {
        self.profiles.get(&codec)
    }

    /// Compute aggregate statistics across all registered profiles.
    pub fn stats(&self) -> CompressionRegistryStats {
        let total_codecs = self.profiles.len();
        let total_bytes_processed = self
            .profiles
            .values()
            .map(|p| p.total_bytes_processed)
            .sum();

        // Only consider codecs that have been used at least once.
        let used_profiles: Vec<&CodecProfile> =
            self.profiles.values().filter(|p| p.uses > 0).collect();

        let best_ratio_codec = used_profiles
            .iter()
            .min_by(|a, b| {
                a.avg_ratio
                    .partial_cmp(&b.avg_ratio)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.codec);

        let fastest_codec = used_profiles
            .iter()
            .min_by_key(|p| p.avg_compress_micros)
            .map(|p| p.codec);

        CompressionRegistryStats {
            total_codecs,
            total_bytes_processed,
            best_ratio_codec,
            fastest_codec,
        }
    }
}

impl Default for StorageCompressionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── new() ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_new_creates_five_profiles() {
        let registry = StorageCompressionRegistry::new();
        assert_eq!(registry.profiles.len(), 5);
    }

    #[test]
    fn test_new_contains_all_codecs() {
        let registry = StorageCompressionRegistry::new();
        assert!(registry.profiles.contains_key(&CompressionCodec::Zstd));
        assert!(registry.profiles.contains_key(&CompressionCodec::Lz4));
        assert!(registry.profiles.contains_key(&CompressionCodec::Snappy));
        assert!(registry.profiles.contains_key(&CompressionCodec::Brotli));
        assert!(registry.profiles.contains_key(&CompressionCodec::None));
    }

    #[test]
    fn test_new_default_ratios() {
        let registry = StorageCompressionRegistry::new();
        let zstd = registry.get_profile(CompressionCodec::Zstd).unwrap();
        let lz4 = registry.get_profile(CompressionCodec::Lz4).unwrap();
        let snappy = registry.get_profile(CompressionCodec::Snappy).unwrap();
        let brotli = registry.get_profile(CompressionCodec::Brotli).unwrap();
        let none = registry.get_profile(CompressionCodec::None).unwrap();

        assert!((zstd.avg_ratio - 0.40).abs() < 1e-9);
        assert!((lz4.avg_ratio - 0.60).abs() < 1e-9);
        assert!((snappy.avg_ratio - 0.70).abs() < 1e-9);
        assert!((brotli.avg_ratio - 0.35).abs() < 1e-9);
        assert!((none.avg_ratio - 1.00).abs() < 1e-9);
    }

    #[test]
    fn test_new_default_compress_micros() {
        let registry = StorageCompressionRegistry::new();
        assert_eq!(
            registry
                .get_profile(CompressionCodec::Zstd)
                .unwrap()
                .avg_compress_micros,
            500
        );
        assert_eq!(
            registry
                .get_profile(CompressionCodec::Lz4)
                .unwrap()
                .avg_compress_micros,
            50
        );
        assert_eq!(
            registry
                .get_profile(CompressionCodec::Snappy)
                .unwrap()
                .avg_compress_micros,
            100
        );
        assert_eq!(
            registry
                .get_profile(CompressionCodec::Brotli)
                .unwrap()
                .avg_compress_micros,
            2000
        );
        assert_eq!(
            registry
                .get_profile(CompressionCodec::None)
                .unwrap()
                .avg_compress_micros,
            1
        );
    }

    #[test]
    fn test_new_default_decompress_micros() {
        let registry = StorageCompressionRegistry::new();
        assert_eq!(
            registry
                .get_profile(CompressionCodec::Zstd)
                .unwrap()
                .avg_decompress_micros,
            100
        );
        assert_eq!(
            registry
                .get_profile(CompressionCodec::Lz4)
                .unwrap()
                .avg_decompress_micros,
            30
        );
        assert_eq!(
            registry
                .get_profile(CompressionCodec::Snappy)
                .unwrap()
                .avg_decompress_micros,
            50
        );
        assert_eq!(
            registry
                .get_profile(CompressionCodec::Brotli)
                .unwrap()
                .avg_decompress_micros,
            100
        );
        assert_eq!(
            registry
                .get_profile(CompressionCodec::None)
                .unwrap()
                .avg_decompress_micros,
            1
        );
    }

    #[test]
    fn test_new_uses_zero() {
        let registry = StorageCompressionRegistry::new();
        for profile in registry.profiles.values() {
            assert_eq!(
                profile.uses, 0,
                "Codec {:?} should start with uses=0",
                profile.codec
            );
        }
    }

    #[test]
    fn test_new_total_bytes_zero() {
        let registry = StorageCompressionRegistry::new();
        for profile in registry.profiles.values() {
            assert_eq!(profile.total_bytes_processed, 0);
        }
    }

    // ── record_usage — first use (direct set) ─────────────────────────────────

    #[test]
    fn test_record_usage_first_use_direct_ratio() {
        let mut registry = StorageCompressionRegistry::new();
        // Override default; first use should set directly.
        registry.record_usage(CompressionCodec::Zstd, 1000, 300, 400, 80);
        let p = registry.get_profile(CompressionCodec::Zstd).unwrap();
        assert!((p.avg_ratio - 0.30).abs() < 1e-9);
        assert_eq!(p.avg_compress_micros, 400);
        assert_eq!(p.avg_decompress_micros, 80);
        assert_eq!(p.uses, 1);
        assert_eq!(p.total_bytes_processed, 1000);
    }

    #[test]
    fn test_record_usage_first_use_updates_uses_and_bytes() {
        let mut registry = StorageCompressionRegistry::new();
        registry.record_usage(CompressionCodec::Lz4, 2048, 1200, 40, 25);
        let p = registry.get_profile(CompressionCodec::Lz4).unwrap();
        assert_eq!(p.uses, 1);
        assert_eq!(p.total_bytes_processed, 2048);
    }

    // ── record_usage — EMA update ─────────────────────────────────────────────

    #[test]
    fn test_record_usage_ema_ratio() {
        let mut registry = StorageCompressionRegistry::new();
        // First use: sets ratio to 0.30
        registry.record_usage(CompressionCodec::Zstd, 1000, 300, 500, 100);
        // Second use: ratio = 0.20, EMA -> 0.9*0.30 + 0.1*0.20 = 0.29
        registry.record_usage(CompressionCodec::Zstd, 1000, 200, 500, 100);
        let p = registry.get_profile(CompressionCodec::Zstd).unwrap();
        assert!((p.avg_ratio - 0.29).abs() < 1e-9);
    }

    #[test]
    fn test_record_usage_ema_compress_micros() {
        let mut registry = StorageCompressionRegistry::new();
        registry.record_usage(CompressionCodec::Zstd, 1000, 400, 500, 100);
        // Second: micros=100, EMA -> 0.9*500 + 0.1*100 = 460
        registry.record_usage(CompressionCodec::Zstd, 1000, 400, 100, 100);
        let p = registry.get_profile(CompressionCodec::Zstd).unwrap();
        assert_eq!(p.avg_compress_micros, 460);
    }

    #[test]
    fn test_record_usage_ema_decompress_micros() {
        let mut registry = StorageCompressionRegistry::new();
        registry.record_usage(CompressionCodec::Zstd, 1000, 400, 500, 200);
        // Second: decompress=20, EMA -> 0.9*200 + 0.1*20 = 182
        registry.record_usage(CompressionCodec::Zstd, 1000, 400, 500, 20);
        let p = registry.get_profile(CompressionCodec::Zstd).unwrap();
        assert_eq!(p.avg_decompress_micros, 182);
    }

    #[test]
    fn test_record_usage_accumulates_uses() {
        let mut registry = StorageCompressionRegistry::new();
        for _ in 0..5 {
            registry.record_usage(CompressionCodec::Snappy, 512, 350, 90, 40);
        }
        assert_eq!(
            registry.get_profile(CompressionCodec::Snappy).unwrap().uses,
            5
        );
    }

    #[test]
    fn test_record_usage_accumulates_total_bytes() {
        let mut registry = StorageCompressionRegistry::new();
        registry.record_usage(CompressionCodec::Snappy, 1000, 700, 90, 40);
        registry.record_usage(CompressionCodec::Snappy, 2000, 1400, 90, 40);
        assert_eq!(
            registry
                .get_profile(CompressionCodec::Snappy)
                .unwrap()
                .total_bytes_processed,
            3000
        );
    }

    #[test]
    fn test_record_usage_zero_original_bytes_ratio_is_one() {
        let mut registry = StorageCompressionRegistry::new();
        registry.record_usage(CompressionCodec::Zstd, 0, 0, 1, 1);
        let p = registry.get_profile(CompressionCodec::Zstd).unwrap();
        assert!((p.avg_ratio - 1.0).abs() < 1e-9);
    }

    // ── recommend ─────────────────────────────────────────────────────────────

    #[test]
    fn test_recommend_already_compressed_returns_none() {
        let registry = StorageCompressionRegistry::new();
        let data = DataCharacteristics {
            size_bytes: 8192,
            is_text: false,
            is_already_compressed: true,
            latency_sensitive: false,
        };
        let rec = registry.recommend(&data);
        assert_eq!(rec.codec, CompressionCodec::None);
        assert_eq!(rec.reason, "already compressed");
    }

    #[test]
    fn test_recommend_small_data_returns_none() {
        let registry = StorageCompressionRegistry::new();
        let data = DataCharacteristics {
            size_bytes: 100,
            is_text: false,
            is_already_compressed: false,
            latency_sensitive: false,
        };
        let rec = registry.recommend(&data);
        assert_eq!(rec.codec, CompressionCodec::None);
        assert_eq!(rec.reason, "too small");
    }

    #[test]
    fn test_recommend_exactly_256_bytes_not_too_small() {
        let registry = StorageCompressionRegistry::new();
        let data = DataCharacteristics {
            size_bytes: 256,
            is_text: false,
            is_already_compressed: false,
            latency_sensitive: false,
        };
        let rec = registry.recommend(&data);
        // Should not be "too small" — falls through to efficiency selection.
        assert_ne!(rec.reason, "too small");
    }

    #[test]
    fn test_recommend_latency_sensitive_returns_lz4() {
        let registry = StorageCompressionRegistry::new();
        let data = DataCharacteristics {
            size_bytes: 4096,
            is_text: false,
            is_already_compressed: false,
            latency_sensitive: true,
        };
        let rec = registry.recommend(&data);
        assert_eq!(rec.codec, CompressionCodec::Lz4);
        assert_eq!(rec.reason, "latency optimized");
    }

    #[test]
    fn test_recommend_text_returns_brotli() {
        let registry = StorageCompressionRegistry::new();
        let data = DataCharacteristics {
            size_bytes: 4096,
            is_text: true,
            is_already_compressed: false,
            latency_sensitive: false,
        };
        let rec = registry.recommend(&data);
        assert_eq!(rec.codec, CompressionCodec::Brotli);
        assert_eq!(rec.reason, "text compression");
    }

    #[test]
    fn test_recommend_already_compressed_takes_priority_over_small() {
        let registry = StorageCompressionRegistry::new();
        let data = DataCharacteristics {
            size_bytes: 10,
            is_text: false,
            is_already_compressed: true,
            latency_sensitive: false,
        };
        let rec = registry.recommend(&data);
        assert_eq!(rec.reason, "already compressed");
    }

    #[test]
    fn test_recommend_best_efficiency_excludes_none() {
        let registry = StorageCompressionRegistry::new();
        let data = DataCharacteristics {
            size_bytes: 4096,
            is_text: false,
            is_already_compressed: false,
            latency_sensitive: false,
        };
        let rec = registry.recommend(&data);
        assert_ne!(rec.codec, CompressionCodec::None);
        assert_eq!(rec.reason, "best efficiency");
    }

    #[test]
    fn test_recommend_best_efficiency_selects_highest_score() {
        // Manipulate profiles so Lz4 has an artificially great efficiency score.
        let mut registry = StorageCompressionRegistry::new();
        // First use sets directly — give Lz4 a very low ratio and very fast time.
        registry.record_usage(CompressionCodec::Lz4, 10000, 100, 1, 1);

        let data = DataCharacteristics {
            size_bytes: 8192,
            is_text: false,
            is_already_compressed: false,
            latency_sensitive: false,
        };
        let rec = registry.recommend(&data);
        // Lz4 should win with ratio=0.01, compress_micros=1 → score=(1-0.01)/(1+1)=0.495
        assert_eq!(rec.codec, CompressionCodec::Lz4);
    }

    // ── efficiency_score ─────────────────────────────────────────────────────

    #[test]
    fn test_efficiency_score_formula() {
        let profile = CodecProfile {
            codec: CompressionCodec::Zstd,
            avg_ratio: 0.4,
            avg_compress_micros: 499,
            avg_decompress_micros: 100,
            total_bytes_processed: 0,
            uses: 0,
        };
        // (1.0 - 0.4) / (499.0 + 1.0) = 0.6 / 500.0 = 0.0012
        let expected = 0.6 / 500.0;
        assert!((profile.efficiency_score() - expected).abs() < 1e-12);
    }

    #[test]
    fn test_efficiency_score_none_codec_is_zero() {
        let profile = CodecProfile {
            codec: CompressionCodec::None,
            avg_ratio: 1.0,
            avg_compress_micros: 1,
            avg_decompress_micros: 1,
            total_bytes_processed: 0,
            uses: 0,
        };
        // (1.0 - 1.0) / 2.0 = 0.0
        assert!((profile.efficiency_score() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_efficiency_score_higher_for_better_ratio() {
        let good = CodecProfile {
            codec: CompressionCodec::Brotli,
            avg_ratio: 0.2,
            avg_compress_micros: 1000,
            avg_decompress_micros: 100,
            total_bytes_processed: 0,
            uses: 0,
        };
        let bad = CodecProfile {
            codec: CompressionCodec::Snappy,
            avg_ratio: 0.8,
            avg_compress_micros: 1000,
            avg_decompress_micros: 100,
            total_bytes_processed: 0,
            uses: 0,
        };
        assert!(good.efficiency_score() > bad.efficiency_score());
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_total_codecs() {
        let registry = StorageCompressionRegistry::new();
        assert_eq!(registry.stats().total_codecs, 5);
    }

    #[test]
    fn test_stats_total_bytes_processed() {
        let mut registry = StorageCompressionRegistry::new();
        registry.record_usage(CompressionCodec::Zstd, 1000, 400, 500, 100);
        registry.record_usage(CompressionCodec::Lz4, 2000, 1200, 50, 30);
        assert_eq!(registry.stats().total_bytes_processed, 3000);
    }

    #[test]
    fn test_stats_best_ratio_codec_ignores_uses_zero() {
        // No codec has been used yet → best_ratio_codec should be None.
        let registry = StorageCompressionRegistry::new();
        let s = registry.stats();
        assert!(s.best_ratio_codec.is_none(), "All uses=0, expected None");
    }

    #[test]
    fn test_stats_best_ratio_codec_after_use() {
        let mut registry = StorageCompressionRegistry::new();
        // Give Brotli a use with a great ratio.
        registry.record_usage(CompressionCodec::Brotli, 1000, 100, 2000, 100);
        let s = registry.stats();
        assert_eq!(s.best_ratio_codec, Some(CompressionCodec::Brotli));
    }

    #[test]
    fn test_stats_fastest_codec_ignores_uses_zero() {
        let registry = StorageCompressionRegistry::new();
        let s = registry.stats();
        assert!(s.fastest_codec.is_none(), "All uses=0, expected None");
    }

    #[test]
    fn test_stats_fastest_codec_after_use() {
        let mut registry = StorageCompressionRegistry::new();
        // Give None codec a use (compress_micros=1).
        registry.record_usage(CompressionCodec::None, 1000, 1000, 1, 1);
        let s = registry.stats();
        assert_eq!(s.fastest_codec, Some(CompressionCodec::None));
    }

    #[test]
    fn test_stats_best_ratio_selects_lowest_ratio_among_used() {
        let mut registry = StorageCompressionRegistry::new();
        // Use Zstd with ratio 0.50 and Lz4 with ratio 0.40.
        registry.record_usage(CompressionCodec::Zstd, 1000, 500, 500, 100);
        registry.record_usage(CompressionCodec::Lz4, 1000, 400, 50, 30);
        let s = registry.stats();
        assert_eq!(s.best_ratio_codec, Some(CompressionCodec::Lz4));
    }

    #[test]
    fn test_get_profile_existing_codec() {
        let registry = StorageCompressionRegistry::new();
        assert!(registry.get_profile(CompressionCodec::Zstd).is_some());
    }

    #[test]
    fn test_default_impl() {
        let registry = StorageCompressionRegistry::default();
        assert_eq!(registry.profiles.len(), 5);
    }
}
