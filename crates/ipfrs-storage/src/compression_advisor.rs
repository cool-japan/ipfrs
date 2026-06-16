//! Block compression advisor — analyses access patterns and sizes to recommend
//! optimal compression codecs and tracks achieved compression ratios.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// CompressionCodec
// ─────────────────────────────────────────────────────────────────────────────

/// Supported compression codecs that the advisor can recommend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompressionCodec {
    /// No compression applied.
    None,
    /// LZ4 — very fast with moderate ratio.
    Lz4,
    /// Zstd — balanced speed and compression ratio.
    Zstd,
    /// Snappy — fast with lower compression ratio than Lz4.
    Snappy,
    /// Brotli — slow but best compression ratio.
    Brotli,
}

impl CompressionCodec {
    /// Typical compression ratio achieved by this codec (raw / compressed).
    ///
    /// A value of `1.0` means no compression gain.
    pub fn typical_ratio(&self) -> f64 {
        match self {
            Self::None => 1.0,
            Self::Lz4 => 1.5,
            Self::Zstd => 2.5,
            Self::Snappy => 1.3,
            Self::Brotli => 3.0,
        }
    }

    /// Relative compression speed on a scale of 0–100 (higher = faster).
    pub fn compression_speed(&self) -> u32 {
        match self {
            Self::None => 100,
            Self::Lz4 => 80,
            Self::Zstd => 40,
            Self::Snappy => 90,
            Self::Brotli => 10,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BlockProfile
// ─────────────────────────────────────────────────────────────────────────────

/// Profile tracking the size and access characteristics of a single block.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockProfile {
    /// Content identifier of the block.
    pub cid: String,
    /// Size of the block before any compression is applied, in bytes.
    pub raw_size_bytes: u64,
    /// Size after compression, if the block has been compressed.
    pub compressed_size_bytes: Option<u64>,
    /// Codec used for the current compressed copy, if any.
    pub codec: Option<CompressionCodec>,
    /// Number of times this block has been accessed.
    pub access_count: u64,
}

impl BlockProfile {
    /// Returns the achieved compression ratio (`raw / compressed`).
    ///
    /// Returns `1.0` when compression data is unavailable.
    pub fn compression_ratio(&self) -> f64 {
        match self.compressed_size_bytes {
            Some(compressed) if compressed > 0 => self.raw_size_bytes as f64 / compressed as f64,
            _ => 1.0,
        }
    }

    /// Returns the number of bytes saved by compression (`raw − compressed`).
    ///
    /// Returns `0` when the block has not been compressed or when compression
    /// actually expanded the block (saturating subtraction).
    pub fn space_savings_bytes(&self) -> u64 {
        match self.compressed_size_bytes {
            Some(compressed) => self.raw_size_bytes.saturating_sub(compressed),
            None => 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AdvisorRecommendation
// ─────────────────────────────────────────────────────────────────────────────

/// A concrete codec recommendation produced by [`BlockCompressionAdvisor`].
#[derive(Debug, Clone, PartialEq)]
pub struct AdvisorRecommendation {
    /// Content identifier of the block this recommendation applies to.
    pub cid: String,
    /// Codec the advisor suggests using.
    pub recommended_codec: CompressionCodec,
    /// Estimated bytes that would be saved by applying the recommended codec.
    pub estimated_savings_bytes: u64,
    /// Human-readable explanation of why this codec was chosen.
    pub reason: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// AdvisorConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration knobs for the [`BlockCompressionAdvisor`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdvisorConfig {
    /// Blocks smaller than this threshold (in bytes) are not worth compressing.
    ///
    /// Default: 1 024 bytes.
    pub min_size_for_compression: u64,
    /// Blocks accessed at least this many times are considered *hot* and favour
    /// speed over compression ratio.
    ///
    /// Default: 100 accesses.
    pub hot_access_threshold: u64,
    /// Blocks accessed at most this many times are considered *cold* and favour
    /// the best compression ratio.
    ///
    /// Default: 5 accesses.
    pub cold_access_threshold: u64,
}

impl Default for AdvisorConfig {
    fn default() -> Self {
        Self {
            min_size_for_compression: 1_024,
            hot_access_threshold: 100,
            cold_access_threshold: 5,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BlockCompressionAdvisor
// ─────────────────────────────────────────────────────────────────────────────

/// Analyses block access patterns and sizes to recommend optimal compression
/// codecs and tracks achieved compression ratios.
pub struct BlockCompressionAdvisor {
    /// Per-block profiles keyed by CID.
    pub profiles: HashMap<String, BlockProfile>,
    /// Configuration thresholds used by the recommendation logic.
    pub config: AdvisorConfig,
}

impl BlockCompressionAdvisor {
    /// Creates a new advisor with the supplied configuration.
    pub fn new(config: AdvisorConfig) -> Self {
        Self {
            profiles: HashMap::new(),
            config,
        }
    }

    /// Registers a block (or updates its `raw_size_bytes` and `access_count`
    /// if it already exists).
    pub fn register_block(&mut self, cid: String, raw_size_bytes: u64, access_count: u64) {
        let profile = self
            .profiles
            .entry(cid.clone())
            .or_insert_with(|| BlockProfile {
                cid: cid.clone(),
                raw_size_bytes,
                compressed_size_bytes: None,
                codec: None,
                access_count,
            });
        profile.raw_size_bytes = raw_size_bytes;
        profile.access_count = access_count;
    }

    /// Records the outcome of a compression operation for an existing block.
    ///
    /// Has no effect when the CID is unknown.
    pub fn record_compression(
        &mut self,
        cid: &str,
        compressed_size_bytes: u64,
        codec: CompressionCodec,
    ) {
        if let Some(profile) = self.profiles.get_mut(cid) {
            profile.compressed_size_bytes = Some(compressed_size_bytes);
            profile.codec = Some(codec);
        }
    }

    /// Returns a codec recommendation for the block identified by `cid`.
    ///
    /// Returns `None` when:
    /// - the block is not registered, or
    /// - the block's raw size is below [`AdvisorConfig::min_size_for_compression`].
    pub fn recommend(&self, cid: &str) -> Option<AdvisorRecommendation> {
        let profile = self.profiles.get(cid)?;

        if profile.raw_size_bytes < self.config.min_size_for_compression {
            return None;
        }

        let (codec, reason) = if profile.access_count >= self.config.hot_access_threshold {
            (
                CompressionCodec::Lz4,
                format!(
                    "Block is hot ({} accesses >= threshold {}): prefer speed with Lz4",
                    profile.access_count, self.config.hot_access_threshold
                ),
            )
        } else if profile.access_count <= self.config.cold_access_threshold {
            (
                CompressionCodec::Brotli,
                format!(
                    "Block is cold ({} accesses <= threshold {}): prefer ratio with Brotli",
                    profile.access_count, self.config.cold_access_threshold
                ),
            )
        } else {
            (
                CompressionCodec::Zstd,
                format!(
                    "Block is warm ({} accesses): balanced Zstd recommended",
                    profile.access_count
                ),
            )
        };

        let typical_ratio = codec.typical_ratio();
        // estimated_savings = raw * (1 − 1 / typical_ratio)
        let savings_fraction = 1.0 - 1.0 / typical_ratio;
        let estimated_savings_bytes = (profile.raw_size_bytes as f64 * savings_fraction) as u64;

        Some(AdvisorRecommendation {
            cid: cid.to_string(),
            recommended_codec: codec,
            estimated_savings_bytes,
            reason,
        })
    }

    /// Returns recommendations for every registered block that qualifies for
    /// compression, sorted by `estimated_savings_bytes` in descending order.
    pub fn recommend_all(&self) -> Vec<AdvisorRecommendation> {
        let mut recs: Vec<AdvisorRecommendation> = self
            .profiles
            .keys()
            .filter_map(|cid| self.recommend(cid))
            .collect();
        recs.sort_by_key(|r| std::cmp::Reverse(r.estimated_savings_bytes));
        recs
    }

    /// Returns the total bytes saved across all profiles that have been
    /// compressed.
    pub fn total_space_savings(&self) -> u64 {
        self.profiles
            .values()
            .map(|p| p.space_savings_bytes())
            .fold(0u64, |acc, s| acc.saturating_add(s))
    }

    /// Returns an immutable reference to the profile for `cid`, if it exists.
    pub fn stats_for(&self, cid: &str) -> Option<&BlockProfile> {
        self.profiles.get(cid)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_advisor() -> BlockCompressionAdvisor {
        BlockCompressionAdvisor::new(AdvisorConfig::default())
    }

    // ── CompressionCodec::typical_ratio ──────────────────────────────────────

    #[test]
    fn typical_ratio_none_is_one() {
        assert!((CompressionCodec::None.typical_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn typical_ratio_lz4() {
        assert!((CompressionCodec::Lz4.typical_ratio() - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn typical_ratio_zstd() {
        assert!((CompressionCodec::Zstd.typical_ratio() - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn typical_ratio_snappy() {
        assert!((CompressionCodec::Snappy.typical_ratio() - 1.3).abs() < f64::EPSILON);
    }

    #[test]
    fn typical_ratio_brotli() {
        assert!((CompressionCodec::Brotli.typical_ratio() - 3.0).abs() < f64::EPSILON);
    }

    // ── CompressionCodec::compression_speed ordering ─────────────────────────

    #[test]
    fn compression_speed_ordering() {
        // None > Snappy > Lz4 > Zstd > Brotli
        assert!(
            CompressionCodec::None.compression_speed()
                > CompressionCodec::Snappy.compression_speed()
        );
        assert!(
            CompressionCodec::Snappy.compression_speed()
                > CompressionCodec::Lz4.compression_speed()
        );
        assert!(
            CompressionCodec::Lz4.compression_speed() > CompressionCodec::Zstd.compression_speed()
        );
        assert!(
            CompressionCodec::Zstd.compression_speed()
                > CompressionCodec::Brotli.compression_speed()
        );
    }

    #[test]
    fn compression_speed_exact_values() {
        assert_eq!(CompressionCodec::None.compression_speed(), 100);
        assert_eq!(CompressionCodec::Lz4.compression_speed(), 80);
        assert_eq!(CompressionCodec::Zstd.compression_speed(), 40);
        assert_eq!(CompressionCodec::Snappy.compression_speed(), 90);
        assert_eq!(CompressionCodec::Brotli.compression_speed(), 10);
    }

    // ── BlockProfile helpers ──────────────────────────────────────────────────

    #[test]
    fn compression_ratio_without_compressed_size() {
        let profile = BlockProfile {
            cid: "bafy1".to_string(),
            raw_size_bytes: 4096,
            compressed_size_bytes: None,
            codec: None,
            access_count: 1,
        };
        assert!((profile.compression_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn compression_ratio_with_compressed_size() {
        let profile = BlockProfile {
            cid: "bafy2".to_string(),
            raw_size_bytes: 4096,
            compressed_size_bytes: Some(2048),
            codec: Some(CompressionCodec::Zstd),
            access_count: 10,
        };
        assert!((profile.compression_ratio() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn space_savings_bytes_without_compression() {
        let profile = BlockProfile {
            cid: "bafy3".to_string(),
            raw_size_bytes: 8192,
            compressed_size_bytes: None,
            codec: None,
            access_count: 3,
        };
        assert_eq!(profile.space_savings_bytes(), 0);
    }

    #[test]
    fn space_savings_bytes_with_compression() {
        let profile = BlockProfile {
            cid: "bafy4".to_string(),
            raw_size_bytes: 8192,
            compressed_size_bytes: Some(3000),
            codec: Some(CompressionCodec::Brotli),
            access_count: 1,
        };
        assert_eq!(profile.space_savings_bytes(), 8192 - 3000);
    }

    #[test]
    fn space_savings_bytes_saturates_when_compressed_larger() {
        let profile = BlockProfile {
            cid: "bafy5".to_string(),
            raw_size_bytes: 100,
            compressed_size_bytes: Some(150),
            codec: Some(CompressionCodec::Lz4),
            access_count: 200,
        };
        assert_eq!(profile.space_savings_bytes(), 0);
    }

    // ── Advisor recommendation logic ──────────────────────────────────────────

    #[test]
    fn recommend_hot_block_returns_lz4() {
        let mut adv = default_advisor();
        // hot_access_threshold = 100
        adv.register_block("cid_hot".to_string(), 65_536, 150);
        let rec = adv.recommend("cid_hot").expect("should recommend");
        assert_eq!(rec.recommended_codec, CompressionCodec::Lz4);
    }

    #[test]
    fn recommend_cold_block_returns_brotli() {
        let mut adv = default_advisor();
        // cold_access_threshold = 5
        adv.register_block("cid_cold".to_string(), 65_536, 2);
        let rec = adv.recommend("cid_cold").expect("should recommend");
        assert_eq!(rec.recommended_codec, CompressionCodec::Brotli);
    }

    #[test]
    fn recommend_medium_block_returns_zstd() {
        let mut adv = default_advisor();
        // 5 < 50 < 100 → warm
        adv.register_block("cid_medium".to_string(), 65_536, 50);
        let rec = adv.recommend("cid_medium").expect("should recommend");
        assert_eq!(rec.recommended_codec, CompressionCodec::Zstd);
    }

    #[test]
    fn recommend_too_small_block_returns_none() {
        let mut adv = default_advisor();
        // min_size_for_compression = 1024; this block is 512 bytes
        adv.register_block("cid_tiny".to_string(), 512, 50);
        assert!(adv.recommend("cid_tiny").is_none());
    }

    #[test]
    fn recommend_unknown_cid_returns_none() {
        let adv = default_advisor();
        assert!(adv.recommend("does_not_exist").is_none());
    }

    #[test]
    fn recommend_exactly_at_hot_threshold() {
        let mut adv = default_advisor();
        adv.register_block("cid_exact_hot".to_string(), 8_192, 100);
        let rec = adv.recommend("cid_exact_hot").expect("should recommend");
        assert_eq!(rec.recommended_codec, CompressionCodec::Lz4);
    }

    #[test]
    fn recommend_exactly_at_cold_threshold() {
        let mut adv = default_advisor();
        adv.register_block("cid_exact_cold".to_string(), 8_192, 5);
        let rec = adv.recommend("cid_exact_cold").expect("should recommend");
        assert_eq!(rec.recommended_codec, CompressionCodec::Brotli);
    }

    // ── record_compression updates profile ───────────────────────────────────

    #[test]
    fn record_compression_updates_profile() {
        let mut adv = default_advisor();
        adv.register_block("cid_comp".to_string(), 10_000, 10);
        adv.record_compression("cid_comp", 4_000, CompressionCodec::Zstd);

        let profile = adv.stats_for("cid_comp").expect("profile should exist");
        assert_eq!(profile.compressed_size_bytes, Some(4_000));
        assert_eq!(profile.codec, Some(CompressionCodec::Zstd));
    }

    #[test]
    fn record_compression_on_unknown_cid_does_nothing() {
        let mut adv = default_advisor();
        // Should not panic; just no-op.
        adv.record_compression("ghost", 100, CompressionCodec::Lz4);
        assert!(adv.stats_for("ghost").is_none());
    }

    // ── recommend_all ─────────────────────────────────────────────────────────

    #[test]
    fn recommend_all_sorted_by_savings_desc() {
        let mut adv = default_advisor();
        // large block (more savings)
        adv.register_block("cid_large".to_string(), 1_000_000, 50);
        // small block (less savings)
        adv.register_block("cid_small".to_string(), 2_048, 50);
        // too small — should be excluded
        adv.register_block("cid_micro".to_string(), 256, 50);

        let recs = adv.recommend_all();
        assert_eq!(recs.len(), 2);
        assert!(
            recs[0].estimated_savings_bytes >= recs[1].estimated_savings_bytes,
            "results must be sorted descending by estimated_savings_bytes"
        );
        assert_eq!(recs[0].cid, "cid_large");
    }

    // ── total_space_savings ───────────────────────────────────────────────────

    #[test]
    fn total_space_savings_sums_all_profiles() {
        let mut adv = default_advisor();
        adv.register_block("a".to_string(), 10_000, 10);
        adv.register_block("b".to_string(), 20_000, 10);
        adv.record_compression("a", 4_000, CompressionCodec::Zstd); // saves 6 000
        adv.record_compression("b", 8_000, CompressionCodec::Brotli); // saves 12 000

        assert_eq!(adv.total_space_savings(), 6_000 + 12_000);
    }

    #[test]
    fn total_space_savings_zero_when_no_compressions() {
        let mut adv = default_advisor();
        adv.register_block("x".to_string(), 5_000, 1);
        assert_eq!(adv.total_space_savings(), 0);
    }

    // ── estimated savings formula ─────────────────────────────────────────────

    #[test]
    fn estimated_savings_formula_brotli() {
        let mut adv = default_advisor();
        let raw = 30_000u64;
        adv.register_block("cid_brotli".to_string(), raw, 1);
        let rec = adv.recommend("cid_brotli").expect("should recommend");
        assert_eq!(rec.recommended_codec, CompressionCodec::Brotli);

        // expected = raw * (1 - 1/3.0) = 30000 * 0.6666... = 20000
        let expected = (raw as f64 * (1.0 - 1.0 / 3.0)) as u64;
        assert_eq!(rec.estimated_savings_bytes, expected);
    }
}
