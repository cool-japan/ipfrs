//! Content Identifier (CID) wrapper and utilities
//!
//! Provides CID generation, parsing, and multibase encoding/decoding support.

use crate::error::{Error, Result};
pub use ::cid::Cid;
use multibase::Base;
use multihash_codetable::{Code, MultihashDigest};
use serde::{Deserialize, Serialize};

/// Hash algorithm to use for CID generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HashAlgorithm {
    /// SHA2-256 (default, most compatible with IPFS)
    #[default]
    Sha256,
    /// SHA2-512 (64-byte hash for enhanced security)
    Sha512,
    /// SHA3-256 (Keccak-based, secure)
    Sha3_256,
    /// SHA3-512 (Keccak-based, 64-byte hash)
    Sha3_512,
    /// BLAKE2b-256 (fast, 32-byte hash, optimized for 64-bit)
    Blake2b256,
    /// BLAKE2b-512 (fast, 64-byte hash, maximum security)
    Blake2b512,
    /// BLAKE2s-256 (fast, 32-byte hash, optimized for 8-32 bit)
    Blake2s256,
    /// BLAKE3 (fastest, 32-byte hash, modern cryptographic design)
    Blake3,
}

impl HashAlgorithm {
    /// Get the multihash code for this algorithm
    #[inline]
    pub fn code(&self) -> Code {
        match self {
            HashAlgorithm::Sha256 => Code::Sha2_256,
            HashAlgorithm::Sha512 => Code::Sha2_512,
            HashAlgorithm::Sha3_256 => Code::Sha3_256,
            HashAlgorithm::Sha3_512 => Code::Sha3_512,
            HashAlgorithm::Blake2b256 => Code::Blake2b256,
            HashAlgorithm::Blake2b512 => Code::Blake2b512,
            HashAlgorithm::Blake2s256 => Code::Blake2s256,
            HashAlgorithm::Blake3 => Code::Blake3_256,
        }
    }

    /// Get the name of the hash algorithm
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            HashAlgorithm::Sha256 => "SHA2-256",
            HashAlgorithm::Sha512 => "SHA2-512",
            HashAlgorithm::Sha3_256 => "SHA3-256",
            HashAlgorithm::Sha3_512 => "SHA3-512",
            HashAlgorithm::Blake2b256 => "BLAKE2b-256",
            HashAlgorithm::Blake2b512 => "BLAKE2b-512",
            HashAlgorithm::Blake2s256 => "BLAKE2s-256",
            HashAlgorithm::Blake3 => "BLAKE3",
        }
    }

    /// Get the expected hash output size in bytes
    #[inline]
    pub const fn hash_size(&self) -> usize {
        match self {
            HashAlgorithm::Sha256 => 32,
            HashAlgorithm::Sha512 => 64,
            HashAlgorithm::Sha3_256 => 32,
            HashAlgorithm::Sha3_512 => 64,
            HashAlgorithm::Blake2b256 => 32,
            HashAlgorithm::Blake2b512 => 64,
            HashAlgorithm::Blake2s256 => 32,
            HashAlgorithm::Blake3 => 32,
        }
    }

    /// Check if this is a SHA-family algorithm
    #[inline]
    pub const fn is_sha(&self) -> bool {
        matches!(
            self,
            HashAlgorithm::Sha256
                | HashAlgorithm::Sha512
                | HashAlgorithm::Sha3_256
                | HashAlgorithm::Sha3_512
        )
    }

    /// Check if this is a BLAKE-family algorithm
    #[inline]
    pub const fn is_blake(&self) -> bool {
        matches!(
            self,
            HashAlgorithm::Blake2b256
                | HashAlgorithm::Blake2b512
                | HashAlgorithm::Blake2s256
                | HashAlgorithm::Blake3
        )
    }

    /// Check if this algorithm produces a 32-byte (256-bit) hash
    #[inline]
    pub const fn is_256_bit(&self) -> bool {
        matches!(
            self,
            HashAlgorithm::Sha256
                | HashAlgorithm::Sha3_256
                | HashAlgorithm::Blake2b256
                | HashAlgorithm::Blake2s256
                | HashAlgorithm::Blake3
        )
    }

    /// Check if this algorithm produces a 64-byte (512-bit) hash
    #[inline]
    pub const fn is_512_bit(&self) -> bool {
        matches!(
            self,
            HashAlgorithm::Sha512 | HashAlgorithm::Sha3_512 | HashAlgorithm::Blake2b512
        )
    }

    /// Check if this algorithm has SIMD support on the current platform
    #[inline]
    pub fn has_simd_support(&self) -> bool {
        match self {
            // SHA2 has SIMD on x86_64 (SHA-NI, AVX2) and ARM (NEON)
            HashAlgorithm::Sha256 | HashAlgorithm::Sha512 => {
                cfg!(target_arch = "x86_64") || cfg!(target_arch = "aarch64")
            }
            // SHA3 has optimized implementation
            HashAlgorithm::Sha3_256 | HashAlgorithm::Sha3_512 => true,
            // BLAKE2 has SIMD support
            HashAlgorithm::Blake2b256 | HashAlgorithm::Blake2b512 | HashAlgorithm::Blake2s256 => {
                true
            }
            // BLAKE3 has built-in SIMD
            HashAlgorithm::Blake3 => true,
        }
    }

    /// Get all available hash algorithms
    pub fn all() -> &'static [HashAlgorithm] {
        &[
            HashAlgorithm::Sha256,
            HashAlgorithm::Sha512,
            HashAlgorithm::Sha3_256,
            HashAlgorithm::Sha3_512,
            HashAlgorithm::Blake2b256,
            HashAlgorithm::Blake2b512,
            HashAlgorithm::Blake2s256,
            HashAlgorithm::Blake3,
        ]
    }
}

impl std::fmt::Display for HashAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::str::FromStr for HashAlgorithm {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_uppercase().as_str() {
            "SHA256" | "SHA2-256" | "SHA-256" => Ok(HashAlgorithm::Sha256),
            "SHA512" | "SHA2-512" | "SHA-512" => Ok(HashAlgorithm::Sha512),
            "SHA3-256" | "SHA3_256" => Ok(HashAlgorithm::Sha3_256),
            "SHA3-512" | "SHA3_512" => Ok(HashAlgorithm::Sha3_512),
            "BLAKE2B256" | "BLAKE2B-256" | "BLAKE2B_256" => Ok(HashAlgorithm::Blake2b256),
            "BLAKE2B512" | "BLAKE2B-512" | "BLAKE2B_512" => Ok(HashAlgorithm::Blake2b512),
            "BLAKE2S256" | "BLAKE2S-256" | "BLAKE2S_256" => Ok(HashAlgorithm::Blake2s256),
            "BLAKE3" | "BLAKE3-256" | "BLAKE3_256" => Ok(HashAlgorithm::Blake3),
            _ => Err(Error::InvalidData(format!("Unknown hash algorithm: {}", s))),
        }
    }
}

/// Multibase encoding options for CID string representation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MultibaseEncoding {
    /// Base32 lower case (default for CIDv1, starts with 'b')
    #[default]
    Base32Lower,
    /// Base58 Bitcoin (classic IPFS format, starts with 'z' for CIDv1)
    Base58Btc,
    /// Base64 standard (starts with 'm')
    Base64,
    /// Base64 URL-safe (starts with 'u')
    Base64Url,
    /// Base32 upper case (starts with 'B')
    Base32Upper,
}

impl MultibaseEncoding {
    /// Get the multibase base for this encoding
    #[inline]
    pub fn base(&self) -> Base {
        match self {
            MultibaseEncoding::Base32Lower => Base::Base32Lower,
            MultibaseEncoding::Base58Btc => Base::Base58Btc,
            MultibaseEncoding::Base64 => Base::Base64,
            MultibaseEncoding::Base64Url => Base::Base64Url,
            MultibaseEncoding::Base32Upper => Base::Base32Upper,
        }
    }

    /// Get the multibase prefix character for this encoding
    #[inline]
    pub const fn prefix(&self) -> char {
        match self {
            MultibaseEncoding::Base32Lower => 'b',
            MultibaseEncoding::Base58Btc => 'z',
            MultibaseEncoding::Base64 => 'm',
            MultibaseEncoding::Base64Url => 'u',
            MultibaseEncoding::Base32Upper => 'B',
        }
    }

    /// Get the name of this encoding
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            MultibaseEncoding::Base32Lower => "base32 (lowercase)",
            MultibaseEncoding::Base58Btc => "base58btc",
            MultibaseEncoding::Base64 => "base64",
            MultibaseEncoding::Base64Url => "base64url",
            MultibaseEncoding::Base32Upper => "base32 (uppercase)",
        }
    }

    /// Detect encoding from a CID string prefix
    #[inline]
    pub const fn from_prefix(c: char) -> Option<Self> {
        match c {
            'b' => Some(MultibaseEncoding::Base32Lower),
            'z' => Some(MultibaseEncoding::Base58Btc),
            'm' => Some(MultibaseEncoding::Base64),
            'u' => Some(MultibaseEncoding::Base64Url),
            'B' => Some(MultibaseEncoding::Base32Upper),
            _ => None,
        }
    }

    /// Get all available multibase encodings
    pub const fn all() -> &'static [MultibaseEncoding] {
        &[
            MultibaseEncoding::Base32Lower,
            MultibaseEncoding::Base58Btc,
            MultibaseEncoding::Base64,
            MultibaseEncoding::Base64Url,
            MultibaseEncoding::Base32Upper,
        ]
    }
}

impl std::fmt::Display for MultibaseEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// CID builder for creating content identifiers
#[derive(Debug, Clone)]
pub struct CidBuilder {
    version: cid::Version,
    codec: u64,
    hash_algorithm: HashAlgorithm,
}

impl Default for CidBuilder {
    fn default() -> Self {
        Self {
            version: cid::Version::V1,
            codec: 0x55, // raw codec
            hash_algorithm: HashAlgorithm::Sha256,
        }
    }
}

impl CidBuilder {
    /// Creates a new `CidBuilder` with default settings.
    ///
    /// Uses CIDv1, raw codec (0x55), and SHA2-256 hash algorithm by default.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder configured for CIDv0 (legacy IPFS format)
    ///
    /// CIDv0 uses:
    /// - SHA2-256 hash algorithm
    /// - DAG-PB codec (0x70)
    /// - Base58btc encoding (implicit, starts with "Qm")
    pub fn v0() -> Self {
        Self {
            version: cid::Version::V0,
            codec: 0x70, // DAG-PB
            hash_algorithm: HashAlgorithm::Sha256,
        }
    }

    /// Sets the CID version (V0 or V1).
    ///
    /// Note: CIDv0 has restrictions (must use SHA2-256 and DAG-PB codec).
    pub fn version(mut self, version: cid::Version) -> Self {
        self.version = version;
        self
    }

    /// Sets the codec for the CID.
    ///
    /// Common codecs:
    /// - `0x55` - raw binary
    /// - `0x70` - DAG-PB (IPFS DAG protobuf)
    /// - `0x71` - DAG-CBOR
    /// - `0x0129` - DAG-JSON
    pub fn codec(mut self, codec: u64) -> Self {
        self.codec = codec;
        self
    }

    /// Sets the hash algorithm to use for CID generation.
    ///
    /// Supported algorithms: SHA2-256, SHA3-256, BLAKE3.
    pub fn hash_algorithm(mut self, algorithm: HashAlgorithm) -> Self {
        self.hash_algorithm = algorithm;
        self
    }

    /// Build a CID from data using the configured hash algorithm
    pub fn build(&self, data: &[u8]) -> Result<Cid> {
        let hash = self.hash_algorithm.code().digest(data);

        // CIDv0 requires SHA2-256 and DAG-PB codec
        if self.version == cid::Version::V0 {
            if self.hash_algorithm != HashAlgorithm::Sha256 {
                return Err(Error::InvalidInput(
                    "CIDv0 requires SHA2-256 hash algorithm".to_string(),
                ));
            }
            // CIDv0 always uses DAG-PB codec implicitly
            return Cid::new_v0(hash)
                .map_err(|e| Error::Cid(format!("Failed to create CIDv0: {}", e)));
        }

        Cid::new(self.version, self.codec, hash)
            .map_err(|e| Error::Cid(format!("Failed to create CID: {}", e)))
    }

    /// Build a CID using DAG-CBOR codec
    pub fn build_dag_cbor(&self, data: &[u8]) -> Result<Cid> {
        let mut builder = self.clone();
        builder.codec = 0x71; // DAG-CBOR codec
        builder.build(data)
    }

    /// Build a CID using raw codec (default)
    pub fn build_raw(&self, data: &[u8]) -> Result<Cid> {
        let mut builder = self.clone();
        builder.codec = 0x55; // raw codec
        builder.build(data)
    }

    /// Build a CIDv0 (legacy format) from data
    ///
    /// This is a convenience method that creates a CIDv0 regardless
    /// of the builder's current configuration.
    pub fn build_v0(&self, data: &[u8]) -> Result<Cid> {
        let hash = Code::Sha2_256.digest(data);
        Cid::new_v0(hash).map_err(|e| Error::Cid(format!("Failed to create CIDv0: {}", e)))
    }
}

/// Extension trait for CID with additional encoding utilities
pub trait CidExt {
    /// Encode the CID to a string with the specified multibase encoding
    fn to_string_with_base(&self, base: MultibaseEncoding) -> String;

    /// Convert CIDv0 to CIDv1
    ///
    /// CIDv0 is converted to CIDv1 with DAG-PB codec (0x70)
    fn to_v1(&self) -> Result<Cid>;

    /// Try to convert CIDv1 to CIDv0
    ///
    /// This only succeeds if:
    /// - The hash algorithm is SHA2-256
    /// - The codec is DAG-PB (0x70)
    fn to_v0(&self) -> Result<Cid>;

    /// Check if this CID can be represented as CIDv0
    ///
    /// Returns true if the CID uses SHA2-256 and DAG-PB codec
    fn can_be_v0(&self) -> bool;

    /// Check if this is a CIDv0
    fn is_v0(&self) -> bool;

    /// Check if this is a CIDv1
    fn is_v1(&self) -> bool;

    /// Get the codec code
    fn codec_code(&self) -> u64;

    /// Get the hash algorithm name
    fn hash_algorithm_name(&self) -> &'static str;

    /// Get the hash algorithm code
    fn hash_algorithm_code(&self) -> u64;
}

impl CidExt for Cid {
    fn to_string_with_base(&self, base: MultibaseEncoding) -> String {
        let bytes = self.to_bytes();
        multibase::encode(base.base(), bytes)
    }

    fn to_v1(&self) -> Result<Cid> {
        if self.version() == cid::Version::V1 {
            return Ok(*self);
        }

        // CIDv0 is always SHA2-256 with DAG-PB codec
        Ok(Cid::new_v1(codec::DAG_PB, *self.hash()))
    }

    fn to_v0(&self) -> Result<Cid> {
        if self.version() == cid::Version::V0 {
            return Ok(*self);
        }

        // CIDv0 requires SHA2-256 hash algorithm
        if self.hash().code() != 0x12 {
            return Err(Error::InvalidInput(
                "CIDv0 requires SHA2-256 hash algorithm".to_string(),
            ));
        }

        // CIDv0 requires DAG-PB codec
        if self.codec() != codec::DAG_PB {
            return Err(Error::InvalidInput(
                "CIDv0 requires DAG-PB codec (0x70)".to_string(),
            ));
        }

        Cid::new_v0(*self.hash()).map_err(|e| Error::Cid(format!("Failed to create CIDv0: {}", e)))
    }

    fn can_be_v0(&self) -> bool {
        // CIDv0 requires SHA2-256 (code 0x12) and DAG-PB codec (0x70)
        self.hash().code() == 0x12 && self.codec() == codec::DAG_PB
    }

    fn is_v0(&self) -> bool {
        self.version() == cid::Version::V0
    }

    fn is_v1(&self) -> bool {
        self.version() == cid::Version::V1
    }

    fn codec_code(&self) -> u64 {
        self.codec()
    }

    fn hash_algorithm_name(&self) -> &'static str {
        match self.hash().code() {
            0x12 => "sha2-256",
            0x14 => "sha3-256", // SHA3-256 (Keccak)
            0x16 => "sha3-256", // SHA3-256 alternative code
            0x1b => "keccak-256",
            0x1e => "blake2b-256",
            _ => "unknown",
        }
    }

    fn hash_algorithm_code(&self) -> u64 {
        self.hash().code()
    }
}

/// Parse a CID from a multibase-encoded string with automatic base detection
pub fn parse_cid(s: &str) -> Result<Cid> {
    s.parse()
        .map_err(|e| Error::Cid(format!("Failed to parse CID: {}", e)))
}

/// Parse a CID from a multibase-encoded string and return the detected base
pub fn parse_cid_with_base(s: &str) -> Result<(Cid, MultibaseEncoding)> {
    let first_char = s
        .chars()
        .next()
        .ok_or_else(|| Error::Cid("Empty CID string".to_string()))?;

    // Check if it looks like CIDv0 (starts with 'Qm')
    if s.starts_with("Qm") {
        let cid: Cid = s
            .parse()
            .map_err(|e| Error::Cid(format!("Failed to parse CIDv0: {}", e)))?;
        return Ok((cid, MultibaseEncoding::Base58Btc));
    }

    let base = MultibaseEncoding::from_prefix(first_char)
        .ok_or_else(|| Error::Cid(format!("Unknown multibase prefix: {}", first_char)))?;

    let cid: Cid = s
        .parse()
        .map_err(|e| Error::Cid(format!("Failed to parse CID: {}", e)))?;

    Ok((cid, base))
}

/// Serializable CID wrapper
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SerializableCid(#[serde(with = "cid_serde")] pub Cid);

impl SerializableCid {
    /// Create a new SerializableCid from a Cid
    pub fn new(cid: Cid) -> Self {
        Self(cid)
    }

    /// Get the inner CID
    pub fn inner(&self) -> &Cid {
        &self.0
    }

    /// Convert to string with specified encoding
    pub fn to_string_with_base(&self, base: MultibaseEncoding) -> String {
        self.0.to_string_with_base(base)
    }
}

impl Copy for SerializableCid {}

impl From<Cid> for SerializableCid {
    fn from(cid: Cid) -> Self {
        Self(cid)
    }
}

impl From<SerializableCid> for Cid {
    fn from(cid: SerializableCid) -> Self {
        cid.0
    }
}

impl std::fmt::Display for SerializableCid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

mod cid_serde {
    use super::*;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S>(cid: &Cid, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&cid.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> std::result::Result<Cid, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// Common IPLD codec constants
pub mod codec {
    /// Raw binary data
    pub const RAW: u64 = 0x55;
    /// DAG-CBOR (IPLD CBOR)
    pub const DAG_CBOR: u64 = 0x71;
    /// DAG-JSON (IPLD JSON)
    pub const DAG_JSON: u64 = 0x0129;
    /// DAG-PB (Protocol Buffers, used by IPFS UnixFS)
    pub const DAG_PB: u64 = 0x70;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multibase_encoding() {
        let cid = CidBuilder::new().build(b"hello world").unwrap();

        // Test different encodings
        let base32 = cid.to_string_with_base(MultibaseEncoding::Base32Lower);
        let base58 = cid.to_string_with_base(MultibaseEncoding::Base58Btc);
        let base64 = cid.to_string_with_base(MultibaseEncoding::Base64);

        assert!(base32.starts_with('b'));
        assert!(base58.starts_with('z'));
        assert!(base64.starts_with('m'));

        // All should parse back to the same CID
        let parsed32: Cid = base32.parse().unwrap();
        let parsed58: Cid = base58.parse().unwrap();
        let parsed64: Cid = base64.parse().unwrap();

        assert_eq!(cid, parsed32);
        assert_eq!(cid, parsed58);
        assert_eq!(cid, parsed64);
    }

    #[test]
    fn test_parse_cid_with_base() {
        let cid = CidBuilder::new().build(b"test data").unwrap();

        // Encode with base32
        let base32_str = cid.to_string_with_base(MultibaseEncoding::Base32Lower);
        let (parsed_cid, detected_base) = parse_cid_with_base(&base32_str).unwrap();

        assert_eq!(cid, parsed_cid);
        assert_eq!(detected_base, MultibaseEncoding::Base32Lower);
    }

    #[test]
    fn test_cid_ext_methods() {
        let cid = CidBuilder::new().build(b"test").unwrap();

        assert!(cid.is_v1());
        assert!(!cid.is_v0());
        assert_eq!(cid.codec_code(), codec::RAW);
        assert_eq!(cid.hash_algorithm_name(), "sha2-256");
    }

    #[test]
    fn test_serializable_cid() {
        let cid = CidBuilder::new().build(b"test").unwrap();
        let serializable = SerializableCid::new(cid);

        // Test JSON serialization
        let json = serde_json::to_string(&serializable).unwrap();
        let deserialized: SerializableCid = serde_json::from_str(&json).unwrap();

        assert_eq!(serializable, deserialized);
    }

    #[test]
    fn test_cidv0_creation() {
        // Create CIDv0 using the builder
        let cid_v0 = CidBuilder::v0().build(b"hello world").unwrap();

        // CIDv0 should start with "Qm"
        let cid_str = cid_v0.to_string();
        assert!(cid_str.starts_with("Qm"), "CIDv0 should start with 'Qm'");

        // Verify it's actually v0
        assert!(cid_v0.is_v0());
        assert!(!cid_v0.is_v1());
    }

    #[test]
    fn test_cidv0_build_v0_method() {
        // Build CIDv0 using convenience method
        let cid_v0 = CidBuilder::new().build_v0(b"test data").unwrap();

        // Verify it's v0
        assert!(cid_v0.is_v0());
        assert!(cid_v0.to_string().starts_with("Qm"));
    }

    #[test]
    fn test_cidv0_to_v1_conversion() {
        let cid_v0 = CidBuilder::v0().build(b"test").unwrap();
        let cid_v1 = cid_v0.to_v1().unwrap();

        // v1 should be different but represent the same content
        assert!(cid_v1.is_v1());
        assert!(!cid_v1.is_v0());

        // The hash should be the same
        assert_eq!(cid_v0.hash(), cid_v1.hash());

        // v1 should have DAG-PB codec (from v0)
        assert_eq!(cid_v1.codec_code(), codec::DAG_PB);
    }

    #[test]
    fn test_cidv1_to_v0_conversion() {
        // Create a CIDv1 with DAG-PB codec and SHA2-256 (compatible with v0)
        let cid_v1 = CidBuilder::new()
            .codec(codec::DAG_PB)
            .build(b"test")
            .unwrap();

        // Convert to v0
        let cid_v0 = cid_v1.to_v0().unwrap();

        assert!(cid_v0.is_v0());
        assert!(cid_v0.to_string().starts_with("Qm"));
    }

    #[test]
    fn test_cidv1_to_v0_fails_wrong_codec() {
        // Create a CIDv1 with RAW codec (not compatible with v0)
        let cid_v1 = CidBuilder::new().build(b"test").unwrap();

        // Should fail because RAW codec is not compatible with v0
        let result = cid_v1.to_v0();
        assert!(result.is_err());
    }

    #[test]
    fn test_can_be_v0() {
        // CIDv1 with DAG-PB and SHA2-256 can be v0
        let cid_compatible = CidBuilder::new()
            .codec(codec::DAG_PB)
            .build(b"test")
            .unwrap();
        assert!(cid_compatible.can_be_v0());

        // CIDv1 with RAW codec cannot be v0
        let cid_incompatible = CidBuilder::new().build(b"test").unwrap();
        assert!(!cid_incompatible.can_be_v0());

        // CIDv1 with SHA3-256 cannot be v0
        let cid_sha3 = CidBuilder::new()
            .codec(codec::DAG_PB)
            .hash_algorithm(HashAlgorithm::Sha3_256)
            .build(b"test")
            .unwrap();
        assert!(!cid_sha3.can_be_v0());
    }

    #[test]
    fn test_parse_cidv0_string() {
        // Create and stringify a CIDv0
        let original = CidBuilder::v0().build(b"hello ipfs").unwrap();
        let cid_str = original.to_string();

        // Parse it back
        let parsed = parse_cid(&cid_str).unwrap();
        assert_eq!(original, parsed);

        // Test with parse_cid_with_base
        let (parsed2, base) = parse_cid_with_base(&cid_str).unwrap();
        assert_eq!(original, parsed2);
        assert_eq!(base, MultibaseEncoding::Base58Btc);
    }

    #[test]
    fn test_cidv0_roundtrip() {
        let data = b"test content for roundtrip";

        // Create v0
        let cid_v0 = CidBuilder::v0().build(data).unwrap();

        // Convert to v1
        let cid_v1 = cid_v0.to_v1().unwrap();

        // Convert back to v0
        let cid_v0_again = cid_v1.to_v0().unwrap();

        // Should be equal to original
        assert_eq!(cid_v0, cid_v0_again);
    }
}
