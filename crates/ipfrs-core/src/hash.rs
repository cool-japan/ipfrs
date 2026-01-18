//! Hardware-accelerated hashing with SIMD support
//!
//! This module provides a pluggable hash algorithm system with:
//! - Runtime CPU feature detection
//! - SIMD-optimized implementations (AVX2 for x86_64, NEON for ARM)
//! - Fallback to standard implementations
//! - Modern hash algorithms (BLAKE3)
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_core::hash::{HashEngine, Sha256Engine, Sha512Engine, Blake3Engine};
//!
//! let data = b"hello world";
//!
//! // SHA2-256 (32 bytes)
//! let sha256 = Sha256Engine::new();
//! let hash256 = sha256.digest(data);
//!
//! // SHA2-512 (64 bytes) - stronger security
//! let sha512 = Sha512Engine::new();
//! let hash512 = sha512.digest(data);
//!
//! // BLAKE3 is fastest with built-in SIMD
//! let blake3 = Blake3Engine::new();
//! let blake3_hash = blake3.digest(data);
//! ```

use crate::error::Result;
use blake2::{Blake2b512 as Blake2b512Impl, Blake2s256 as Blake2s256Impl, Digest as Blake2Digest};
use multihash_codetable::Code;
#[allow(unused_imports)]
use sha2::{Digest, Sha256 as Sha256Impl, Sha512 as Sha512Impl};
use sha3::{Sha3_256 as Sha3_256Impl, Sha3_512 as Sha3_512Impl};
use std::sync::Arc;

/// Trait for hardware-accelerated hash computation
pub trait HashEngine: Send + Sync {
    /// Compute hash of data
    fn digest(&self, data: &[u8]) -> Vec<u8>;

    /// Get the multihash code for this algorithm
    fn code(&self) -> Code;

    /// Get the name of this hash algorithm
    fn name(&self) -> &'static str;

    /// Check if SIMD acceleration is enabled
    #[inline]
    fn is_simd_enabled(&self) -> bool {
        false
    }
}

/// CPU feature detection result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuFeatures {
    /// AVX2 support (x86_64)
    pub avx2: bool,
    /// NEON support (ARM)
    pub neon: bool,
}

impl CpuFeatures {
    /// Detect CPU features at runtime
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx2: is_x86_feature_detected!("avx2"),
                neon: false,
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self {
                avx2: false,
                // NEON is always available on aarch64
                neon: true,
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                avx2: false,
                neon: false,
            }
        }
    }
}

/// SHA2-256 hash engine with SIMD support
pub struct Sha256Engine {
    features: CpuFeatures,
}

impl Sha256Engine {
    /// Create a new SHA2-256 hash engine
    pub fn new() -> Self {
        Self {
            features: CpuFeatures::detect(),
        }
    }

    /// Compute hash using AVX2 instructions (x86_64)
    ///
    /// The sha2 crate automatically uses SIMD instructions when available.
    /// This includes SHA-NI (SHA extensions), AVX2, and SSE4.1 on x86_64.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn digest_avx2(&self, data: &[u8]) -> Vec<u8> {
        // The sha2 crate automatically uses hardware SHA extensions (SHA-NI)
        // when available, falling back to AVX2/SSE optimizations
        let mut hasher = Sha256Impl::new();
        hasher.update(data);
        hasher.finalize().to_vec()
    }

    /// Compute hash using NEON instructions (ARM)
    ///
    /// The sha2 crate uses NEON intrinsics on ARM architectures for better performance.
    #[cfg(target_arch = "aarch64")]
    fn digest_neon(&self, data: &[u8]) -> Vec<u8> {
        // The sha2 crate automatically uses NEON intrinsics on ARM
        let mut hasher = Sha256Impl::new();
        hasher.update(data);
        hasher.finalize().to_vec()
    }

    /// Fallback scalar implementation
    ///
    /// Even the "scalar" implementation in sha2 is highly optimized.
    fn digest_scalar(&self, data: &[u8]) -> Vec<u8> {
        let mut hasher = Sha256Impl::new();
        hasher.update(data);
        hasher.finalize().to_vec()
    }
}

impl Default for Sha256Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl HashEngine for Sha256Engine {
    fn digest(&self, data: &[u8]) -> Vec<u8> {
        #[cfg(target_arch = "x86_64")]
        {
            if self.features.avx2 {
                // Safety: We checked that AVX2 is available
                unsafe { self.digest_avx2(data) }
            } else {
                self.digest_scalar(data)
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if self.features.neon {
                self.digest_neon(data)
            } else {
                self.digest_scalar(data)
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            self.digest_scalar(data)
        }
    }

    #[inline]
    fn code(&self) -> Code {
        Code::Sha2_256
    }

    #[inline]
    fn name(&self) -> &'static str {
        "sha2-256"
    }

    #[inline]
    fn is_simd_enabled(&self) -> bool {
        self.features.avx2 || self.features.neon
    }
}

/// SHA-512 hash engine with SIMD support
///
/// Similar to SHA2-256 but produces 512-bit (64-byte) hashes.
/// Provides stronger security margin for applications requiring it.
pub struct Sha512Engine {
    features: CpuFeatures,
}

impl Sha512Engine {
    /// Create a new SHA-512 hash engine
    pub fn new() -> Self {
        Self {
            features: CpuFeatures::detect(),
        }
    }

    /// Compute SHA-512 hash using scalar implementation
    fn digest_scalar(&self, data: &[u8]) -> Vec<u8> {
        let mut hasher = Sha512Impl::new();
        hasher.update(data);
        hasher.finalize().to_vec()
    }
}

impl Default for Sha512Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl HashEngine for Sha512Engine {
    fn digest(&self, data: &[u8]) -> Vec<u8> {
        // SHA-512 uses same SIMD optimizations as SHA-256
        self.digest_scalar(data)
    }

    #[inline]
    fn code(&self) -> Code {
        Code::Sha2_512
    }

    #[inline]
    fn name(&self) -> &'static str {
        "sha2-512"
    }

    #[inline]
    fn is_simd_enabled(&self) -> bool {
        self.features.avx2 || self.features.neon
    }
}

/// SHA3-256 hash engine
///
/// SHA3 uses the Keccak sponge construction and is optimized
/// differently than SHA2. The sha3 crate provides optimized implementations.
pub struct Sha3_256Engine;

impl Sha3_256Engine {
    /// Create a new SHA3-256 hash engine
    pub fn new() -> Self {
        Self
    }

    /// Compute SHA3-256 hash
    ///
    /// The sha3 crate provides optimized Keccak implementation.
    fn digest_impl(&self, data: &[u8]) -> Vec<u8> {
        let mut hasher = Sha3_256Impl::new();
        hasher.update(data);
        hasher.finalize().to_vec()
    }
}

impl Default for Sha3_256Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl HashEngine for Sha3_256Engine {
    fn digest(&self, data: &[u8]) -> Vec<u8> {
        self.digest_impl(data)
    }

    #[inline]
    fn code(&self) -> Code {
        Code::Sha3_256
    }

    #[inline]
    fn name(&self) -> &'static str {
        "sha3-256"
    }

    #[inline]
    fn is_simd_enabled(&self) -> bool {
        // SHA3 implementations are optimized but don't typically
        // use explicit SIMD like SHA2 does
        false
    }
}

/// SHA3-512 hash engine
///
/// SHA3-512 provides 512-bit (64-byte) output using the Keccak sponge construction.
/// Offers higher security margin than SHA3-256 for applications requiring it.
pub struct Sha3_512Engine;

impl Sha3_512Engine {
    /// Create a new SHA3-512 hash engine
    pub fn new() -> Self {
        Self
    }

    /// Compute SHA3-512 hash
    fn digest_impl(&self, data: &[u8]) -> Vec<u8> {
        let mut hasher = Sha3_512Impl::new();
        hasher.update(data);
        hasher.finalize().to_vec()
    }
}

impl Default for Sha3_512Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl HashEngine for Sha3_512Engine {
    fn digest(&self, data: &[u8]) -> Vec<u8> {
        self.digest_impl(data)
    }

    #[inline]
    fn code(&self) -> Code {
        Code::Sha3_512
    }

    #[inline]
    fn name(&self) -> &'static str {
        "sha3-512"
    }

    #[inline]
    fn is_simd_enabled(&self) -> bool {
        // SHA3 implementations are optimized but don't typically
        // use explicit SIMD like SHA2 does
        false
    }
}

/// BLAKE3 hash engine with built-in SIMD support
///
/// BLAKE3 is a modern, extremely fast cryptographic hash function that:
/// - Has built-in SIMD optimizations (AVX2, AVX-512, NEON)
/// - Is faster than SHA2-256 and SHA3-256
/// - Provides 256-bit output (32 bytes)
/// - Is designed for modern CPUs
///
/// BLAKE3 automatically uses the best available SIMD instructions
/// for the current CPU architecture.
pub struct Blake3Engine;

impl Blake3Engine {
    /// Create a new BLAKE3 hash engine
    pub fn new() -> Self {
        Self
    }
}

impl Default for Blake3Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl HashEngine for Blake3Engine {
    fn digest(&self, data: &[u8]) -> Vec<u8> {
        // BLAKE3 automatically uses SIMD when available
        let mut hasher = blake3::Hasher::new();
        hasher.update(data);
        hasher.finalize().as_bytes().to_vec()
    }

    #[inline]
    fn code(&self) -> Code {
        // BLAKE3-256 is natively supported in multihash-codetable
        Code::Blake3_256
    }

    #[inline]
    fn name(&self) -> &'static str {
        "blake3"
    }

    #[inline]
    fn is_simd_enabled(&self) -> bool {
        // BLAKE3 always uses SIMD when available
        true
    }
}

/// BLAKE2b-256 hash engine
///
/// BLAKE2b is a cryptographic hash function optimized for 64-bit platforms.
/// This variant produces 256-bit (32-byte) hashes.
///
/// BLAKE2b features:
/// - Faster than MD5, SHA-1, SHA-2, and SHA-3
/// - At least as secure as SHA-3
/// - No known attacks better than brute force
/// - Simple design with no padding required
#[derive(Debug, Clone, Copy, Default)]
pub struct Blake2b256Engine;

impl Blake2b256Engine {
    /// Create a new BLAKE2b-256 hash engine
    pub fn new() -> Self {
        Self
    }
}

impl HashEngine for Blake2b256Engine {
    fn digest(&self, data: &[u8]) -> Vec<u8> {
        use blake2::digest::FixedOutput;

        // BLAKE2b-256: Use Blake2b512 and truncate to 32 bytes
        let mut hasher = Blake2b512Impl::new();
        Blake2Digest::update(&mut hasher, data);
        let result = hasher.finalize_fixed();
        result[..32].to_vec()
    }

    #[inline]
    fn code(&self) -> Code {
        Code::Blake2b256
    }

    #[inline]
    fn name(&self) -> &'static str {
        "blake2b-256"
    }

    #[inline]
    fn is_simd_enabled(&self) -> bool {
        // BLAKE2 implementations typically use SIMD when available
        true
    }
}

/// BLAKE2b-512 hash engine
///
/// BLAKE2b is a cryptographic hash function optimized for 64-bit platforms.
/// This variant produces 512-bit (64-byte) hashes.
///
/// This is the full-length BLAKE2b output, providing maximum security.
#[derive(Debug, Clone, Copy, Default)]
pub struct Blake2b512Engine;

impl Blake2b512Engine {
    /// Create a new BLAKE2b-512 hash engine
    pub fn new() -> Self {
        Self
    }
}

impl HashEngine for Blake2b512Engine {
    fn digest(&self, data: &[u8]) -> Vec<u8> {
        let mut hasher = Blake2b512Impl::new();
        Blake2Digest::update(&mut hasher, data);
        hasher.finalize().to_vec()
    }

    #[inline]
    fn code(&self) -> Code {
        Code::Blake2b512
    }

    #[inline]
    fn name(&self) -> &'static str {
        "blake2b-512"
    }

    #[inline]
    fn is_simd_enabled(&self) -> bool {
        // BLAKE2 implementations typically use SIMD when available
        true
    }
}

/// BLAKE2s-256 hash engine
///
/// BLAKE2s is a cryptographic hash function optimized for 8-32 bit platforms.
/// This variant produces 256-bit (32-byte) hashes.
///
/// BLAKE2s is optimized for smaller architectures and embedded systems,
/// while still providing strong cryptographic security.
#[derive(Debug, Clone, Copy, Default)]
pub struct Blake2s256Engine;

impl Blake2s256Engine {
    /// Create a new BLAKE2s-256 hash engine
    pub fn new() -> Self {
        Self
    }
}

impl HashEngine for Blake2s256Engine {
    fn digest(&self, data: &[u8]) -> Vec<u8> {
        let mut hasher = Blake2s256Impl::new();
        Blake2Digest::update(&mut hasher, data);
        hasher.finalize().to_vec()
    }

    #[inline]
    fn code(&self) -> Code {
        Code::Blake2s256
    }

    #[inline]
    fn name(&self) -> &'static str {
        "blake2s-256"
    }

    #[inline]
    fn is_simd_enabled(&self) -> bool {
        // BLAKE2 implementations typically use SIMD when available
        true
    }
}

/// Hash algorithm registry for pluggable hash support
pub struct HashRegistry {
    algorithms: std::collections::HashMap<u64, Arc<dyn HashEngine>>,
}

impl HashRegistry {
    /// Create a new hash registry with default algorithms
    pub fn new() -> Self {
        let mut registry = Self {
            algorithms: std::collections::HashMap::new(),
        };

        // Register default algorithms
        registry.register(Arc::new(Sha256Engine::new()));
        registry.register(Arc::new(Sha512Engine::new()));
        registry.register(Arc::new(Sha3_256Engine::new()));
        registry.register(Arc::new(Sha3_512Engine::new()));
        registry.register(Arc::new(Blake2b256Engine::new()));
        registry.register(Arc::new(Blake2b512Engine::new()));
        registry.register(Arc::new(Blake2s256Engine::new()));
        registry.register(Arc::new(Blake3Engine::new()));

        registry
    }

    /// Register a hash algorithm
    pub fn register(&mut self, engine: Arc<dyn HashEngine>) {
        let code_u64 = engine.code() as u64;
        self.algorithms.insert(code_u64, engine);
    }

    /// Get a hash engine by code
    pub fn get(&self, code: Code) -> Option<Arc<dyn HashEngine>> {
        let code_u64 = code as u64;
        self.algorithms.get(&code_u64).cloned()
    }

    /// Compute hash using the specified algorithm
    pub fn digest(&self, code: Code, data: &[u8]) -> Result<Vec<u8>> {
        let engine = self.get(code).ok_or_else(|| {
            crate::error::Error::InvalidInput(format!("Unsupported hash algorithm: {:?}", code))
        })?;
        Ok(engine.digest(data))
    }
}

impl Default for HashRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global hash registry instance
static HASH_REGISTRY: once_cell::sync::Lazy<HashRegistry> =
    once_cell::sync::Lazy::new(HashRegistry::new);

/// Get the global hash registry
pub fn global_hash_registry() -> &'static HashRegistry {
    &HASH_REGISTRY
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_feature_detection() {
        let features = CpuFeatures::detect();

        #[cfg(target_arch = "x86_64")]
        {
            // AVX2 may or may not be available
            assert!(!features.neon);
        }

        #[cfg(target_arch = "aarch64")]
        {
            assert!(features.neon);
            assert!(!features.avx2);
        }
    }

    #[test]
    fn test_sha256_engine() {
        let engine = Sha256Engine::new();
        let hash = engine.digest(b"hello world");

        // SHA256 produces 32-byte hashes
        assert_eq!(hash.len(), 32);

        // Same input should produce same hash
        let hash2 = engine.digest(b"hello world");
        assert_eq!(hash, hash2);

        // Different input should produce different hash
        let hash3 = engine.digest(b"hello mars");
        assert_ne!(hash, hash3);
    }

    #[test]
    fn test_sha3_256_engine() {
        let engine = Sha3_256Engine::new();
        let hash = engine.digest(b"test data");

        // SHA3-256 produces 32-byte hashes
        assert_eq!(hash.len(), 32);

        // Deterministic
        let hash2 = engine.digest(b"test data");
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_hash_registry() {
        let registry = HashRegistry::new();

        // Should have SHA256
        let sha256 = registry.get(Code::Sha2_256);
        assert!(sha256.is_some());

        // Should have SHA512
        let sha512 = registry.get(Code::Sha2_512);
        assert!(sha512.is_some());

        // Should have SHA3-256
        let sha3_256 = registry.get(Code::Sha3_256);
        assert!(sha3_256.is_some());

        // Should have SHA3-512
        let sha3_512 = registry.get(Code::Sha3_512);
        assert!(sha3_512.is_some());
    }

    #[test]
    fn test_registry_digest() {
        let registry = HashRegistry::new();

        let hash = registry.digest(Code::Sha2_256, b"test").unwrap();
        assert_eq!(hash.len(), 32);
    }

    #[test]
    fn test_global_registry() {
        let registry = global_hash_registry();

        let hash = registry.digest(Code::Sha2_256, b"global test").unwrap();
        assert_eq!(hash.len(), 32);
    }

    #[test]
    fn test_sha256_deterministic() {
        let engine = Sha256Engine::new();

        // Test with various data sizes
        for size in [0, 1, 64, 256, 1024, 4096] {
            let data = vec![42u8; size];
            let hash1 = engine.digest(&data);
            let hash2 = engine.digest(&data);
            assert_eq!(hash1, hash2, "Failed for size {}", size);
        }
    }

    #[test]
    fn test_sha512_engine() {
        let engine = Sha512Engine::new();
        let hash = engine.digest(b"hello world");

        // SHA-512 produces 64-byte hashes
        assert_eq!(hash.len(), 64);

        // Same input should produce same hash
        let hash2 = engine.digest(b"hello world");
        assert_eq!(hash, hash2);

        // Different input should produce different hash
        let hash3 = engine.digest(b"hello mars");
        assert_ne!(hash, hash3);
    }

    #[test]
    fn test_sha3_512_engine() {
        let engine = Sha3_512Engine::new();
        let hash = engine.digest(b"hello world");

        // SHA3-512 produces 64-byte hashes
        assert_eq!(hash.len(), 64);

        // Same input should produce same hash
        let hash2 = engine.digest(b"hello world");
        assert_eq!(hash, hash2);

        // Different input should produce different hash
        let hash3 = engine.digest(b"hello mars");
        assert_ne!(hash, hash3);
    }

    #[test]
    fn test_sha512_deterministic() {
        let engine = Sha512Engine::new();

        // Test with various data sizes
        for size in [0, 1, 64, 256, 1024, 4096] {
            let data = vec![42u8; size];
            let hash1 = engine.digest(&data);
            let hash2 = engine.digest(&data);
            assert_eq!(hash1, hash2, "Failed for size {}", size);
        }
    }

    #[test]
    fn test_sha3_512_deterministic() {
        let engine = Sha3_512Engine::new();

        // Test with various data sizes
        for size in [0, 1, 64, 256, 1024, 4096] {
            let data = vec![42u8; size];
            let hash1 = engine.digest(&data);
            let hash2 = engine.digest(&data);
            assert_eq!(hash1, hash2, "Failed for size {}", size);
        }
    }

    #[test]
    fn test_sha512_vs_sha256() {
        let sha512 = Sha512Engine::new();
        let sha256 = Sha256Engine::new();

        let data = b"test data";
        let hash512 = sha512.digest(data);
        let hash256 = sha256.digest(data);

        // SHA-512 should produce 64 bytes, SHA-256 should produce 32 bytes
        assert_eq!(hash512.len(), 64);
        assert_eq!(hash256.len(), 32);

        // The hashes should be different
        assert_ne!(&hash512[..32], &hash256[..]);
    }

    #[test]
    fn test_sha3_512_vs_sha3_256() {
        let sha3_512 = Sha3_512Engine::new();
        let sha3_256 = Sha3_256Engine::new();

        let data = b"test data";
        let hash512 = sha3_512.digest(data);
        let hash256 = sha3_256.digest(data);

        // SHA3-512 should produce 64 bytes, SHA3-256 should produce 32 bytes
        assert_eq!(hash512.len(), 64);
        assert_eq!(hash256.len(), 32);

        // The hashes should be different
        assert_ne!(&hash512[..32], &hash256[..]);
    }

    #[test]
    fn test_blake3_engine() {
        let engine = Blake3Engine::new();
        let hash = engine.digest(b"hello world");

        // BLAKE3 produces 32-byte hashes
        assert_eq!(hash.len(), 32);

        // Same input should produce same hash
        let hash2 = engine.digest(b"hello world");
        assert_eq!(hash, hash2);

        // Different input should produce different hash
        let hash3 = engine.digest(b"hello mars");
        assert_ne!(hash, hash3);
    }

    #[test]
    fn test_blake3_deterministic() {
        let engine = Blake3Engine::new();

        // Test with various data sizes
        for size in [0, 1, 64, 256, 1024, 4096] {
            let data = vec![42u8; size];
            let hash1 = engine.digest(&data);
            let hash2 = engine.digest(&data);
            assert_eq!(hash1, hash2, "Failed for size {}", size);
        }
    }

    #[test]
    fn test_blake3_simd_enabled() {
        let engine = Blake3Engine::new();
        // BLAKE3 should always report SIMD as enabled
        assert!(engine.is_simd_enabled());
    }

    #[test]
    fn test_blake3_vs_sha256() {
        let blake3 = Blake3Engine::new();
        let sha256 = Sha256Engine::new();

        let data = b"test data for comparison";

        let blake3_hash = blake3.digest(data);
        let sha256_hash = sha256.digest(data);

        // Both should produce 32-byte hashes
        assert_eq!(blake3_hash.len(), 32);
        assert_eq!(sha256_hash.len(), 32);

        // But the hashes should be different (different algorithms)
        assert_ne!(blake3_hash, sha256_hash);
    }

    #[test]
    fn test_blake3_empty_input() {
        let engine = Blake3Engine::new();
        let hash = engine.digest(b"");

        // BLAKE3 hash of empty string
        assert_eq!(hash.len(), 32);

        // Empty input should be deterministic
        let hash2 = engine.digest(b"");
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_blake2b256_engine() {
        let engine = Blake2b256Engine::new();
        let hash = engine.digest(b"hello world");

        // BLAKE2b-256 produces 32-byte hashes
        assert_eq!(hash.len(), 32);

        // Same input should produce same hash
        let hash2 = engine.digest(b"hello world");
        assert_eq!(hash, hash2);

        // Different input should produce different hash
        let hash3 = engine.digest(b"hello mars");
        assert_ne!(hash, hash3);
    }

    #[test]
    fn test_blake2b512_engine() {
        let engine = Blake2b512Engine::new();
        let hash = engine.digest(b"hello world");

        // BLAKE2b-512 produces 64-byte hashes
        assert_eq!(hash.len(), 64);

        // Same input should produce same hash
        let hash2 = engine.digest(b"hello world");
        assert_eq!(hash, hash2);

        // Different input should produce different hash
        let hash3 = engine.digest(b"hello mars");
        assert_ne!(hash, hash3);
    }

    #[test]
    fn test_blake2s256_engine() {
        let engine = Blake2s256Engine::new();
        let hash = engine.digest(b"test data");

        // BLAKE2s-256 produces 32-byte hashes
        assert_eq!(hash.len(), 32);

        // Deterministic
        let hash2 = engine.digest(b"test data");
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_blake2b256_deterministic() {
        let engine = Blake2b256Engine::new();

        // Test with various data sizes
        for size in [0, 1, 64, 256, 1024, 4096] {
            let data = vec![42u8; size];
            let hash1 = engine.digest(&data);
            let hash2 = engine.digest(&data);
            assert_eq!(hash1, hash2, "Failed for size {}", size);
        }
    }

    #[test]
    fn test_blake2b512_deterministic() {
        let engine = Blake2b512Engine::new();

        // Test with various data sizes
        for size in [0, 1, 64, 256, 1024, 4096] {
            let data = vec![42u8; size];
            let hash1 = engine.digest(&data);
            let hash2 = engine.digest(&data);
            assert_eq!(hash1, hash2, "Failed for size {}", size);
            assert_eq!(hash1.len(), 64);
        }
    }

    #[test]
    fn test_blake2s256_deterministic() {
        let engine = Blake2s256Engine::new();

        // Test with various data sizes
        for size in [0, 1, 64, 256, 1024] {
            let data = vec![42u8; size];
            let hash1 = engine.digest(&data);
            let hash2 = engine.digest(&data);
            assert_eq!(hash1, hash2, "Failed for size {}", size);
        }
    }

    #[test]
    fn test_blake2b_vs_blake2s() {
        let blake2b = Blake2b256Engine::new();
        let blake2s = Blake2s256Engine::new();

        let data = b"test data for comparison";

        let blake2b_hash = blake2b.digest(data);
        let blake2s_hash = blake2s.digest(data);

        // Both should produce 32-byte hashes
        assert_eq!(blake2b_hash.len(), 32);
        assert_eq!(blake2s_hash.len(), 32);

        // But the hashes should be different (different algorithms)
        assert_ne!(blake2b_hash, blake2s_hash);
    }

    #[test]
    fn test_blake2_empty_input() {
        let blake2b256 = Blake2b256Engine::new();
        let blake2b512 = Blake2b512Engine::new();
        let blake2s = Blake2s256Engine::new();

        let hash256 = blake2b256.digest(b"");
        let hash512 = blake2b512.digest(b"");
        let hashs = blake2s.digest(b"");

        assert_eq!(hash256.len(), 32);
        assert_eq!(hash512.len(), 64);
        assert_eq!(hashs.len(), 32);

        // Empty input should be deterministic
        assert_eq!(hash256, blake2b256.digest(b""));
        assert_eq!(hash512, blake2b512.digest(b""));
        assert_eq!(hashs, blake2s.digest(b""));
    }

    #[test]
    fn test_blake2_simd_enabled() {
        let blake2b256 = Blake2b256Engine::new();
        let blake2b512 = Blake2b512Engine::new();
        let blake2s = Blake2s256Engine::new();

        // BLAKE2 should report SIMD as enabled
        assert!(blake2b256.is_simd_enabled());
        assert!(blake2b512.is_simd_enabled());
        assert!(blake2s.is_simd_enabled());
    }

    #[test]
    fn test_hash_registry_blake2() {
        let registry = HashRegistry::new();

        // Should have BLAKE2b-256
        let blake2b256 = registry.get(Code::Blake2b256);
        assert!(blake2b256.is_some());

        // Should have BLAKE2b-512
        let blake2b512 = registry.get(Code::Blake2b512);
        assert!(blake2b512.is_some());

        // Should have BLAKE2s-256
        let blake2s256 = registry.get(Code::Blake2s256);
        assert!(blake2s256.is_some());
    }

    #[test]
    fn test_registry_digest_blake2() {
        let registry = HashRegistry::new();

        let hash256 = registry.digest(Code::Blake2b256, b"test").unwrap();
        assert_eq!(hash256.len(), 32);

        let hash512 = registry.digest(Code::Blake2b512, b"test").unwrap();
        assert_eq!(hash512.len(), 64);

        let hashs = registry.digest(Code::Blake2s256, b"test").unwrap();
        assert_eq!(hashs.len(), 32);
    }

    #[test]
    fn test_blake2_names() {
        assert_eq!(Blake2b256Engine::new().name(), "blake2b-256");
        assert_eq!(Blake2b512Engine::new().name(), "blake2b-512");
        assert_eq!(Blake2s256Engine::new().name(), "blake2s-256");
    }

    #[test]
    fn test_blake2_codes() {
        assert_eq!(Blake2b256Engine::new().code(), Code::Blake2b256);
        assert_eq!(Blake2b512Engine::new().code(), Code::Blake2b512);
        assert_eq!(Blake2s256Engine::new().code(), Code::Blake2s256);
    }
}
