//! Storage Encryption Layer
//!
//! Provides a simple XOR-based encryption layer for block storage.
//! This is for demonstration/educational purposes — not production cryptography.
//!
//! Supports two cipher modes:
//! - `Xor`: Repeating key XOR
//! - `XorWithNonce`: XOR with a key derived from nonce + base key

/// Cipher mode for the encryption layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CipherMode {
    /// Repeating key XOR
    Xor,
    /// XOR with key derived from nonce + key
    XorWithNonce,
}

/// Configuration for the encryption layer.
#[derive(Debug, Clone)]
pub struct EncryptionLayerConfig {
    /// Cipher mode to use
    pub mode: CipherMode,
    /// Encryption key bytes
    pub key: Vec<u8>,
    /// Nonce size in bytes (default 12)
    pub nonce_size: usize,
}

impl Default for EncryptionLayerConfig {
    fn default() -> Self {
        Self {
            mode: CipherMode::Xor,
            key: vec![0u8; 32],
            nonce_size: 12,
        }
    }
}

/// An encrypted block with metadata.
#[derive(Debug, Clone)]
pub struct EncryptedBlock {
    /// Content identifier
    pub cid: String,
    /// Encrypted data
    pub ciphertext: Vec<u8>,
    /// Nonce used for XorWithNonce mode
    pub nonce: Option<Vec<u8>>,
    /// Original plaintext size
    pub original_size: usize,
}

/// Statistics for the encryption layer.
#[derive(Debug, Clone)]
pub struct EncryptionLayerStats {
    /// Number of blocks encrypted
    pub blocks_encrypted: u64,
    /// Number of blocks decrypted
    pub blocks_decrypted: u64,
    /// Total bytes encrypted
    pub bytes_encrypted: u64,
    /// Total bytes decrypted
    pub bytes_decrypted: u64,
}

/// Storage encryption layer providing XOR-based encryption for blocks.
///
/// This is an educational/demonstration implementation. Do not use for
/// production security.
pub struct StorageEncryptionLayer {
    config: EncryptionLayerConfig,
    blocks_encrypted: u64,
    blocks_decrypted: u64,
    bytes_encrypted: u64,
    bytes_decrypted: u64,
}

impl StorageEncryptionLayer {
    /// Create a new encryption layer with the given configuration.
    pub fn new(config: EncryptionLayerConfig) -> Self {
        Self {
            config,
            blocks_encrypted: 0,
            blocks_decrypted: 0,
            bytes_encrypted: 0,
            bytes_decrypted: 0,
        }
    }

    /// Encrypt plaintext data for a given CID.
    ///
    /// For `Xor` mode, XORs plaintext with the repeating key.
    /// For `XorWithNonce` mode, generates a deterministic nonce from the CID,
    /// derives a working key by XORing the base key with the repeated nonce,
    /// then XORs the plaintext with the working key.
    pub fn encrypt(&mut self, cid: &str, plaintext: &[u8]) -> EncryptedBlock {
        let (ciphertext, nonce) = match self.config.mode {
            CipherMode::Xor => {
                let ct = xor_with_repeating_key(plaintext, &self.config.key);
                (ct, None)
            }
            CipherMode::XorWithNonce => {
                let nonce = Self::generate_nonce(cid, self.config.nonce_size);
                let working_key = Self::derive_key(&self.config.key, &nonce);
                let ct = xor_with_repeating_key(plaintext, &working_key);
                (ct, Some(nonce))
            }
        };

        self.blocks_encrypted += 1;
        self.bytes_encrypted += plaintext.len() as u64;

        EncryptedBlock {
            cid: cid.to_string(),
            ciphertext,
            nonce,
            original_size: plaintext.len(),
        }
    }

    /// Decrypt an encrypted block, returning the original plaintext.
    ///
    /// Returns an error if the decrypted size does not match `original_size`.
    pub fn decrypt(&mut self, block: &EncryptedBlock) -> Result<Vec<u8>, String> {
        if block.ciphertext.len() != block.original_size {
            return Err(format!(
                "ciphertext length {} does not match original_size {}",
                block.ciphertext.len(),
                block.original_size
            ));
        }

        let plaintext = match self.config.mode {
            CipherMode::Xor => xor_with_repeating_key(&block.ciphertext, &self.config.key),
            CipherMode::XorWithNonce => {
                let nonce = block.nonce.as_ref().ok_or_else(|| {
                    "XorWithNonce mode requires a nonce in the encrypted block".to_string()
                })?;
                let working_key = Self::derive_key(&self.config.key, nonce);
                xor_with_repeating_key(&block.ciphertext, &working_key)
            }
        };

        self.blocks_decrypted += 1;
        self.bytes_decrypted += plaintext.len() as u64;

        Ok(plaintext)
    }

    /// Derive a working key by XORing the base key with a repeated nonce.
    ///
    /// The output length matches the base key length. The nonce is repeated
    /// cyclically to cover the full key length.
    pub fn derive_key(base_key: &[u8], nonce: &[u8]) -> Vec<u8> {
        if nonce.is_empty() {
            return base_key.to_vec();
        }
        base_key
            .iter()
            .enumerate()
            .map(|(i, &b)| b ^ nonce[i % nonce.len()])
            .collect()
    }

    /// Generate a deterministic nonce from a CID using FNV-1a hash.
    ///
    /// Produces `size` bytes by repeatedly hashing the CID with different
    /// seed offsets derived from FNV-1a.
    pub fn generate_nonce(cid: &str, size: usize) -> Vec<u8> {
        let mut nonce = Vec::with_capacity(size);
        let cid_bytes = cid.as_bytes();

        // Generate nonce bytes using FNV-1a with varying seeds
        let mut remaining = size;
        let mut round: u64 = 0;
        while remaining > 0 {
            let hash = fnv1a_with_seed(cid_bytes, round);
            let hash_bytes = hash.to_le_bytes();
            let take = remaining.min(hash_bytes.len());
            nonce.extend_from_slice(&hash_bytes[..take]);
            remaining = remaining.saturating_sub(take);
            round += 1;
        }

        nonce.truncate(size);
        nonce
    }

    /// Heuristic check whether data appears to be encrypted.
    ///
    /// Estimates Shannon entropy; if > 7.5 bits/byte, data is likely
    /// encrypted or incompressible random data.
    pub fn is_encrypted(data: &[u8]) -> bool {
        if data.is_empty() {
            return false;
        }

        let mut freq = [0u64; 256];
        for &b in data {
            freq[b as usize] += 1;
        }

        let len = data.len() as f64;
        let mut entropy = 0.0_f64;
        for &count in &freq {
            if count > 0 {
                let p = count as f64 / len;
                entropy -= p * p.log2();
            }
        }

        entropy > 7.5
    }

    /// Return current encryption layer statistics.
    pub fn stats(&self) -> EncryptionLayerStats {
        EncryptionLayerStats {
            blocks_encrypted: self.blocks_encrypted,
            blocks_decrypted: self.blocks_decrypted,
            bytes_encrypted: self.bytes_encrypted,
            bytes_decrypted: self.bytes_decrypted,
        }
    }
}

/// FNV-1a hash with a seed offset for nonce generation.
fn fnv1a_with_seed(data: &[u8], seed: u64) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325_u64.wrapping_add(seed.wrapping_mul(0x100000001b3));
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// XOR data with a repeating key.
fn xor_with_repeating_key(data: &[u8], key: &[u8]) -> Vec<u8> {
    if key.is_empty() {
        return data.to_vec();
    }
    data.iter()
        .enumerate()
        .map(|(i, &b)| b ^ key[i % key.len()])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(mode: CipherMode) -> EncryptionLayerConfig {
        EncryptionLayerConfig {
            mode,
            key: vec![0xAB, 0xCD, 0xEF, 0x12, 0x34, 0x56, 0x78, 0x9A],
            nonce_size: 12,
        }
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip_xor() {
        let mut layer = StorageEncryptionLayer::new(make_config(CipherMode::Xor));
        let plaintext = b"Hello, IPFRS storage encryption!";
        let encrypted = layer.encrypt("QmTest1", plaintext);
        let decrypted = layer.decrypt(&encrypted).expect("decrypt should succeed");
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip_xor_with_nonce() {
        let mut layer = StorageEncryptionLayer::new(make_config(CipherMode::XorWithNonce));
        let plaintext = b"Hello, IPFRS storage encryption with nonce!";
        let encrypted = layer.encrypt("QmTest2", plaintext);
        assert!(encrypted.nonce.is_some());
        let decrypted = layer.decrypt(&encrypted).expect("decrypt should succeed");
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_xor_mode_correctness() {
        let mut layer = StorageEncryptionLayer::new(make_config(CipherMode::Xor));
        let plaintext = vec![0x00, 0xFF, 0x55, 0xAA];
        let encrypted = layer.encrypt("QmXor", &plaintext);
        // XOR with key: 0x00^0xAB=0xAB, 0xFF^0xCD=0x32, 0x55^0xEF=0xBA, 0xAA^0x12=0xB8
        assert_eq!(encrypted.ciphertext, vec![0xAB, 0x32, 0xBA, 0xB8]);
    }

    #[test]
    fn test_ciphertext_differs_from_plaintext() {
        let mut layer = StorageEncryptionLayer::new(make_config(CipherMode::Xor));
        let plaintext = b"This should be encrypted";
        let encrypted = layer.encrypt("QmDiff", plaintext);
        assert_ne!(encrypted.ciphertext, plaintext);
    }

    #[test]
    fn test_different_keys_different_ciphertext() {
        let config1 = EncryptionLayerConfig {
            mode: CipherMode::Xor,
            key: vec![0x01, 0x02, 0x03, 0x04],
            nonce_size: 12,
        };
        let config2 = EncryptionLayerConfig {
            mode: CipherMode::Xor,
            key: vec![0x05, 0x06, 0x07, 0x08],
            nonce_size: 12,
        };
        let mut layer1 = StorageEncryptionLayer::new(config1);
        let mut layer2 = StorageEncryptionLayer::new(config2);
        let plaintext = b"Same plaintext, different keys";
        let enc1 = layer1.encrypt("QmKeys", plaintext);
        let enc2 = layer2.encrypt("QmKeys", plaintext);
        assert_ne!(enc1.ciphertext, enc2.ciphertext);
    }

    #[test]
    fn test_deterministic_nonce_from_cid() {
        let nonce1 = StorageEncryptionLayer::generate_nonce("QmDeterministic", 12);
        let nonce2 = StorageEncryptionLayer::generate_nonce("QmDeterministic", 12);
        assert_eq!(nonce1, nonce2);
        assert_eq!(nonce1.len(), 12);
    }

    #[test]
    fn test_different_cids_different_nonces() {
        let nonce1 = StorageEncryptionLayer::generate_nonce("QmCid1", 12);
        let nonce2 = StorageEncryptionLayer::generate_nonce("QmCid2", 12);
        assert_ne!(nonce1, nonce2);
    }

    #[test]
    fn test_derive_key() {
        let base_key = vec![0xAA, 0xBB, 0xCC, 0xDD];
        let nonce = vec![0x11, 0x22];
        let derived = StorageEncryptionLayer::derive_key(&base_key, &nonce);
        // 0xAA^0x11=0xBB, 0xBB^0x22=0x99, 0xCC^0x11=0xDD, 0xDD^0x22=0xFF
        assert_eq!(derived, vec![0xBB, 0x99, 0xDD, 0xFF]);
    }

    #[test]
    fn test_derive_key_empty_nonce() {
        let base_key = vec![0xAA, 0xBB, 0xCC];
        let derived = StorageEncryptionLayer::derive_key(&base_key, &[]);
        assert_eq!(derived, base_key);
    }

    #[test]
    fn test_empty_plaintext() {
        let mut layer = StorageEncryptionLayer::new(make_config(CipherMode::Xor));
        let encrypted = layer.encrypt("QmEmpty", b"");
        assert!(encrypted.ciphertext.is_empty());
        assert_eq!(encrypted.original_size, 0);
        let decrypted = layer.decrypt(&encrypted).expect("decrypt should succeed");
        assert!(decrypted.is_empty());
    }

    #[test]
    fn test_empty_plaintext_with_nonce() {
        let mut layer = StorageEncryptionLayer::new(make_config(CipherMode::XorWithNonce));
        let encrypted = layer.encrypt("QmEmptyNonce", b"");
        assert!(encrypted.ciphertext.is_empty());
        let decrypted = layer.decrypt(&encrypted).expect("decrypt should succeed");
        assert!(decrypted.is_empty());
    }

    #[test]
    fn test_stats_tracking() {
        let mut layer = StorageEncryptionLayer::new(make_config(CipherMode::Xor));
        let s = layer.stats();
        assert_eq!(s.blocks_encrypted, 0);
        assert_eq!(s.blocks_decrypted, 0);

        let data1 = b"first block";
        let enc1 = layer.encrypt("QmStats1", data1);
        let data2 = b"second block data";
        let enc2 = layer.encrypt("QmStats2", data2);

        let s = layer.stats();
        assert_eq!(s.blocks_encrypted, 2);
        assert_eq!(s.bytes_encrypted, (data1.len() + data2.len()) as u64);

        let _ = layer.decrypt(&enc1).expect("decrypt should succeed");
        let _ = layer.decrypt(&enc2).expect("decrypt should succeed");

        let s = layer.stats();
        assert_eq!(s.blocks_decrypted, 2);
        assert_eq!(s.bytes_decrypted, (data1.len() + data2.len()) as u64);
    }

    #[test]
    fn test_large_block() {
        let mut layer = StorageEncryptionLayer::new(make_config(CipherMode::XorWithNonce));
        let plaintext: Vec<u8> = (0..10_000).map(|i| (i % 256) as u8).collect();
        let encrypted = layer.encrypt("QmLargeBlock", &plaintext);
        assert_eq!(encrypted.ciphertext.len(), plaintext.len());
        let decrypted = layer.decrypt(&encrypted).expect("decrypt should succeed");
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_key_shorter_than_data() {
        let config = EncryptionLayerConfig {
            mode: CipherMode::Xor,
            key: vec![0xFF],
            nonce_size: 12,
        };
        let mut layer = StorageEncryptionLayer::new(config);
        let plaintext = vec![0x00, 0x01, 0x02, 0x03, 0x04];
        let encrypted = layer.encrypt("QmShortKey", &plaintext);
        // Each byte XORed with 0xFF
        assert_eq!(encrypted.ciphertext, vec![0xFF, 0xFE, 0xFD, 0xFC, 0xFB]);
        let decrypted = layer.decrypt(&encrypted).expect("decrypt should succeed");
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_same_cid_same_nonce() {
        let nonce_a = StorageEncryptionLayer::generate_nonce("QmSameCid", 16);
        let nonce_b = StorageEncryptionLayer::generate_nonce("QmSameCid", 16);
        assert_eq!(nonce_a, nonce_b);
    }

    #[test]
    fn test_is_encrypted_random_data() {
        // Generate high-entropy data
        let data: Vec<u8> = (0..1000)
            .map(|i: u64| {
                let h = fnv1a_with_seed(&i.to_le_bytes(), 42);
                (h & 0xFF) as u8
            })
            .collect();
        assert!(StorageEncryptionLayer::is_encrypted(&data));
    }

    #[test]
    fn test_is_encrypted_low_entropy() {
        // Repetitive data has low entropy
        let data = vec![0xAA; 1000];
        assert!(!StorageEncryptionLayer::is_encrypted(&data));
    }

    #[test]
    fn test_is_encrypted_empty() {
        assert!(!StorageEncryptionLayer::is_encrypted(&[]));
    }

    #[test]
    fn test_decrypt_size_mismatch() {
        let mut layer = StorageEncryptionLayer::new(make_config(CipherMode::Xor));
        let bad_block = EncryptedBlock {
            cid: "QmBad".to_string(),
            ciphertext: vec![0x01, 0x02, 0x03],
            nonce: None,
            original_size: 5, // mismatch
        };
        let result = layer.decrypt(&bad_block);
        assert!(result.is_err());
    }

    #[test]
    fn test_decrypt_xor_with_nonce_missing_nonce() {
        let mut layer = StorageEncryptionLayer::new(make_config(CipherMode::XorWithNonce));
        let bad_block = EncryptedBlock {
            cid: "QmNoNonce".to_string(),
            ciphertext: vec![0x01, 0x02],
            nonce: None, // missing
            original_size: 2,
        };
        let result = layer.decrypt(&bad_block);
        assert!(result.is_err());
    }

    #[test]
    fn test_xor_with_nonce_different_cids_different_ciphertext() {
        let mut layer = StorageEncryptionLayer::new(make_config(CipherMode::XorWithNonce));
        let plaintext = b"Identical data for different CIDs";
        let enc1 = layer.encrypt("QmCidA", plaintext);
        let enc2 = layer.encrypt("QmCidB", plaintext);
        assert_ne!(enc1.ciphertext, enc2.ciphertext);
    }

    #[test]
    fn test_encrypted_block_cid_preserved() {
        let mut layer = StorageEncryptionLayer::new(make_config(CipherMode::Xor));
        let cid = "QmPreservedCid12345";
        let encrypted = layer.encrypt(cid, b"data");
        assert_eq!(encrypted.cid, cid);
    }

    #[test]
    fn test_generate_nonce_various_sizes() {
        for size in [0, 1, 8, 12, 16, 32, 64] {
            let nonce = StorageEncryptionLayer::generate_nonce("QmVarySize", size);
            assert_eq!(nonce.len(), size);
        }
    }

    #[test]
    fn test_xor_self_inverse() {
        // XOR is its own inverse: encrypt(encrypt(x)) == x
        let key = vec![0x42, 0x73, 0x99];
        let data = b"self inverse test data";
        let encrypted = xor_with_repeating_key(data, &key);
        let decrypted = xor_with_repeating_key(&encrypted, &key);
        assert_eq!(decrypted, data);
    }

    #[test]
    fn test_xor_with_empty_key() {
        let data = b"no encryption";
        let result = xor_with_repeating_key(data, &[]);
        assert_eq!(result, data);
    }

    #[test]
    fn test_default_config() {
        let config = EncryptionLayerConfig::default();
        assert_eq!(config.mode, CipherMode::Xor);
        assert_eq!(config.key.len(), 32);
        assert_eq!(config.nonce_size, 12);
    }

    #[test]
    fn test_stats_initial_zeros() {
        let layer = StorageEncryptionLayer::new(make_config(CipherMode::Xor));
        let s = layer.stats();
        assert_eq!(s.blocks_encrypted, 0);
        assert_eq!(s.blocks_decrypted, 0);
        assert_eq!(s.bytes_encrypted, 0);
        assert_eq!(s.bytes_decrypted, 0);
    }

    #[test]
    fn test_single_byte_data() {
        let mut layer = StorageEncryptionLayer::new(make_config(CipherMode::XorWithNonce));
        let plaintext = &[0x42];
        let encrypted = layer.encrypt("QmSingleByte", plaintext);
        let decrypted = layer.decrypt(&encrypted).expect("decrypt should succeed");
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_all_byte_values() {
        let mut layer = StorageEncryptionLayer::new(make_config(CipherMode::Xor));
        let plaintext: Vec<u8> = (0..=255).collect();
        let encrypted = layer.encrypt("QmAllBytes", &plaintext);
        let decrypted = layer.decrypt(&encrypted).expect("decrypt should succeed");
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_temp_dir_usage() {
        // Demonstrate use of temp_dir per policy (write encrypted data to temp file)
        let tmp = std::env::temp_dir().join("ipfrs_encryption_layer_test");
        let mut layer = StorageEncryptionLayer::new(make_config(CipherMode::Xor));
        let plaintext = b"temp dir test";
        let encrypted = layer.encrypt("QmTmpDir", plaintext);
        std::fs::write(&tmp, &encrypted.ciphertext).expect("write to temp should succeed");
        let read_back = std::fs::read(&tmp).expect("read from temp should succeed");
        assert_eq!(read_back, encrypted.ciphertext);
        let _ = std::fs::remove_file(&tmp);
    }
}
