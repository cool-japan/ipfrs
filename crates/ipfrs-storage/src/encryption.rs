//! Encryption at rest for block storage.
//!
//! This module provides transparent encryption/decryption for any BlockStore implementation.
//! Supports multiple cipher algorithms with minimal performance overhead.
//!
//! # Features
//! - ChaCha20-Poly1305 and AES-256-GCM ciphers
//! - Argon2 key derivation from passwords
//! - Transparent encryption wrapper for any BlockStore
//! - Per-block nonce generation for security
//! - Zeroization of sensitive key material

use crate::traits::BlockStore;
use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Nonce as AesNonce,
};
use argon2::password_hash::{PasswordHash, PasswordVerifier, SaltString};
use argon2::{Argon2, PasswordHasher};
use async_trait::async_trait;
use bytes::Bytes;
use chacha20poly1305::{ChaCha20Poly1305, Nonce as ChachaNonce};
use ipfrs_core::{Block, Cid, Error, Result};
use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Supported cipher algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Cipher {
    /// ChaCha20-Poly1305 (fast on all platforms, especially ARM)
    ChaCha20Poly1305,
    /// AES-256-GCM (hardware-accelerated on modern x86/ARM)
    Aes256Gcm,
}

impl Cipher {
    /// Get the key size in bytes for this cipher
    pub fn key_size(&self) -> usize {
        match self {
            Cipher::ChaCha20Poly1305 => 32,
            Cipher::Aes256Gcm => 32,
        }
    }

    /// Get the nonce size in bytes for this cipher
    pub fn nonce_size(&self) -> usize {
        match self {
            Cipher::ChaCha20Poly1305 => 12,
            Cipher::Aes256Gcm => 12,
        }
    }

    /// Get the authentication tag size in bytes
    pub fn tag_size(&self) -> usize {
        16 // Both ciphers use 128-bit tags
    }
}

/// Encryption key with automatic zeroization
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct EncryptionKey {
    #[zeroize(skip)]
    cipher: Cipher,
    key_bytes: Vec<u8>,
}

impl EncryptionKey {
    /// Create a new encryption key from raw bytes
    pub fn from_bytes(cipher: Cipher, key_bytes: Vec<u8>) -> Result<Self> {
        if key_bytes.len() != cipher.key_size() {
            return Err(Error::InvalidInput(format!(
                "Invalid key size: expected {}, got {}",
                cipher.key_size(),
                key_bytes.len()
            )));
        }

        Ok(Self { cipher, key_bytes })
    }

    /// Generate a random encryption key
    pub fn generate(cipher: Cipher) -> Self {
        let mut rng = rand::rng();
        let key_bytes: Vec<u8> = (0..cipher.key_size())
            .map(|_| rng.random_range(0..=255))
            .collect();

        Self { cipher, key_bytes }
    }

    /// Derive a key from a password using Argon2
    pub fn derive_from_password(
        cipher: Cipher,
        password: &[u8],
        salt: Option<&[u8]>,
    ) -> Result<(Self, Vec<u8>)> {
        let argon2 = Argon2::default();

        let salt_string = if let Some(salt_bytes) = salt {
            SaltString::encode_b64(salt_bytes)
                .map_err(|e| Error::InvalidInput(format!("Invalid salt: {e}")))?
        } else {
            SaltString::generate(&mut OsRng)
        };

        let password_hash = argon2
            .hash_password(password, &salt_string)
            .map_err(|e| Error::Encryption(format!("Key derivation failed: {e}")))?;

        let hash_output = password_hash
            .hash
            .ok_or_else(|| Error::Encryption("No hash output".to_string()))?;

        let hash_bytes = hash_output.as_bytes();

        // Take first 32 bytes for the key
        let key_bytes = hash_bytes[..cipher.key_size()].to_vec();
        let salt_bytes = salt_string.as_str().as_bytes().to_vec();

        Ok((Self { cipher, key_bytes }, salt_bytes))
    }

    /// Verify a password against a previously derived key
    #[allow(dead_code)]
    pub fn verify_password(password: &[u8], salt: &[u8]) -> Result<()> {
        let salt_string = SaltString::encode_b64(salt)
            .map_err(|e| Error::InvalidInput(format!("Invalid salt: {e}")))?;

        let argon2 = Argon2::default();
        let password_hash = argon2
            .hash_password(password, &salt_string)
            .map_err(|e| Error::Encryption(format!("Password verification failed: {e}")))?;

        let hash_string = password_hash.to_string();
        let parsed_hash = PasswordHash::new(&hash_string)
            .map_err(|e| Error::Encryption(format!("Failed to parse hash: {e}")))?;

        argon2
            .verify_password(password, &parsed_hash)
            .map_err(|e| Error::Encryption(format!("Password verification failed: {e}")))?;

        Ok(())
    }

    /// Get the cipher type
    pub fn cipher(&self) -> Cipher {
        self.cipher
    }

    /// Encrypt data with this key
    pub fn encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>> {
        // Generate random nonce
        let mut rng = rand::rng();
        let nonce: Vec<u8> = (0..self.cipher.nonce_size())
            .map(|_| rng.random_range(0..=255))
            .collect();

        let ciphertext = match self.cipher {
            Cipher::ChaCha20Poly1305 => {
                let cipher = ChaCha20Poly1305::new_from_slice(&self.key_bytes)
                    .map_err(|e| Error::Encryption(format!("Cipher init failed: {e}")))?;
                let nonce_array = ChachaNonce::from_slice(&nonce);
                cipher
                    .encrypt(nonce_array, plaintext)
                    .map_err(|e| Error::Encryption(format!("Encryption failed: {e}")))?
            }
            Cipher::Aes256Gcm => {
                let cipher = Aes256Gcm::new_from_slice(&self.key_bytes)
                    .map_err(|e| Error::Encryption(format!("Cipher init failed: {e}")))?;
                let nonce_array = AesNonce::from_slice(&nonce);
                cipher
                    .encrypt(nonce_array, plaintext)
                    .map_err(|e| Error::Encryption(format!("Encryption failed: {e}")))?
            }
        };

        // Format: nonce || ciphertext (includes auth tag)
        let mut result = nonce;
        result.extend_from_slice(&ciphertext);

        Ok(result)
    }

    /// Decrypt data with this key
    pub fn decrypt(&self, ciphertext: &[u8]) -> Result<Vec<u8>> {
        let nonce_size = self.cipher.nonce_size();

        if ciphertext.len() < nonce_size {
            return Err(Error::Encryption("Invalid ciphertext format".to_string()));
        }

        let (nonce, encrypted_data) = ciphertext.split_at(nonce_size);

        let plaintext = match self.cipher {
            Cipher::ChaCha20Poly1305 => {
                let cipher = ChaCha20Poly1305::new_from_slice(&self.key_bytes)
                    .map_err(|e| Error::Encryption(format!("Cipher init failed: {e}")))?;
                let nonce_array = ChachaNonce::from_slice(nonce);
                cipher
                    .decrypt(nonce_array, encrypted_data)
                    .map_err(|e| Error::Encryption(format!("Decryption failed: {e}")))?
            }
            Cipher::Aes256Gcm => {
                let cipher = Aes256Gcm::new_from_slice(&self.key_bytes)
                    .map_err(|e| Error::Encryption(format!("Cipher init failed: {e}")))?;
                let nonce_array = AesNonce::from_slice(nonce);
                cipher
                    .decrypt(nonce_array, encrypted_data)
                    .map_err(|e| Error::Encryption(format!("Decryption failed: {e}")))?
            }
        };

        Ok(plaintext)
    }
}

/// Configuration for encrypted block store
#[derive(Debug, Clone)]
pub struct EncryptionConfig {
    /// Cipher algorithm to use
    pub cipher: Cipher,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            cipher: Cipher::ChaCha20Poly1305, // Fast on all platforms
        }
    }
}

/// Transparent encryption wrapper for any BlockStore
pub struct EncryptedBlockStore<S> {
    inner: S,
    key: Arc<EncryptionKey>,
    #[allow(dead_code)]
    config: EncryptionConfig,
}

impl<S> EncryptedBlockStore<S> {
    /// Create a new encrypted block store
    pub fn new(inner: S, key: EncryptionKey, config: EncryptionConfig) -> Self {
        Self {
            inner,
            key: Arc::new(key),
            config,
        }
    }

    /// Create with password-derived key
    pub fn with_password(
        inner: S,
        password: &[u8],
        salt: Option<&[u8]>,
        config: EncryptionConfig,
    ) -> Result<(Self, Vec<u8>)> {
        let (key, salt_bytes) = EncryptionKey::derive_from_password(config.cipher, password, salt)?;

        Ok((Self::new(inner, key, config), salt_bytes))
    }

    /// Get the underlying store
    pub fn into_inner(self) -> S {
        self.inner
    }

    /// Get a reference to the underlying store
    pub fn inner(&self) -> &S {
        &self.inner
    }
}

#[async_trait]
impl<S: BlockStore> BlockStore for EncryptedBlockStore<S> {
    async fn put(&self, block: &Block) -> Result<()> {
        // Encrypt the block data
        let ciphertext = self.key.encrypt(block.data())?;

        // Create new block with encrypted data
        let encrypted_block = Block::from_parts(*block.cid(), Bytes::from(ciphertext));

        self.inner.put(&encrypted_block).await
    }

    async fn get(&self, cid: &Cid) -> Result<Option<Block>> {
        let encrypted_block = self.inner.get(cid).await?;

        match encrypted_block {
            Some(block) => {
                let plaintext = self.key.decrypt(block.data())?;
                Ok(Some(Block::from_parts(*cid, Bytes::from(plaintext))))
            }
            None => Ok(None),
        }
    }

    async fn has(&self, cid: &Cid) -> Result<bool> {
        self.inner.has(cid).await
    }

    async fn delete(&self, cid: &Cid) -> Result<()> {
        self.inner.delete(cid).await
    }

    fn list_cids(&self) -> Result<Vec<Cid>> {
        self.inner.list_cids()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    async fn flush(&self) -> Result<()> {
        self.inner.flush().await
    }

    async fn close(&self) -> Result<()> {
        self.inner.close().await
    }

    async fn put_many(&self, blocks: &[Block]) -> Result<()> {
        let encrypted_blocks: Result<Vec<_>> = blocks
            .iter()
            .map(|block| {
                let ciphertext = self.key.encrypt(block.data())?;
                Ok(Block::from_parts(*block.cid(), Bytes::from(ciphertext)))
            })
            .collect();

        self.inner.put_many(&encrypted_blocks?).await
    }

    async fn get_many(&self, cids: &[Cid]) -> Result<Vec<Option<Block>>> {
        let encrypted_results = self.inner.get_many(cids).await?;

        let decrypted_results: Result<Vec<_>> = encrypted_results
            .into_iter()
            .enumerate()
            .map(|(i, opt_block)| match opt_block {
                Some(block) => {
                    let plaintext = self.key.decrypt(block.data())?;
                    Ok(Some(Block::from_parts(cids[i], Bytes::from(plaintext))))
                }
                None => Ok(None),
            })
            .collect();

        decrypted_results
    }

    async fn has_many(&self, cids: &[Cid]) -> Result<Vec<bool>> {
        self.inner.has_many(cids).await
    }

    async fn delete_many(&self, cids: &[Cid]) -> Result<()> {
        self.inner.delete_many(cids).await
    }
}

#[cfg(all(test, feature = "sled-backend"))]
mod tests {
    use super::*;
    use crate::blockstore::{BlockStoreConfig, SledBlockStore};

    #[test]
    fn test_cipher_sizes() {
        assert_eq!(Cipher::ChaCha20Poly1305.key_size(), 32);
        assert_eq!(Cipher::ChaCha20Poly1305.nonce_size(), 12);
        assert_eq!(Cipher::Aes256Gcm.key_size(), 32);
        assert_eq!(Cipher::Aes256Gcm.nonce_size(), 12);
    }

    #[test]
    fn test_key_generation() {
        let key1 = EncryptionKey::generate(Cipher::ChaCha20Poly1305);
        let key2 = EncryptionKey::generate(Cipher::ChaCha20Poly1305);

        // Keys should be different
        assert_ne!(key1.key_bytes, key2.key_bytes);
        assert_eq!(key1.key_bytes.len(), 32);
    }

    #[test]
    fn test_encrypt_decrypt_chacha() {
        let key = EncryptionKey::generate(Cipher::ChaCha20Poly1305);
        let plaintext = b"Hello, encrypted world!";

        let ciphertext = key
            .encrypt(plaintext)
            .expect("test: chacha encrypt should succeed");
        assert_ne!(ciphertext.as_slice(), plaintext);

        let decrypted = key
            .decrypt(&ciphertext)
            .expect("test: chacha decrypt should succeed");
        assert_eq!(decrypted.as_slice(), plaintext);
    }

    #[test]
    fn test_encrypt_decrypt_aes() {
        let key = EncryptionKey::generate(Cipher::Aes256Gcm);
        let plaintext = b"Hello, AES world!";

        let ciphertext = key
            .encrypt(plaintext)
            .expect("test: aes encrypt should succeed");
        assert_ne!(ciphertext.as_slice(), plaintext);

        let decrypted = key
            .decrypt(&ciphertext)
            .expect("test: aes decrypt should succeed");
        assert_eq!(decrypted.as_slice(), plaintext);
    }

    #[test]
    fn test_password_derivation() {
        let password = b"super_secret_password";
        let (key1, salt1) =
            EncryptionKey::derive_from_password(Cipher::ChaCha20Poly1305, password, None)
                .expect("test: password key derivation should succeed");

        // Verify key can encrypt/decrypt
        let plaintext = b"Test data";
        let ciphertext = key1
            .encrypt(plaintext)
            .expect("test: encrypt with derived key should succeed");
        let decrypted = key1
            .decrypt(&ciphertext)
            .expect("test: decrypt with derived key should succeed");
        assert_eq!(decrypted.as_slice(), plaintext);

        // Same password and salt should derive a key that works
        let (key2, _) =
            EncryptionKey::derive_from_password(Cipher::ChaCha20Poly1305, password, Some(&salt1))
                .expect("test: re-derivation with same salt should succeed");

        // key2 should be able to encrypt/decrypt as well
        let ciphertext2 = key2
            .encrypt(plaintext)
            .expect("test: encrypt with re-derived key should succeed");
        let decrypted2 = key2
            .decrypt(&ciphertext2)
            .expect("test: decrypt with re-derived key should succeed");
        assert_eq!(decrypted2.as_slice(), plaintext);

        // Different salt should give different key
        let (_key3, salt3) =
            EncryptionKey::derive_from_password(Cipher::ChaCha20Poly1305, password, None)
                .expect("test: second key derivation should succeed");

        // Salt should be different
        assert_ne!(salt1, salt3);
    }

    #[tokio::test]
    async fn test_encrypted_blockstore() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-encrypted-blockstore"),
            cache_size: 1024 * 1024,
        };

        // Clean up from previous test
        let _ = std::fs::remove_dir_all(&config.path);

        let inner = SledBlockStore::new(config).expect("test: SledBlockStore::new should succeed");
        let key = EncryptionKey::generate(Cipher::ChaCha20Poly1305);
        let config = EncryptionConfig::default();
        let store = EncryptedBlockStore::new(inner, key, config);

        // Create test data
        let data = Bytes::from("Test block data for encryption");
        let block = Block::new(data.clone()).expect("test: Block::new should succeed");

        // Put encrypted data
        store
            .put(&block)
            .await
            .expect("test: store.put should succeed");

        // Get and verify
        let retrieved = store
            .get(block.cid())
            .await
            .expect("test: store.get should succeed")
            .expect("test: block should exist");
        assert_eq!(retrieved.data(), &data);

        // Verify data is encrypted in inner store
        let inner_block = store
            .inner()
            .get(block.cid())
            .await
            .expect("test: inner store.get should succeed")
            .expect("test: inner block should exist");
        assert_ne!(inner_block.data(), &data);
        assert!(inner_block.data().len() > data.len()); // Overhead from nonce + tag
    }

    #[tokio::test]
    async fn test_encrypted_blockstore_batch_ops() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-test-encrypted-batch"),
            cache_size: 1024 * 1024,
        };

        // Clean up from previous test
        let _ = std::fs::remove_dir_all(&config.path);

        let inner = SledBlockStore::new(config).expect("test: SledBlockStore::new should succeed");
        let key = EncryptionKey::generate(Cipher::Aes256Gcm);
        let enc_config = EncryptionConfig {
            cipher: Cipher::Aes256Gcm,
        };
        let store = EncryptedBlockStore::new(inner, key, enc_config);

        // Create test blocks
        let blocks: Vec<_> = (0..10)
            .map(|i| {
                let data = Bytes::from(format!("Block {}", i));
                Block::new(data).expect("test: Block::new should succeed")
            })
            .collect();

        // Put many
        store
            .put_many(&blocks)
            .await
            .expect("test: put_many should succeed");

        // Get many
        let cids: Vec<_> = blocks.iter().map(|b| *b.cid()).collect();
        let retrieved = store
            .get_many(&cids)
            .await
            .expect("test: get_many should succeed");

        // Verify all blocks
        for (i, opt_block) in retrieved.iter().enumerate() {
            let block = opt_block.as_ref().expect("test: block should be present");
            assert_eq!(block.data(), blocks[i].data());
        }
    }

    #[test]
    fn test_wrong_key_fails() {
        let key1 = EncryptionKey::generate(Cipher::ChaCha20Poly1305);
        let key2 = EncryptionKey::generate(Cipher::ChaCha20Poly1305);
        let plaintext = b"Secret message";

        let ciphertext = key1
            .encrypt(plaintext)
            .expect("test: encrypt with key1 should succeed");

        // Decrypting with wrong key should fail
        assert!(key2.decrypt(&ciphertext).is_err());
    }

    #[test]
    fn test_invalid_ciphertext() {
        let key = EncryptionKey::generate(Cipher::ChaCha20Poly1305);
        let invalid_data = b"not encrypted data";

        // Should fail to decrypt
        assert!(key.decrypt(invalid_data).is_err());
    }
}
