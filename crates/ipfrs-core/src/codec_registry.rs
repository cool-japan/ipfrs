//! Codec registry system for pluggable encoding/decoding.
//!
//! This module provides a trait-based system for registering and using different
//! codecs for IPLD data encoding/decoding. Similar to the hash registry, this allows
//! runtime codec selection and custom codec implementations.
//!
//! # Examples
//!
//! ```rust
//! use ipfrs_core::{Codec, CodecRegistry, Ipld};
//! use ipfrs_core::cid::codec;
//!
//! // Get the global codec registry
//! let registry = ipfrs_core::global_codec_registry();
//!
//! // Encode data with DAG-CBOR
//! let data = Ipld::String("Hello, IPLD!".to_string());
//! let encoded = registry.encode(codec::DAG_CBOR, &data).unwrap();
//!
//! // Decode back
//! let decoded = registry.decode(codec::DAG_CBOR, &encoded).unwrap();
//! assert_eq!(data, decoded);
//! ```

use crate::{Error, Ipld, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Trait for codec implementations.
///
/// Codecs encode and decode IPLD data to/from binary formats.
pub trait Codec: Send + Sync {
    /// Encode IPLD data to bytes.
    fn encode(&self, data: &Ipld) -> Result<Vec<u8>>;

    /// Decode bytes to IPLD data.
    fn decode(&self, bytes: &[u8]) -> Result<Ipld>;

    /// Get the codec code (multicodec identifier).
    fn code(&self) -> u64;

    /// Get the codec name.
    fn name(&self) -> &str;
}

/// DAG-CBOR codec implementation.
#[derive(Debug, Clone, Default)]
pub struct DagCborCodec;

impl Codec for DagCborCodec {
    fn encode(&self, data: &Ipld) -> Result<Vec<u8>> {
        data.to_dag_cbor()
    }

    fn decode(&self, bytes: &[u8]) -> Result<Ipld> {
        Ipld::from_dag_cbor(bytes)
    }

    fn code(&self) -> u64 {
        crate::cid::codec::DAG_CBOR
    }

    fn name(&self) -> &str {
        "dag-cbor"
    }
}

/// DAG-JSON codec implementation.
#[derive(Debug, Clone, Default)]
pub struct DagJsonCodec;

impl Codec for DagJsonCodec {
    fn encode(&self, data: &Ipld) -> Result<Vec<u8>> {
        let json_str = data.to_dag_json()?;
        Ok(json_str.into_bytes())
    }

    fn decode(&self, bytes: &[u8]) -> Result<Ipld> {
        let json_str = std::str::from_utf8(bytes)
            .map_err(|e| Error::Deserialization(format!("Invalid UTF-8: {}", e)))?;
        Ipld::from_dag_json(json_str)
    }

    fn code(&self) -> u64 {
        crate::cid::codec::DAG_JSON
    }

    fn name(&self) -> &str {
        "dag-json"
    }
}

/// RAW codec implementation (no-op, stores bytes as-is).
#[derive(Debug, Clone, Default)]
pub struct RawCodec;

impl Codec for RawCodec {
    fn encode(&self, data: &Ipld) -> Result<Vec<u8>> {
        match data {
            Ipld::Bytes(bytes) => Ok(bytes.clone()),
            _ => Err(Error::Serialization(
                "RAW codec requires Ipld::Bytes".to_string(),
            )),
        }
    }

    fn decode(&self, bytes: &[u8]) -> Result<Ipld> {
        Ok(Ipld::Bytes(bytes.to_vec()))
    }

    fn code(&self) -> u64 {
        crate::cid::codec::RAW
    }

    fn name(&self) -> &str {
        "raw"
    }
}

/// Registry for codecs.
///
/// Allows runtime selection of encoding/decoding algorithms and
/// registration of custom codec implementations.
#[derive(Clone)]
pub struct CodecRegistry {
    codecs: Arc<RwLock<HashMap<u64, Arc<dyn Codec>>>>,
}

impl Default for CodecRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl CodecRegistry {
    /// Create a new codec registry with default codecs.
    ///
    /// Default codecs:
    /// - RAW (0x55)
    /// - DAG-CBOR (0x71)
    /// - DAG-JSON (0x0129)
    pub fn new() -> Self {
        let registry = Self {
            codecs: Arc::new(RwLock::new(HashMap::new())),
        };

        // Register default codecs
        registry.register(Arc::new(RawCodec));
        registry.register(Arc::new(DagCborCodec));
        registry.register(Arc::new(DagJsonCodec));

        registry
    }

    /// Register a codec implementation.
    ///
    /// If a codec with the same code already exists, it will be replaced.
    pub fn register(&self, codec: Arc<dyn Codec>) {
        let code = codec.code();
        let mut codecs = self.codecs.write().unwrap_or_else(|e| e.into_inner());
        codecs.insert(code, codec);
    }

    /// Get a codec by its code.
    pub fn get(&self, code: u64) -> Option<Arc<dyn Codec>> {
        let codecs = self.codecs.read().unwrap_or_else(|e| e.into_inner());
        codecs.get(&code).cloned()
    }

    /// Check if a codec is registered.
    pub fn has_codec(&self, code: u64) -> bool {
        let codecs = self.codecs.read().unwrap_or_else(|e| e.into_inner());
        codecs.contains_key(&code)
    }

    /// List all registered codec codes.
    pub fn list_codecs(&self) -> Vec<u64> {
        let codecs = self.codecs.read().unwrap_or_else(|e| e.into_inner());
        codecs.keys().copied().collect()
    }

    /// Get codec name by code.
    pub fn get_name(&self, code: u64) -> Option<String> {
        self.get(code).map(|c| c.name().to_string())
    }

    /// Encode IPLD data using the specified codec.
    pub fn encode(&self, codec_code: u64, data: &Ipld) -> Result<Vec<u8>> {
        let codec = self.get(codec_code).ok_or_else(|| {
            Error::Serialization(format!("Codec 0x{:x} not registered", codec_code))
        })?;
        codec.encode(data)
    }

    /// Decode bytes using the specified codec.
    pub fn decode(&self, codec_code: u64, bytes: &[u8]) -> Result<Ipld> {
        let codec = self.get(codec_code).ok_or_else(|| {
            Error::Deserialization(format!("Codec 0x{:x} not registered", codec_code))
        })?;
        codec.decode(bytes)
    }
}

/// Global codec registry instance.
static GLOBAL_CODEC_REGISTRY: std::sync::OnceLock<CodecRegistry> = std::sync::OnceLock::new();

/// Get the global codec registry.
///
/// # Examples
///
/// ```rust
/// use ipfrs_core::global_codec_registry;
/// use ipfrs_core::cid::codec;
///
/// let registry = global_codec_registry();
/// assert!(registry.has_codec(codec::DAG_CBOR));
/// assert!(registry.has_codec(codec::DAG_JSON));
/// assert!(registry.has_codec(codec::RAW));
/// ```
pub fn global_codec_registry() -> &'static CodecRegistry {
    GLOBAL_CODEC_REGISTRY.get_or_init(CodecRegistry::new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cid::codec;
    use std::collections::BTreeMap;

    #[test]
    fn test_raw_codec() {
        let codec = RawCodec;
        let data = Ipld::Bytes(b"hello".to_vec());

        let encoded = codec.encode(&data).unwrap();
        assert_eq!(encoded, b"hello");

        let decoded = codec.decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_raw_codec_requires_bytes() {
        let codec = RawCodec;
        let data = Ipld::String("hello".to_string());

        assert!(codec.encode(&data).is_err());
    }

    #[test]
    fn test_dag_cbor_codec() {
        let codec = DagCborCodec;
        let mut map = BTreeMap::new();
        map.insert("key".to_string(), Ipld::Integer(42));
        let data = Ipld::Map(map);

        let encoded = codec.encode(&data).unwrap();
        let decoded = codec.decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_dag_json_codec() {
        let codec = DagJsonCodec;
        let data = Ipld::String("hello".to_string());

        let encoded = codec.encode(&data).unwrap();
        let decoded = codec.decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_codec_registry() {
        let registry = CodecRegistry::new();

        // Default codecs should be registered
        assert!(registry.has_codec(codec::RAW));
        assert!(registry.has_codec(codec::DAG_CBOR));
        assert!(registry.has_codec(codec::DAG_JSON));

        // Get codec by code
        let cbor_codec = registry.get(codec::DAG_CBOR).unwrap();
        assert_eq!(cbor_codec.code(), codec::DAG_CBOR);
        assert_eq!(cbor_codec.name(), "dag-cbor");
    }

    #[test]
    fn test_registry_encode_decode() {
        let registry = CodecRegistry::new();
        let data = Ipld::Integer(12345);

        // Encode with DAG-CBOR
        let encoded = registry.encode(codec::DAG_CBOR, &data).unwrap();

        // Decode back
        let decoded = registry.decode(codec::DAG_CBOR, &encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_registry_list_codecs() {
        let registry = CodecRegistry::new();
        let codecs = registry.list_codecs();

        assert!(codecs.contains(&codec::RAW));
        assert!(codecs.contains(&codec::DAG_CBOR));
        assert!(codecs.contains(&codec::DAG_JSON));
        assert_eq!(codecs.len(), 3);
    }

    #[test]
    fn test_registry_get_name() {
        let registry = CodecRegistry::new();

        assert_eq!(registry.get_name(codec::RAW).unwrap(), "raw");
        assert_eq!(registry.get_name(codec::DAG_CBOR).unwrap(), "dag-cbor");
        assert_eq!(registry.get_name(codec::DAG_JSON).unwrap(), "dag-json");
        assert!(registry.get_name(0x9999).is_none());
    }

    #[test]
    fn test_unregistered_codec_error() {
        let registry = CodecRegistry::new();
        let data = Ipld::Integer(42);

        let result = registry.encode(0x9999, &data);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not registered"));
    }

    #[test]
    fn test_global_registry() {
        let registry = global_codec_registry();

        assert!(registry.has_codec(codec::RAW));
        assert!(registry.has_codec(codec::DAG_CBOR));
        assert!(registry.has_codec(codec::DAG_JSON));
    }

    #[test]
    fn test_codec_replacement() {
        let registry = CodecRegistry::new();

        // Register a custom codec with same code
        registry.register(Arc::new(RawCodec));

        // Should still work
        assert!(registry.has_codec(codec::RAW));
    }

    #[test]
    fn test_codec_codes() {
        assert_eq!(RawCodec.code(), codec::RAW);
        assert_eq!(DagCborCodec.code(), codec::DAG_CBOR);
        assert_eq!(DagJsonCodec.code(), codec::DAG_JSON);
    }

    #[test]
    fn test_codec_names() {
        assert_eq!(RawCodec.name(), "raw");
        assert_eq!(DagCborCodec.name(), "dag-cbor");
        assert_eq!(DagJsonCodec.name(), "dag-json");
    }
}
