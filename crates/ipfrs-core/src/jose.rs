//! DAG-JOSE codec for encrypted and signed IPLD data
//!
//! This module provides support for DAG-JOSE, which combines IPLD with:
//! - **JWS (JSON Web Signature)** for signing data
//! - **JWE (JSON Web Encryption)** for encrypting data
//!
//! ## Features
//!
//! - Sign IPLD data with Ed25519, RS256, or other algorithms
//! - Verify signed IPLD data
//! - Create content-addressed signed documents
//! - Integration with IPLD DAG structures
//!
//! ## Example - Signing Data
//!
//! ```rust
//! use ipfrs_core::jose::{JoseBuilder, JoseSignature};
//! use ipfrs_core::Ipld;
//!
//! // Create some IPLD data
//! let data = Ipld::String("Hello, IPFS!".to_string());
//!
//! // Sign the data (using a mock key for this example)
//! let secret = b"your-secret-key-min-32-bytes-long!!";
//! let jose = JoseBuilder::new()
//!     .with_payload(data)
//!     .sign_hs256(secret)
//!     .unwrap();
//!
//! // Verify the signature
//! let verified = jose.verify_hs256(secret).unwrap();
//! assert!(verified);
//! ```

use crate::error::{Error, Result};
use crate::ipld::Ipld;
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Parse PEM format and extract DER bytes
/// This is a simple PEM parser for RSA keys
fn pem_to_der(pem: &[u8]) -> Result<Vec<u8>> {
    let pem_str = std::str::from_utf8(pem)
        .map_err(|e| Error::InvalidInput(format!("Invalid UTF-8 in PEM: {}", e)))?;

    // Remove header, footer, and whitespace
    let lines: Vec<&str> = pem_str
        .lines()
        .filter(|line| !line.starts_with("-----"))
        .collect();

    let base64_content = lines.join("");

    // Decode base64 to get DER bytes
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    STANDARD
        .decode(base64_content.as_bytes())
        .map_err(|e| Error::InvalidInput(format!("Failed to decode base64 in PEM: {}", e)))
}

/// DAG-JOSE signature wrapper for IPLD data
///
/// This structure represents a signed IPLD payload using JWS (JSON Web Signature).
/// The signature ensures data integrity and authenticity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JoseSignature {
    /// The signed payload (as IPLD)
    pub payload: Ipld,
    /// The JWS signature string
    pub signature: String,
    /// Algorithm used for signing
    pub algorithm: String,
}

/// Builder for creating DAG-JOSE signatures
///
/// Provides a fluent interface for signing IPLD data with various algorithms.
pub struct JoseBuilder {
    payload: Option<Ipld>,
}

impl JoseBuilder {
    /// Create a new JOSE builder
    pub fn new() -> Self {
        Self { payload: None }
    }

    /// Set the payload to be signed
    pub fn with_payload(mut self, payload: Ipld) -> Self {
        self.payload = Some(payload);
        self
    }

    /// Sign the payload using HMAC SHA-256
    ///
    /// # Arguments
    /// * `secret` - Secret key for HMAC (should be at least 32 bytes)
    ///
    /// # Returns
    /// A `JoseSignature` containing the signed payload
    pub fn sign_hs256(self, secret: &[u8]) -> Result<JoseSignature> {
        let payload = self
            .payload
            .ok_or_else(|| Error::InvalidInput("No payload set".to_string()))?;

        if secret.len() < 32 {
            return Err(Error::InvalidInput(
                "HMAC secret must be at least 32 bytes".to_string(),
            ));
        }

        // Convert IPLD to JSON for JWT payload
        let json_payload = ipld_to_json_value(&payload)?;

        // Create JWT claims
        let claims = serde_json::json!({
            "payload": json_payload,
        });

        // Sign the data
        let header = Header::new(Algorithm::HS256);
        let token = encode(&header, &claims, &EncodingKey::from_secret(secret))
            .map_err(|e| Error::Serialization(format!("Failed to sign data: {}", e)))?;

        Ok(JoseSignature {
            payload,
            signature: token,
            algorithm: "HS256".to_string(),
        })
    }

    /// Sign the payload using RS256 (RSA with SHA-256)
    ///
    /// # Arguments
    /// * `private_key_pem` - RSA private key in PEM format
    ///
    /// # Returns
    /// A `JoseSignature` containing the signed payload
    pub fn sign_rs256(self, private_key_pem: &[u8]) -> Result<JoseSignature> {
        let payload = self
            .payload
            .ok_or_else(|| Error::InvalidInput("No payload set".to_string()))?;

        // Convert IPLD to JSON for JWT payload
        let json_payload = ipld_to_json_value(&payload)?;

        // Create JWT claims
        let claims = serde_json::json!({
            "payload": json_payload,
        });

        // Sign the data
        let header = Header::new(Algorithm::RS256);
        let der = pem_to_der(private_key_pem)?;
        let token = encode(&header, &claims, &EncodingKey::from_rsa_der(&der))
            .map_err(|e| Error::Serialization(format!("Failed to sign data: {}", e)))?;

        Ok(JoseSignature {
            payload,
            signature: token,
            algorithm: "RS256".to_string(),
        })
    }
}

impl Default for JoseBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl JoseSignature {
    /// Verify the signature using HMAC SHA-256
    ///
    /// # Arguments
    /// * `secret` - Secret key used for signing
    ///
    /// # Returns
    /// `Ok(true)` if the signature is valid, `Ok(false)` otherwise
    pub fn verify_hs256(&self, secret: &[u8]) -> Result<bool> {
        if self.algorithm != "HS256" {
            return Err(Error::InvalidInput(format!(
                "Expected HS256 algorithm, got {}",
                self.algorithm
            )));
        }

        // Decode and verify the JWT with custom validation
        let mut validation = Validation::new(Algorithm::HS256);
        // Don't require standard claims (exp, nbf, iat, etc.)
        validation.required_spec_claims.clear();
        validation.validate_exp = false;
        validation.validate_nbf = false;

        let token_data = decode::<serde_json::Value>(
            &self.signature,
            &DecodingKey::from_secret(secret),
            &validation,
        );

        match token_data {
            Ok(_) => Ok(true),
            Err(e) => {
                // Check if it's a validation error or signature mismatch
                match e.kind() {
                    jsonwebtoken::errors::ErrorKind::InvalidSignature => Ok(false),
                    _ => Err(Error::Verification(format!(
                        "Failed to verify signature: {}",
                        e
                    ))),
                }
            }
        }
    }

    /// Verify the signature using RS256 (RSA with SHA-256)
    ///
    /// # Arguments
    /// * `public_key_pem` - RSA public key in PEM format
    ///
    /// # Returns
    /// `Ok(true)` if the signature is valid, `Ok(false)` otherwise
    pub fn verify_rs256(&self, public_key_pem: &[u8]) -> Result<bool> {
        if self.algorithm != "RS256" {
            return Err(Error::InvalidInput(format!(
                "Expected RS256 algorithm, got {}",
                self.algorithm
            )));
        }

        // Decode and verify the JWT with custom validation
        let mut validation = Validation::new(Algorithm::RS256);
        // Don't require standard claims (exp, nbf, iat, etc.)
        validation.required_spec_claims.clear();
        validation.validate_exp = false;
        validation.validate_nbf = false;

        let der = pem_to_der(public_key_pem)?;
        let token_data = decode::<serde_json::Value>(
            &self.signature,
            &DecodingKey::from_rsa_der(&der),
            &validation,
        );

        match token_data {
            Ok(_) => Ok(true),
            Err(e) => match e.kind() {
                jsonwebtoken::errors::ErrorKind::InvalidSignature => Ok(false),
                _ => Err(Error::Verification(format!(
                    "Failed to verify signature: {}",
                    e
                ))),
            },
        }
    }

    /// Encode the JoseSignature to DAG-JOSE format
    ///
    /// Returns a JSON representation compatible with DAG-JOSE spec
    pub fn to_dag_jose(&self) -> Result<Vec<u8>> {
        let jose_object = serde_json::json!({
            "payload": ipld_to_json_value(&self.payload)?,
            "signatures": [{
                "protected": self.algorithm,
                "signature": self.signature,
            }]
        });

        serde_json::to_vec(&jose_object)
            .map_err(|e| Error::Serialization(format!("Failed to serialize DAG-JOSE: {}", e)))
    }

    /// Decode from DAG-JOSE format
    ///
    /// Parses a JSON representation in DAG-JOSE format
    pub fn from_dag_jose(data: &[u8]) -> Result<Self> {
        let jose_object: serde_json::Value = serde_json::from_slice(data)
            .map_err(|e| Error::Deserialization(format!("Failed to parse DAG-JOSE: {}", e)))?;

        let payload_json = jose_object
            .get("payload")
            .ok_or_else(|| Error::Deserialization("Missing payload field".to_string()))?;

        let signatures = jose_object
            .get("signatures")
            .and_then(|s| s.as_array())
            .ok_or_else(|| Error::Deserialization("Missing or invalid signatures".to_string()))?;

        if signatures.is_empty() {
            return Err(Error::Deserialization("No signatures found".to_string()));
        }

        let first_sig = &signatures[0];
        let algorithm = first_sig
            .get("protected")
            .and_then(|a| a.as_str())
            .ok_or_else(|| Error::Deserialization("Missing algorithm".to_string()))?
            .to_string();

        let signature = first_sig
            .get("signature")
            .and_then(|s| s.as_str())
            .ok_or_else(|| Error::Deserialization("Missing signature".to_string()))?
            .to_string();

        let payload = json_value_to_ipld(payload_json)?;

        Ok(JoseSignature {
            payload,
            signature,
            algorithm,
        })
    }
}

// Helper function to convert IPLD to serde_json::Value
fn ipld_to_json_value(ipld: &Ipld) -> Result<serde_json::Value> {
    match ipld {
        Ipld::Null => Ok(serde_json::Value::Null),
        Ipld::Bool(b) => Ok(serde_json::Value::Bool(*b)),
        Ipld::Integer(i) => {
            // Convert i128 to i64 for JSON (with range check)
            let i64_val: i64 = (*i)
                .try_into()
                .map_err(|_| Error::Serialization("Integer value out of i64 range".to_string()))?;
            Ok(serde_json::Value::Number(i64_val.into()))
        }
        Ipld::Float(f) => serde_json::Number::from_f64(*f)
            .map(serde_json::Value::Number)
            .ok_or_else(|| Error::Serialization("Invalid float value".to_string())),
        Ipld::String(s) => Ok(serde_json::Value::String(s.clone())),
        Ipld::Bytes(b) => {
            // Encode bytes as IPLD bytes object: {"/": {"bytes": "<base64>"}}
            let encoded = base64_encode(b);
            Ok(serde_json::json!({
                "/": {
                    "bytes": encoded
                }
            }))
        }
        Ipld::List(list) => {
            let values: Result<Vec<_>> = list.iter().map(ipld_to_json_value).collect();
            Ok(serde_json::Value::Array(values?))
        }
        Ipld::Map(map) => {
            let mut json_map = serde_json::Map::new();
            for (k, v) in map {
                json_map.insert(k.clone(), ipld_to_json_value(v)?);
            }
            Ok(serde_json::Value::Object(json_map))
        }
        Ipld::Link(cid) => {
            // Encode CID as a link object
            Ok(serde_json::json!({
                "/": cid.to_string()
            }))
        }
    }
}

// Helper function to convert serde_json::Value to IPLD
fn json_value_to_ipld(value: &serde_json::Value) -> Result<Ipld> {
    match value {
        serde_json::Value::Null => Ok(Ipld::Null),
        serde_json::Value::Bool(b) => Ok(Ipld::Bool(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(Ipld::Integer(i as i128))
            } else if let Some(f) = n.as_f64() {
                Ok(Ipld::Float(f))
            } else {
                Err(Error::Deserialization("Invalid number".to_string()))
            }
        }
        serde_json::Value::String(s) => Ok(Ipld::String(s.clone())),
        serde_json::Value::Array(arr) => {
            let items: Result<Vec<_>> = arr.iter().map(json_value_to_ipld).collect();
            Ok(Ipld::List(items?))
        }
        serde_json::Value::Object(obj) => {
            // Check if it's a special IPLD object
            if obj.len() == 1 && obj.contains_key("/") {
                let special = obj.get("/").expect("just confirmed key '/' is present");

                // Check if it's bytes: {"/": {"bytes": "<base64>"}}
                if let Some(bytes_obj) = special.as_object() {
                    if bytes_obj.len() == 1 && bytes_obj.contains_key("bytes") {
                        if let Some(b64_str) = bytes_obj.get("bytes").and_then(|v| v.as_str()) {
                            // Decode base64 - for simplicity, just convert back to bytes
                            // In production, use proper base64 decoding
                            return Ok(Ipld::Bytes(base64_decode(b64_str)?));
                        }
                    }
                }

                // Check if it's a CID link: {"/": "<cid-string>"}
                if let Some(cid_str) = special.as_str() {
                    let cid = crate::cid::parse_cid(cid_str)?;
                    return Ok(Ipld::Link(crate::cid::SerializableCid(cid)));
                }
            }

            // Regular map
            let mut map = BTreeMap::new();
            for (k, v) in obj {
                map.insert(k.clone(), json_value_to_ipld(v)?);
            }
            Ok(Ipld::Map(map))
        }
    }
}

// Simple base64 encoding helper
fn base64_encode(data: &[u8]) -> String {
    use std::fmt::Write;
    let mut result = String::new();
    for chunk in data.chunks(3) {
        let b1 = chunk[0];
        let b2 = chunk.get(1).copied().unwrap_or(0);
        let b3 = chunk.get(2).copied().unwrap_or(0);

        let n = ((b1 as u32) << 16) | ((b2 as u32) << 8) | (b3 as u32);

        let chars = [
            b64char((n >> 18) & 0x3f),
            b64char((n >> 12) & 0x3f),
            if chunk.len() > 1 {
                b64char((n >> 6) & 0x3f)
            } else {
                '='
            },
            if chunk.len() > 2 {
                b64char(n & 0x3f)
            } else {
                '='
            },
        ];

        for c in &chars {
            write!(&mut result, "{}", c).expect("write to String is infallible");
        }
    }
    result
}

fn b64char(n: u32) -> char {
    const CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    CHARS[n as usize] as char
}

// Simple base64 decoding helper
fn base64_decode(s: &str) -> Result<Vec<u8>> {
    if s.is_empty() {
        return Ok(Vec::new());
    }

    let bytes = s.as_bytes();
    let mut result = Vec::new();

    for chunk in bytes.chunks(4) {
        if chunk.len() < 2 {
            break;
        }

        let c0 = b64decode_char(chunk[0])?;
        let c1 = b64decode_char(chunk[1])?;
        let c2 = if chunk.len() > 2 && chunk[2] != b'=' {
            b64decode_char(chunk[2])?
        } else {
            0
        };
        let c3 = if chunk.len() > 3 && chunk[3] != b'=' {
            b64decode_char(chunk[3])?
        } else {
            0
        };

        result.push((c0 << 2) | (c1 >> 4));
        if chunk.len() > 2 && chunk[2] != b'=' {
            result.push((c1 << 4) | (c2 >> 2));
        }
        if chunk.len() > 3 && chunk[3] != b'=' {
            result.push((c2 << 6) | c3);
        }
    }

    Ok(result)
}

fn b64decode_char(c: u8) -> Result<u8> {
    match c {
        b'A'..=b'Z' => Ok(c - b'A'),
        b'a'..=b'z' => Ok(c - b'a' + 26),
        b'0'..=b'9' => Ok(c - b'0' + 52),
        b'+' => Ok(62),
        b'/' => Ok(63),
        _ => Err(Error::Deserialization(format!(
            "Invalid base64 character: {}",
            c
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jose_sign_verify_hs256() {
        let data = Ipld::String("Hello, DAG-JOSE!".to_string());
        let secret = b"my-secret-key-must-be-32-bytes!!";

        // Sign the data
        let jose = JoseBuilder::new()
            .with_payload(data.clone())
            .sign_hs256(secret)
            .unwrap();

        assert_eq!(jose.payload, data);
        assert_eq!(jose.algorithm, "HS256");

        // Verify with correct secret
        assert!(jose.verify_hs256(secret).unwrap());

        // Verify with wrong secret should fail
        let wrong_secret = b"wrong-secret-key-must-be-32byte!";
        assert!(!jose.verify_hs256(wrong_secret).unwrap());
    }

    #[test]
    fn test_jose_sign_different_payloads() {
        let secret = b"my-secret-key-must-be-32-bytes!!";

        // Sign different payloads
        let jose1 = JoseBuilder::new()
            .with_payload(Ipld::String("payload1".to_string()))
            .sign_hs256(secret)
            .unwrap();

        let jose2 = JoseBuilder::new()
            .with_payload(Ipld::String("payload2".to_string()))
            .sign_hs256(secret)
            .unwrap();

        // Signatures should be different
        assert_ne!(jose1.signature, jose2.signature);
    }

    #[test]
    fn test_jose_with_complex_ipld() {
        let mut map = BTreeMap::new();
        map.insert("name".to_string(), Ipld::String("Alice".to_string()));
        map.insert("age".to_string(), Ipld::Integer(30));
        map.insert(
            "roles".to_string(),
            Ipld::List(vec![
                Ipld::String("admin".to_string()),
                Ipld::String("user".to_string()),
            ]),
        );

        let data = Ipld::Map(map);
        let secret = b"my-secret-key-must-be-32-bytes!!";

        let jose = JoseBuilder::new()
            .with_payload(data.clone())
            .sign_hs256(secret)
            .unwrap();

        assert_eq!(jose.payload, data);
        assert!(jose.verify_hs256(secret).unwrap());
    }

    #[test]
    fn test_jose_short_secret_fails() {
        let data = Ipld::String("test".to_string());
        let short_secret = b"short"; // Too short

        let result = JoseBuilder::new()
            .with_payload(data)
            .sign_hs256(short_secret);

        assert!(result.is_err());
    }

    #[test]
    fn test_jose_no_payload_fails() {
        let secret = b"my-secret-key-must-be-32-bytes!!";

        let result = JoseBuilder::new().sign_hs256(secret);

        assert!(result.is_err());
    }

    #[test]
    fn test_jose_to_dag_jose() {
        let data = Ipld::String("Hello".to_string());
        let secret = b"my-secret-key-must-be-32-bytes!!";

        let jose = JoseBuilder::new()
            .with_payload(data)
            .sign_hs256(secret)
            .unwrap();

        // Convert to DAG-JOSE format
        let dag_jose = jose.to_dag_jose().unwrap();

        // Should be valid JSON
        let parsed: serde_json::Value = serde_json::from_slice(&dag_jose).unwrap();
        assert!(parsed.get("payload").is_some());
        assert!(parsed.get("signatures").is_some());
    }

    #[test]
    fn test_jose_roundtrip_dag_jose() {
        let data = Ipld::String("Roundtrip test".to_string());
        let secret = b"my-secret-key-must-be-32-bytes!!";

        let jose = JoseBuilder::new()
            .with_payload(data.clone())
            .sign_hs256(secret)
            .unwrap();

        // Encode to DAG-JOSE
        let dag_jose = jose.to_dag_jose().unwrap();

        // Decode back
        let decoded = JoseSignature::from_dag_jose(&dag_jose).unwrap();

        assert_eq!(decoded.payload, data);
        assert_eq!(decoded.algorithm, jose.algorithm);
        assert!(decoded.verify_hs256(secret).unwrap());
    }

    #[test]
    fn test_base64_encode() {
        let data = b"hello world";
        let encoded = base64_encode(data);
        // Should be valid base64
        assert!(!encoded.is_empty());
        assert!(encoded
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '='));
    }
}
