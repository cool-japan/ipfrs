//! IPLD (InterPlanetary Linked Data) support
//!
//! This module provides IPLD data structure support for IPFRS with proper
//! DAG-CBOR and DAG-JSON codec implementations.

use crate::cid::Cid;
use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// CBOR tag for CID links in DAG-CBOR encoding (tag 42)
const CID_TAG: u64 = 42;

/// IPLD data model
///
/// Represents the core IPLD data types that can be stored and transferred
/// across the IPFRS network.
#[derive(Debug, Clone, PartialEq)]
pub enum Ipld {
    /// Null value
    Null,
    /// Boolean value
    Bool(bool),
    /// Integer value (supports full i128 range)
    Integer(i128),
    /// Float value (IEEE 754 double precision)
    Float(f64),
    /// String value (UTF-8)
    String(String),
    /// Bytes value (raw binary data)
    Bytes(Vec<u8>),
    /// List of IPLD values
    List(Vec<Ipld>),
    /// Map of string keys to IPLD values (keys are sorted)
    Map(BTreeMap<String, Ipld>),
    /// Link to another IPLD node via CID
    Link(crate::cid::SerializableCid),
}

impl Ipld {
    /// Create a link to a CID
    pub fn link(cid: Cid) -> Self {
        Ipld::Link(crate::cid::SerializableCid(cid))
    }

    /// Check if this is a link
    pub fn is_link(&self) -> bool {
        matches!(self, Ipld::Link(_))
    }

    /// Extract CID if this is a link
    pub fn as_link(&self) -> Option<&Cid> {
        match self {
            Ipld::Link(cid) => Some(&cid.0),
            _ => None,
        }
    }

    /// Encode this IPLD value to DAG-CBOR format
    ///
    /// DAG-CBOR is a deterministic subset of CBOR with:
    /// - Map keys sorted by byte ordering
    /// - No indefinite-length items
    /// - CID links encoded with tag 42
    pub fn to_dag_cbor(&self) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        encode_dag_cbor(self, &mut buffer)?;
        Ok(buffer)
    }

    /// Decode IPLD value from DAG-CBOR format
    pub fn from_dag_cbor(data: &[u8]) -> Result<Self> {
        decode_dag_cbor(&mut &data[..])
    }

    /// Encode this IPLD value to DAG-JSON format
    ///
    /// DAG-JSON is a JSON encoding for IPLD with special handling for:
    /// - Bytes (encoded as `{"/": {"bytes": "<base64>"}}`)
    /// - Links (encoded as `{"/": "<cid-string>"}`)
    pub fn to_dag_json(&self) -> Result<String> {
        let json_value = ipld_to_dag_json(self)?;
        serde_json::to_string_pretty(&json_value)
            .map_err(|e| Error::Serialization(format!("Failed to serialize DAG-JSON: {}", e)))
    }

    /// Decode IPLD value from DAG-JSON format
    pub fn from_dag_json(json: &str) -> Result<Self> {
        let json_value: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| Error::Deserialization(format!("Failed to parse DAG-JSON: {}", e)))?;
        dag_json_to_ipld(&json_value)
    }

    /// Encode this IPLD value to JSON format (simple, for debugging)
    pub fn to_json(&self) -> Result<String> {
        self.to_dag_json()
    }

    /// Decode IPLD value from JSON format
    pub fn from_json(json: &str) -> Result<Self> {
        Self::from_dag_json(json)
    }

    /// Get all CID links contained in this IPLD structure (recursively)
    pub fn links(&self) -> Vec<Cid> {
        let mut result = Vec::new();
        self.collect_links(&mut result);
        result
    }

    fn collect_links(&self, result: &mut Vec<Cid>) {
        match self {
            Ipld::Link(cid) => result.push(cid.0),
            Ipld::List(list) => {
                for item in list {
                    item.collect_links(result);
                }
            }
            Ipld::Map(map) => {
                for value in map.values() {
                    value.collect_links(result);
                }
            }
            _ => {}
        }
    }

    /// Check if this is a null value
    #[inline]
    pub const fn is_null(&self) -> bool {
        matches!(self, Ipld::Null)
    }

    /// Check if this is a boolean value
    #[inline]
    pub const fn is_bool(&self) -> bool {
        matches!(self, Ipld::Bool(_))
    }

    /// Check if this is an integer value
    #[inline]
    pub const fn is_integer(&self) -> bool {
        matches!(self, Ipld::Integer(_))
    }

    /// Check if this is a float value
    #[inline]
    pub const fn is_float(&self) -> bool {
        matches!(self, Ipld::Float(_))
    }

    /// Check if this is a string value
    #[inline]
    pub const fn is_string(&self) -> bool {
        matches!(self, Ipld::String(_))
    }

    /// Check if this is a bytes value
    #[inline]
    pub const fn is_bytes(&self) -> bool {
        matches!(self, Ipld::Bytes(_))
    }

    /// Check if this is a list value
    #[inline]
    pub const fn is_list(&self) -> bool {
        matches!(self, Ipld::List(_))
    }

    /// Check if this is a map value
    #[inline]
    pub const fn is_map(&self) -> bool {
        matches!(self, Ipld::Map(_))
    }

    /// Extract boolean value if this is a Bool
    #[inline]
    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            Ipld::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Extract integer value if this is an Integer
    #[inline]
    pub const fn as_integer(&self) -> Option<i128> {
        match self {
            Ipld::Integer(i) => Some(*i),
            _ => None,
        }
    }

    /// Extract float value if this is a Float
    #[inline]
    pub const fn as_float(&self) -> Option<f64> {
        match self {
            Ipld::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Extract string reference if this is a String
    #[inline]
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Ipld::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Extract bytes reference if this is Bytes
    #[inline]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Ipld::Bytes(b) => Some(b.as_slice()),
            _ => None,
        }
    }

    /// Extract list reference if this is a List
    #[inline]
    pub fn as_list(&self) -> Option<&[Ipld]> {
        match self {
            Ipld::List(l) => Some(l.as_slice()),
            _ => None,
        }
    }

    /// Extract map reference if this is a Map
    #[inline]
    pub fn as_map(&self) -> Option<&BTreeMap<String, Ipld>> {
        match self {
            Ipld::Map(m) => Some(m),
            _ => None,
        }
    }

    /// Get a value from a map by key (if this is a Map)
    #[inline]
    pub fn get(&self, key: &str) -> Option<&Ipld> {
        self.as_map()?.get(key)
    }

    /// Get a value from a list by index (if this is a List)
    #[inline]
    pub fn index(&self, idx: usize) -> Option<&Ipld> {
        self.as_list()?.get(idx)
    }

    /// Get the size/length of this IPLD value
    ///
    /// - For List: number of elements
    /// - For Map: number of key-value pairs
    /// - For String: length in bytes
    /// - For Bytes: length in bytes
    /// - For other types: 0
    pub fn len(&self) -> usize {
        match self {
            Ipld::List(l) => l.len(),
            Ipld::Map(m) => m.len(),
            Ipld::String(s) => s.len(),
            Ipld::Bytes(b) => b.len(),
            _ => 0,
        }
    }

    /// Check if this IPLD value is empty
    ///
    /// - For List/Map/String/Bytes: checks if length is 0
    /// - For Null: true
    /// - For other types: false
    pub fn is_empty(&self) -> bool {
        match self {
            Ipld::Null => true,
            Ipld::List(l) => l.is_empty(),
            Ipld::Map(m) => m.is_empty(),
            Ipld::String(s) => s.is_empty(),
            Ipld::Bytes(b) => b.is_empty(),
            _ => false,
        }
    }

    /// Get a human-readable type name for this IPLD value
    pub const fn type_name(&self) -> &'static str {
        match self {
            Ipld::Null => "null",
            Ipld::Bool(_) => "bool",
            Ipld::Integer(_) => "integer",
            Ipld::Float(_) => "float",
            Ipld::String(_) => "string",
            Ipld::Bytes(_) => "bytes",
            Ipld::List(_) => "list",
            Ipld::Map(_) => "map",
            Ipld::Link(_) => "link",
        }
    }
}

// =============================================================================
// DAG-CBOR Encoding
// =============================================================================

fn encode_dag_cbor(ipld: &Ipld, buffer: &mut Vec<u8>) -> Result<()> {
    match ipld {
        Ipld::Null => {
            // CBOR simple value 22 (null)
            buffer.push(0xf6);
        }
        Ipld::Bool(b) => {
            // CBOR simple values 20 (false) and 21 (true)
            buffer.push(if *b { 0xf5 } else { 0xf4 });
        }
        Ipld::Integer(i) => {
            encode_cbor_integer(*i, buffer)?;
        }
        Ipld::Float(f) => {
            // CBOR major type 7 with additional info 27 (64-bit float)
            buffer.push(0xfb);
            buffer.extend_from_slice(&f.to_be_bytes());
        }
        Ipld::String(s) => {
            // CBOR major type 3 (text string)
            encode_cbor_length(3, s.len() as u64, buffer);
            buffer.extend_from_slice(s.as_bytes());
        }
        Ipld::Bytes(b) => {
            // CBOR major type 2 (byte string)
            encode_cbor_length(2, b.len() as u64, buffer);
            buffer.extend_from_slice(b);
        }
        Ipld::List(list) => {
            // CBOR major type 4 (array)
            encode_cbor_length(4, list.len() as u64, buffer);
            for item in list {
                encode_dag_cbor(item, buffer)?;
            }
        }
        Ipld::Map(map) => {
            // CBOR major type 5 (map) - keys must be sorted by byte ordering
            encode_cbor_length(5, map.len() as u64, buffer);
            // BTreeMap already maintains sorted order
            for (key, value) in map {
                encode_cbor_length(3, key.len() as u64, buffer);
                buffer.extend_from_slice(key.as_bytes());
                encode_dag_cbor(value, buffer)?;
            }
        }
        Ipld::Link(cid) => {
            // DAG-CBOR uses tag 42 for CID links
            encode_cbor_tag(CID_TAG, buffer);
            // CID bytes with multibase identity prefix (0x00)
            let cid_bytes = cid.0.to_bytes();
            let mut prefixed = vec![0x00];
            prefixed.extend_from_slice(&cid_bytes);
            encode_cbor_length(2, prefixed.len() as u64, buffer);
            buffer.extend_from_slice(&prefixed);
        }
    }
    Ok(())
}

fn encode_cbor_integer(value: i128, buffer: &mut Vec<u8>) -> Result<()> {
    if value >= 0 {
        // Non-negative integers: CBOR major type 0
        let val = value as u64;
        encode_cbor_length(0, val, buffer);
    } else {
        // Negative integers: CBOR major type 1, encoded as -1-n
        let val = (-1 - value) as u64;
        encode_cbor_length(1, val, buffer);
    }
    Ok(())
}

fn encode_cbor_length(major_type: u8, length: u64, buffer: &mut Vec<u8>) {
    let mt = major_type << 5;
    if length < 24 {
        buffer.push(mt | length as u8);
    } else if length < 256 {
        buffer.push(mt | 24);
        buffer.push(length as u8);
    } else if length < 65536 {
        buffer.push(mt | 25);
        buffer.extend_from_slice(&(length as u16).to_be_bytes());
    } else if length < 4294967296 {
        buffer.push(mt | 26);
        buffer.extend_from_slice(&(length as u32).to_be_bytes());
    } else {
        buffer.push(mt | 27);
        buffer.extend_from_slice(&length.to_be_bytes());
    }
}

fn encode_cbor_tag(tag: u64, buffer: &mut Vec<u8>) {
    // CBOR major type 6 (tag)
    encode_cbor_length(6, tag, buffer);
}

// =============================================================================
// DAG-CBOR Decoding
// =============================================================================

fn decode_dag_cbor<R: std::io::Read>(reader: &mut R) -> Result<Ipld> {
    let mut first_byte = [0u8; 1];
    reader
        .read_exact(&mut first_byte)
        .map_err(|e| Error::Deserialization(format!("Failed to read CBOR: {}", e)))?;

    let major_type = first_byte[0] >> 5;
    let additional_info = first_byte[0] & 0x1f;

    match major_type {
        0 => {
            // Unsigned integer
            let value = decode_cbor_uint(additional_info, reader)?;
            Ok(Ipld::Integer(value as i128))
        }
        1 => {
            // Negative integer
            let value = decode_cbor_uint(additional_info, reader)?;
            Ok(Ipld::Integer(-1 - value as i128))
        }
        2 => {
            // Byte string
            let len = decode_cbor_uint(additional_info, reader)? as usize;
            let mut bytes = vec![0u8; len];
            reader
                .read_exact(&mut bytes)
                .map_err(|e| Error::Deserialization(format!("Failed to read bytes: {}", e)))?;
            Ok(Ipld::Bytes(bytes))
        }
        3 => {
            // Text string
            let len = decode_cbor_uint(additional_info, reader)? as usize;
            let mut bytes = vec![0u8; len];
            reader
                .read_exact(&mut bytes)
                .map_err(|e| Error::Deserialization(format!("Failed to read string: {}", e)))?;
            let s = String::from_utf8(bytes)
                .map_err(|e| Error::Deserialization(format!("Invalid UTF-8: {}", e)))?;
            Ok(Ipld::String(s))
        }
        4 => {
            // Array
            let len = decode_cbor_uint(additional_info, reader)? as usize;
            let mut list = Vec::with_capacity(len);
            for _ in 0..len {
                list.push(decode_dag_cbor(reader)?);
            }
            Ok(Ipld::List(list))
        }
        5 => {
            // Map
            let len = decode_cbor_uint(additional_info, reader)? as usize;
            let mut map = BTreeMap::new();
            for _ in 0..len {
                let key = decode_dag_cbor(reader)?;
                let key_str = match key {
                    Ipld::String(s) => s,
                    _ => {
                        return Err(Error::Deserialization(
                            "Map keys must be strings in IPLD".to_string(),
                        ))
                    }
                };
                let value = decode_dag_cbor(reader)?;
                map.insert(key_str, value);
            }
            Ok(Ipld::Map(map))
        }
        6 => {
            // Tag
            let tag = decode_cbor_uint(additional_info, reader)?;
            if tag == CID_TAG {
                // CID link
                let bytes_ipld = decode_dag_cbor(reader)?;
                match bytes_ipld {
                    Ipld::Bytes(mut bytes) => {
                        // Remove the multibase identity prefix (0x00)
                        if bytes.first() == Some(&0x00) {
                            bytes.remove(0);
                        }
                        let cid = Cid::try_from(&bytes[..])
                            .map_err(|e| Error::Deserialization(format!("Invalid CID: {}", e)))?;
                        Ok(Ipld::Link(crate::cid::SerializableCid(cid)))
                    }
                    _ => Err(Error::Deserialization(
                        "CID tag must wrap bytes".to_string(),
                    )),
                }
            } else {
                // Unknown tag, just decode the content
                decode_dag_cbor(reader)
            }
        }
        7 => {
            // Simple values and floats
            match additional_info {
                20 => Ok(Ipld::Bool(false)),
                21 => Ok(Ipld::Bool(true)),
                22 => Ok(Ipld::Null),
                25 => {
                    // 16-bit float (not commonly used, convert to f64)
                    let mut bytes = [0u8; 2];
                    reader.read_exact(&mut bytes).map_err(|e| {
                        Error::Deserialization(format!("Failed to read f16: {}", e))
                    })?;
                    let bits = u16::from_be_bytes(bytes);
                    Ok(Ipld::Float(f16_to_f64(bits)))
                }
                26 => {
                    // 32-bit float
                    let mut bytes = [0u8; 4];
                    reader.read_exact(&mut bytes).map_err(|e| {
                        Error::Deserialization(format!("Failed to read f32: {}", e))
                    })?;
                    let f = f32::from_be_bytes(bytes);
                    Ok(Ipld::Float(f as f64))
                }
                27 => {
                    // 64-bit float
                    let mut bytes = [0u8; 8];
                    reader.read_exact(&mut bytes).map_err(|e| {
                        Error::Deserialization(format!("Failed to read f64: {}", e))
                    })?;
                    let f = f64::from_be_bytes(bytes);
                    Ok(Ipld::Float(f))
                }
                _ => Err(Error::Deserialization(format!(
                    "Unknown simple value: {}",
                    additional_info
                ))),
            }
        }
        _ => Err(Error::Deserialization(format!(
            "Unknown CBOR major type: {}",
            major_type
        ))),
    }
}

fn decode_cbor_uint<R: std::io::Read>(additional_info: u8, reader: &mut R) -> Result<u64> {
    match additional_info {
        0..=23 => Ok(additional_info as u64),
        24 => {
            let mut buf = [0u8; 1];
            reader
                .read_exact(&mut buf)
                .map_err(|e| Error::Deserialization(format!("Failed to read u8: {}", e)))?;
            Ok(buf[0] as u64)
        }
        25 => {
            let mut buf = [0u8; 2];
            reader
                .read_exact(&mut buf)
                .map_err(|e| Error::Deserialization(format!("Failed to read u16: {}", e)))?;
            Ok(u16::from_be_bytes(buf) as u64)
        }
        26 => {
            let mut buf = [0u8; 4];
            reader
                .read_exact(&mut buf)
                .map_err(|e| Error::Deserialization(format!("Failed to read u32: {}", e)))?;
            Ok(u32::from_be_bytes(buf) as u64)
        }
        27 => {
            let mut buf = [0u8; 8];
            reader
                .read_exact(&mut buf)
                .map_err(|e| Error::Deserialization(format!("Failed to read u64: {}", e)))?;
            Ok(u64::from_be_bytes(buf))
        }
        _ => Err(Error::Deserialization(format!(
            "Invalid additional info for integer: {}",
            additional_info
        ))),
    }
}

/// Convert IEEE 754 half-precision (f16) to double-precision (f64)
fn f16_to_f64(bits: u16) -> f64 {
    let sign = ((bits >> 15) & 1) as u64;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x3ff) as u64;

    if exp == 0 {
        // Subnormal or zero
        if frac == 0 {
            f64::from_bits(sign << 63)
        } else {
            // Subnormal, normalize it
            let mut e = -14;
            let mut f = frac;
            while (f & 0x400) == 0 {
                f <<= 1;
                e -= 1;
            }
            let new_exp = (e + 1023) as u64;
            let new_frac = (f & 0x3ff) << 42;
            f64::from_bits((sign << 63) | (new_exp << 52) | new_frac)
        }
    } else if exp == 31 {
        // Infinity or NaN
        if frac == 0 {
            f64::from_bits((sign << 63) | (0x7ff << 52))
        } else {
            f64::from_bits((sign << 63) | (0x7ff << 52) | (frac << 42))
        }
    } else {
        // Normal number
        let new_exp = ((exp - 15) + 1023) as u64;
        let new_frac = frac << 42;
        f64::from_bits((sign << 63) | (new_exp << 52) | new_frac)
    }
}

// =============================================================================
// DAG-JSON Encoding/Decoding
// =============================================================================

fn ipld_to_dag_json(ipld: &Ipld) -> Result<serde_json::Value> {
    use serde_json::Value;

    match ipld {
        Ipld::Null => Ok(Value::Null),
        Ipld::Bool(b) => Ok(Value::Bool(*b)),
        Ipld::Integer(i) => {
            // JSON numbers have limited precision, use number if safe, string otherwise
            if *i >= i64::MIN as i128 && *i <= i64::MAX as i128 {
                Ok(Value::Number((*i as i64).into()))
            } else {
                // Large integers: encode as string
                Ok(Value::String(i.to_string()))
            }
        }
        Ipld::Float(f) => serde_json::Number::from_f64(*f)
            .map(Value::Number)
            .ok_or_else(|| Error::Serialization("Cannot encode NaN/Inf as JSON".to_string())),
        Ipld::String(s) => Ok(Value::String(s.clone())),
        Ipld::Bytes(b) => {
            // DAG-JSON encodes bytes as {"/": {"bytes": "<base64>"}}
            use multibase::Base;
            let encoded = multibase::encode(Base::Base64, b);
            // multibase::encode includes the base prefix, we just want the data
            let data = &encoded[1..]; // Skip the 'm' prefix for base64
            let mut inner = serde_json::Map::new();
            inner.insert("bytes".to_string(), Value::String(data.to_string()));
            let mut outer = serde_json::Map::new();
            outer.insert("/".to_string(), Value::Object(inner));
            Ok(Value::Object(outer))
        }
        Ipld::List(list) => {
            let arr: Result<Vec<Value>> = list.iter().map(ipld_to_dag_json).collect();
            Ok(Value::Array(arr?))
        }
        Ipld::Map(map) => {
            let mut obj = serde_json::Map::new();
            for (k, v) in map {
                obj.insert(k.clone(), ipld_to_dag_json(v)?);
            }
            Ok(Value::Object(obj))
        }
        Ipld::Link(cid) => {
            // DAG-JSON encodes CID links as {"/": "<cid-string>"}
            let mut obj = serde_json::Map::new();
            obj.insert("/".to_string(), Value::String(cid.0.to_string()));
            Ok(Value::Object(obj))
        }
    }
}

fn dag_json_to_ipld(value: &serde_json::Value) -> Result<Ipld> {
    use serde_json::Value;

    match value {
        Value::Null => Ok(Ipld::Null),
        Value::Bool(b) => Ok(Ipld::Bool(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(Ipld::Integer(i as i128))
            } else if let Some(f) = n.as_f64() {
                Ok(Ipld::Float(f))
            } else {
                Err(Error::Deserialization("Invalid number".to_string()))
            }
        }
        Value::String(s) => Ok(Ipld::String(s.clone())),
        Value::Array(arr) => {
            let list: Result<Vec<Ipld>> = arr.iter().map(dag_json_to_ipld).collect();
            Ok(Ipld::List(list?))
        }
        Value::Object(obj) => {
            // Check for special DAG-JSON encodings
            if let Some(slash_value) = obj.get("/") {
                if obj.len() == 1 {
                    // Could be a link {"/": "<cid>"} or bytes {"/": {"bytes": "<base64>"}}
                    match slash_value {
                        Value::String(cid_str) => {
                            // CID link
                            let cid: Cid = cid_str.parse().map_err(|e| {
                                Error::Deserialization(format!("Invalid CID: {}", e))
                            })?;
                            return Ok(Ipld::Link(crate::cid::SerializableCid(cid)));
                        }
                        Value::Object(inner) => {
                            if let Some(Value::String(bytes_str)) = inner.get("bytes") {
                                // Base64 encoded bytes
                                let decoded = multibase::decode(format!("m{}", bytes_str))
                                    .map_err(|e| {
                                        Error::Deserialization(format!(
                                            "Invalid base64 bytes: {}",
                                            e
                                        ))
                                    })?
                                    .1;
                                return Ok(Ipld::Bytes(decoded));
                            }
                        }
                        _ => {}
                    }
                }
            }

            // Regular map
            let mut map = BTreeMap::new();
            for (k, v) in obj {
                map.insert(k.clone(), dag_json_to_ipld(v)?);
            }
            Ok(Ipld::Map(map))
        }
    }
}

// =============================================================================
// Conversions
// =============================================================================

impl Serialize for Ipld {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Ipld::Null => serializer.serialize_none(),
            Ipld::Bool(b) => serializer.serialize_bool(*b),
            Ipld::Integer(i) => {
                // Serialize as i64 if within range, otherwise as i128
                if *i >= i64::MIN as i128 && *i <= i64::MAX as i128 {
                    serializer.serialize_i64(*i as i64)
                } else {
                    serializer.serialize_i128(*i)
                }
            }
            Ipld::Float(f) => serializer.serialize_f64(*f),
            Ipld::String(s) => serializer.serialize_str(s),
            Ipld::Bytes(b) => serializer.serialize_bytes(b),
            Ipld::List(list) => list.serialize(serializer),
            Ipld::Map(map) => map.serialize(serializer),
            Ipld::Link(cid) => cid.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for Ipld {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{MapAccess, SeqAccess, Visitor};

        struct IpldVisitor;

        impl<'de> Visitor<'de> for IpldVisitor {
            type Value = Ipld;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("an IPLD value")
            }

            fn visit_bool<E>(self, value: bool) -> std::result::Result<Ipld, E> {
                Ok(Ipld::Bool(value))
            }

            fn visit_i64<E>(self, value: i64) -> std::result::Result<Ipld, E> {
                Ok(Ipld::Integer(value as i128))
            }

            fn visit_i128<E>(self, value: i128) -> std::result::Result<Ipld, E> {
                Ok(Ipld::Integer(value))
            }

            fn visit_u64<E>(self, value: u64) -> std::result::Result<Ipld, E> {
                Ok(Ipld::Integer(value as i128))
            }

            fn visit_f64<E>(self, value: f64) -> std::result::Result<Ipld, E> {
                Ok(Ipld::Float(value))
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Ipld, E>
            where
                E: serde::de::Error,
            {
                Ok(Ipld::String(value.to_string()))
            }

            fn visit_string<E>(self, value: String) -> std::result::Result<Ipld, E> {
                Ok(Ipld::String(value))
            }

            fn visit_bytes<E>(self, value: &[u8]) -> std::result::Result<Ipld, E> {
                Ok(Ipld::Bytes(value.to_vec()))
            }

            fn visit_byte_buf<E>(self, value: Vec<u8>) -> std::result::Result<Ipld, E> {
                Ok(Ipld::Bytes(value))
            }

            fn visit_none<E>(self) -> std::result::Result<Ipld, E> {
                Ok(Ipld::Null)
            }

            fn visit_unit<E>(self) -> std::result::Result<Ipld, E> {
                Ok(Ipld::Null)
            }

            fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Ipld, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut list = Vec::new();
                while let Some(elem) = seq.next_element()? {
                    list.push(elem);
                }
                Ok(Ipld::List(list))
            }

            fn visit_map<A>(self, mut map: A) -> std::result::Result<Ipld, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut result = BTreeMap::new();
                while let Some((key, value)) = map.next_entry()? {
                    result.insert(key, value);
                }
                Ok(Ipld::Map(result))
            }
        }

        deserializer.deserialize_any(IpldVisitor)
    }
}

impl From<bool> for Ipld {
    fn from(b: bool) -> Self {
        Ipld::Bool(b)
    }
}

impl From<i64> for Ipld {
    fn from(i: i64) -> Self {
        Ipld::Integer(i as i128)
    }
}

impl From<i128> for Ipld {
    fn from(i: i128) -> Self {
        Ipld::Integer(i)
    }
}

impl From<u64> for Ipld {
    fn from(u: u64) -> Self {
        Ipld::Integer(u as i128)
    }
}

impl From<f64> for Ipld {
    fn from(f: f64) -> Self {
        Ipld::Float(f)
    }
}

impl From<String> for Ipld {
    fn from(s: String) -> Self {
        Ipld::String(s)
    }
}

impl From<&str> for Ipld {
    fn from(s: &str) -> Self {
        Ipld::String(s.to_string())
    }
}

impl From<Vec<u8>> for Ipld {
    fn from(bytes: Vec<u8>) -> Self {
        Ipld::Bytes(bytes)
    }
}

impl From<Cid> for Ipld {
    fn from(cid: Cid) -> Self {
        Ipld::Link(crate::cid::SerializableCid(cid))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dag_cbor_roundtrip_simple() {
        let values = vec![
            Ipld::Null,
            Ipld::Bool(true),
            Ipld::Bool(false),
            Ipld::Integer(0),
            Ipld::Integer(42),
            Ipld::Integer(-1),
            Ipld::Integer(-100),
            Ipld::Float(2.5),
            Ipld::String("hello".to_string()),
            Ipld::Bytes(vec![1, 2, 3]),
        ];

        for value in values {
            let encoded = value.to_dag_cbor().unwrap();
            let decoded = Ipld::from_dag_cbor(&encoded).unwrap();
            assert_eq!(value, decoded, "Failed roundtrip for {:?}", value);
        }
    }

    #[test]
    fn test_dag_cbor_roundtrip_complex() {
        let mut map = BTreeMap::new();
        map.insert("name".to_string(), Ipld::String("test".to_string()));
        map.insert("count".to_string(), Ipld::Integer(42));

        let value = Ipld::Map(map);
        let encoded = value.to_dag_cbor().unwrap();
        let decoded = Ipld::from_dag_cbor(&encoded).unwrap();
        assert_eq!(value, decoded);
    }

    #[test]
    fn test_dag_json_roundtrip() {
        let mut map = BTreeMap::new();
        map.insert("name".to_string(), Ipld::String("test".to_string()));
        map.insert("count".to_string(), Ipld::Integer(42));

        let value = Ipld::Map(map);
        let json = value.to_dag_json().unwrap();
        let decoded = Ipld::from_dag_json(&json).unwrap();
        assert_eq!(value, decoded);
    }

    #[test]
    fn test_dag_json_bytes_encoding() {
        let value = Ipld::Bytes(vec![1, 2, 3, 4, 5]);
        let json = value.to_dag_json().unwrap();
        // Should be encoded as {"/": {"bytes": "..."}}
        assert!(json.contains("\"/\""));
        assert!(json.contains("\"bytes\""));

        let decoded = Ipld::from_dag_json(&json).unwrap();
        assert_eq!(value, decoded);
    }
}
