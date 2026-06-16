//! IPLD path resolution for TensorLogic structures
//!
//! Supports deterministic traversal of JSON-serialized IPLD blocks along paths
//! like:
//!
//! ```text
//!   /rule/<cid>/head/functor
//!   /rule/<cid>/head/args/0
//!   /rule/<cid>/body/0/functor
//!   /fact/<cid>/predicate
//!   /fact/<cid>/args/1
//! ```
//!
//! The resolver operates on the raw block bytes produced by
//! [`crate::ipld_codec`], which encodes blocks as JSON internally (even though
//! the CID carries the DAG-CBOR codec stamp).  Each path segment is applied
//! iteratively: numeric segments index into arrays; string segments key into
//! objects.

use std::collections::HashMap;

/// Resolved value at an IPLD path
#[derive(Debug, Clone, PartialEq)]
pub enum IpldPathValue {
    /// A string value (includes functor names, variable names, atom values, …)
    String(String),
    /// A numeric value (f64 covers integers and floats uniformly)
    Number(f64),
    /// An ordered list of values
    Array(Vec<IpldPathValue>),
    /// A map of string keys to values
    Object(HashMap<String, IpldPathValue>),
    /// Null / absent
    Null,
}

impl IpldPathValue {
    /// Convenience: return inner string if this is `IpldPathValue::String`
    pub fn as_str(&self) -> Option<&str> {
        match self {
            IpldPathValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Convenience: return inner f64 if this is `IpldPathValue::Number`
    pub fn as_number(&self) -> Option<f64> {
        match self {
            IpldPathValue::Number(n) => Some(*n),
            _ => None,
        }
    }
}

/// Errors that can occur during IPLD path resolution
#[derive(Debug, thiserror::Error)]
pub enum PathError {
    #[error("Invalid path: {0}")]
    InvalidPath(String),
    #[error("Path segment not found: {0}")]
    NotFound(String),
    #[error("Type mismatch at {0}: expected {1}")]
    TypeMismatch(String, String),
    #[error("Index out of bounds: {0}")]
    IndexOutOfBounds(usize),
    #[error("Deserialization error: {0}")]
    Deserialization(String),
}

/// IPLD path resolver for TensorLogic block data
///
/// All methods are stateless pure functions that operate on raw block bytes.
pub struct IpldPathResolver;

impl IpldPathResolver {
    /// Resolve an IPLD path against a stored rule block.
    ///
    /// `block_data` must be the raw bytes of a block encoded by
    /// [`crate::ipld_codec::rule_to_block`].
    ///
    /// `path` is the *full* path string, e.g.
    /// `"/rule/bafkrei.../head/args/0"`.  The first two segments (`rule` and
    /// the CID) are skipped; traversal begins at the third segment.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let val = IpldPathResolver::resolve_rule_path(block_data, "/rule/bafk.../head/functor")?;
    /// assert_eq!(val, IpldPathValue::String("grandparent".to_string()));
    /// ```
    pub fn resolve_rule_path(block_data: &[u8], path: &str) -> Result<IpldPathValue, PathError> {
        let segments = Self::parse_path(path);

        // Expect at least /rule/<cid>/…  (i.e. ≥ 3 segments after splitting)
        // segments[0] == "rule", segments[1] == <cid>, segments[2..] == traversal
        if segments.len() < 2 {
            return Err(PathError::InvalidPath(format!(
                "Rule path must start with /rule/<cid>/…; got: {}",
                path
            )));
        }
        if segments[0] != "rule" {
            return Err(PathError::InvalidPath(format!(
                "Expected 'rule' as first segment, got '{}'",
                segments[0]
            )));
        }

        let root: serde_json::Value = serde_json::from_slice(block_data)
            .map_err(|e| PathError::Deserialization(e.to_string()))?;

        // Traverse segments[2..] (skip "rule" and the CID)
        let traversal = &segments[2..];
        Self::traverse(&root, traversal)
    }

    /// Resolve a path against a stored fact block.
    ///
    /// `block_data` must be the raw bytes of a block encoded by
    /// [`crate::ipld_codec::fact_to_block`].
    ///
    /// `path` is the full path, e.g. `"/fact/bafk.../predicate"`.  The first
    /// two segments are skipped.
    pub fn resolve_fact_path(block_data: &[u8], path: &str) -> Result<IpldPathValue, PathError> {
        let segments = Self::parse_path(path);

        if segments.len() < 2 {
            return Err(PathError::InvalidPath(format!(
                "Fact path must start with /fact/<cid>/…; got: {}",
                path
            )));
        }
        if segments[0] != "fact" {
            return Err(PathError::InvalidPath(format!(
                "Expected 'fact' as first segment, got '{}'",
                segments[0]
            )));
        }

        let root: serde_json::Value = serde_json::from_slice(block_data)
            .map_err(|e| PathError::Deserialization(e.to_string()))?;

        let traversal = &segments[2..];
        Self::traverse(&root, traversal)
    }

    /// Parse a path string into non-empty segments by splitting on `/`.
    ///
    /// Leading and trailing slashes are stripped; consecutive slashes are
    /// treated as a single separator.
    pub fn parse_path(path: &str) -> Vec<String> {
        path.split('/')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Recursively traverse `segments` starting from `current`.
    fn traverse(
        current: &serde_json::Value,
        segments: &[String],
    ) -> Result<IpldPathValue, PathError> {
        if segments.is_empty() {
            return Self::json_to_ipld(current);
        }

        let seg = &segments[0];
        let rest = &segments[1..];

        match current {
            serde_json::Value::Object(map) => {
                let child = map.get(seg.as_str()).ok_or_else(|| {
                    PathError::NotFound(format!("Key '{}' not found in object", seg))
                })?;
                Self::traverse(child, rest)
            }
            serde_json::Value::Array(arr) => {
                let idx: usize = seg.parse().map_err(|_| {
                    PathError::TypeMismatch(
                        seg.clone(),
                        "numeric index for array traversal".to_string(),
                    )
                })?;
                let child = arr.get(idx).ok_or(PathError::IndexOutOfBounds(idx))?;
                Self::traverse(child, rest)
            }
            other => Err(PathError::TypeMismatch(
                seg.clone(),
                format!(
                    "object or array (cannot descend into {})",
                    Self::json_type_name(other)
                ),
            )),
        }
    }

    /// Convert a `serde_json::Value` leaf into an [`IpldPathValue`].
    fn json_to_ipld(value: &serde_json::Value) -> Result<IpldPathValue, PathError> {
        match value {
            serde_json::Value::Null => Ok(IpldPathValue::Null),
            serde_json::Value::Bool(b) => Ok(IpldPathValue::Number(if *b { 1.0 } else { 0.0 })),
            serde_json::Value::Number(n) => {
                let f = n.as_f64().ok_or_else(|| {
                    PathError::Deserialization(format!("Cannot convert number {} to f64", n))
                })?;
                Ok(IpldPathValue::Number(f))
            }
            serde_json::Value::String(s) => Ok(IpldPathValue::String(s.clone())),
            serde_json::Value::Array(arr) => {
                let items = arr
                    .iter()
                    .map(Self::json_to_ipld)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(IpldPathValue::Array(items))
            }
            serde_json::Value::Object(map) => {
                let kv = map
                    .iter()
                    .map(|(k, v)| Self::json_to_ipld(v).map(|ipld| (k.clone(), ipld)))
                    .collect::<Result<HashMap<_, _>, _>>()?;
                Ok(IpldPathValue::Object(kv))
            }
        }
    }

    /// Return a human-readable type name for a JSON value (for error messages).
    fn json_type_name(v: &serde_json::Value) -> &'static str {
        match v {
            serde_json::Value::Null => "null",
            serde_json::Value::Bool(_) => "bool",
            serde_json::Value::Number(_) => "number",
            serde_json::Value::String(_) => "string",
            serde_json::Value::Array(_) => "array",
            serde_json::Value::Object(_) => "object",
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipld_codec::{
        fact_to_block, predicate_to_fact_ipld, rule_to_block, rule_to_rule_ipld,
    };
    use crate::ir::{Constant, Predicate, Rule, Term};

    /// Build a rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z)
    fn grandparent_rule() -> Rule {
        let head = Predicate::new(
            "grandparent".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Z".to_string())],
        );
        let body = vec![
            Predicate::new(
                "parent".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            Predicate::new(
                "parent".to_string(),
                vec![Term::Var("Y".to_string()), Term::Var("Z".to_string())],
            ),
        ];
        Rule::new(head, body)
    }

    /// Encode a rule as block bytes
    fn rule_bytes(rule: &Rule) -> Vec<u8> {
        let ipld = rule_to_rule_ipld(rule).expect("rule_to_rule_ipld");
        let block = rule_to_block(&ipld).expect("rule_to_block");
        block.data().to_vec()
    }

    /// Encode a fact as block bytes
    fn fact_bytes(pred: &Predicate) -> Vec<u8> {
        let ipld = predicate_to_fact_ipld(pred).expect("predicate_to_fact_ipld");
        let block = fact_to_block(&ipld).expect("fact_to_block");
        block.data().to_vec()
    }

    #[test]
    fn test_resolve_head_functor() {
        let rule = grandparent_rule();
        let data = rule_bytes(&rule);
        let cid = "bafktest000";

        let val =
            IpldPathResolver::resolve_rule_path(&data, &format!("/rule/{}/head/functor", cid))
                .expect("resolve head/functor");

        assert_eq!(val, IpldPathValue::String("grandparent".to_string()));
    }

    #[test]
    fn test_resolve_head_arg_by_index() {
        let rule = grandparent_rule();
        let data = rule_bytes(&rule);
        let cid = "bafktest001";

        // head args are Variable nodes; args[0] is X, args[1] is Z
        let val0 =
            IpldPathResolver::resolve_rule_path(&data, &format!("/rule/{}/head/args/0/name", cid))
                .expect("head/args/0/name");

        assert_eq!(val0, IpldPathValue::String("X".to_string()));

        let val1 =
            IpldPathResolver::resolve_rule_path(&data, &format!("/rule/{}/head/args/1/name", cid))
                .expect("head/args/1/name");

        assert_eq!(val1, IpldPathValue::String("Z".to_string()));
    }

    #[test]
    fn test_resolve_body_goal() {
        let rule = grandparent_rule();
        let data = rule_bytes(&rule);
        let cid = "bafktest002";

        // body[0] is parent(X, Y) – functor is "parent"
        let val =
            IpldPathResolver::resolve_rule_path(&data, &format!("/rule/{}/body/0/functor", cid))
                .expect("body/0/functor");

        assert_eq!(val, IpldPathValue::String("parent".to_string()));

        // body[1] is parent(Y, Z)
        let val2 =
            IpldPathResolver::resolve_rule_path(&data, &format!("/rule/{}/body/1/functor", cid))
                .expect("body/1/functor");

        assert_eq!(val2, IpldPathValue::String("parent".to_string()));
    }

    #[test]
    fn test_invalid_path_error_wrong_prefix() {
        let rule = grandparent_rule();
        let data = rule_bytes(&rule);

        let err = IpldPathResolver::resolve_rule_path(&data, "/fact/bafk/head").unwrap_err();
        assert!(
            matches!(err, PathError::InvalidPath(_)),
            "Expected InvalidPath, got {:?}",
            err
        );
    }

    #[test]
    fn test_invalid_path_too_short() {
        let rule = grandparent_rule();
        let data = rule_bytes(&rule);

        let err = IpldPathResolver::resolve_rule_path(&data, "/rule").unwrap_err();
        assert!(
            matches!(err, PathError::InvalidPath(_)),
            "Expected InvalidPath for short path, got {:?}",
            err
        );
    }

    #[test]
    fn test_index_out_of_bounds() {
        let rule = grandparent_rule();
        let data = rule_bytes(&rule);
        let cid = "bafktest003";

        // head has 2 args; index 99 must fail
        let err =
            IpldPathResolver::resolve_rule_path(&data, &format!("/rule/{}/head/args/99", cid))
                .unwrap_err();

        assert!(
            matches!(err, PathError::IndexOutOfBounds(99)),
            "Expected IndexOutOfBounds(99), got {:?}",
            err
        );
    }

    #[test]
    fn test_key_not_found() {
        let rule = grandparent_rule();
        let data = rule_bytes(&rule);
        let cid = "bafktest004";

        let err = IpldPathResolver::resolve_rule_path(
            &data,
            &format!("/rule/{}/head/nonexistent_key", cid),
        )
        .unwrap_err();

        assert!(
            matches!(err, PathError::NotFound(_)),
            "Expected NotFound, got {:?}",
            err
        );
    }

    #[test]
    fn test_fact_path_predicate() {
        let pred = Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String("alice".to_string())),
                Term::Const(Constant::String("bob".to_string())),
            ],
        );
        let data = fact_bytes(&pred);
        let cid = "bafktest005";

        let val = IpldPathResolver::resolve_fact_path(&data, &format!("/fact/{}/predicate", cid))
            .expect("fact/predicate");

        assert_eq!(val, IpldPathValue::String("parent".to_string()));
    }

    #[test]
    fn test_fact_path_arg_by_index() {
        let pred = Predicate::new(
            "likes".to_string(),
            vec![
                Term::Const(Constant::String("alice".to_string())),
                Term::Const(Constant::String("chocolate".to_string())),
            ],
        );
        let data = fact_bytes(&pred);
        let cid = "bafktest006";

        // args[1] is "chocolate" – stored as TermIpld::Atom { value: "chocolate" }
        let val =
            IpldPathResolver::resolve_fact_path(&data, &format!("/fact/{}/args/1/value", cid))
                .expect("fact/args/1/value");

        assert_eq!(val, IpldPathValue::String("chocolate".to_string()));
    }

    #[test]
    fn test_parse_path_strips_leading_slash() {
        let segments = IpldPathResolver::parse_path("/rule/bafk/head/args/0");
        assert_eq!(segments, vec!["rule", "bafk", "head", "args", "0"]);
    }

    #[test]
    fn test_parse_path_no_leading_slash() {
        let segments = IpldPathResolver::parse_path("rule/bafk/head");
        assert_eq!(segments, vec!["rule", "bafk", "head"]);
    }

    #[test]
    fn test_rule_body_arg_resolution() {
        let rule = grandparent_rule();
        let data = rule_bytes(&rule);
        let cid = "bafktest007";

        // body[0]/args[0] == Variable {name: "X"}
        let val = IpldPathResolver::resolve_rule_path(
            &data,
            &format!("/rule/{}/body/0/args/0/name", cid),
        )
        .expect("body/0/args/0/name");

        assert_eq!(val, IpldPathValue::String("X".to_string()));
    }
}
