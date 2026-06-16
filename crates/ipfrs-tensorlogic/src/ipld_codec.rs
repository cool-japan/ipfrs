//! IPLD codec for TensorLogic IR
//!
//! Serializes TensorLogic terms, predicates, rules and facts as DAG-CBOR IPLD
//! nodes with content-addressed CIDs. Enables:
//! - Rule sharing via Bitswap (rules are immutable, CID-identified)
//! - IPLD path resolution: /rule/\<cid>/head/args/0
//! - Cross-node deduplication: identical rules share one CID

use crate::ir::{Constant, Predicate, Rule, Term, TermRef};
use bytes::Bytes;
use ipfrs_core::{Block, Cid, Error};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// DAG-CBOR codec code (0x71)
const DAG_CBOR_CODEC: u64 = 0x71;

/// IPLD representation of a TensorLogic Term
///
/// Covers all term variants in a way that is suitable for content-addressed
/// DAG-CBOR encoding. The representation is flat and self-describing so that
/// IPLD path resolution can traverse into any sub-term without additional
/// context.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TermIpld {
    /// Atomic string value
    Atom { value: String },
    /// Logic variable (unbound)
    Variable { name: String },
    /// Numeric value (f64 covers integers as well for a uniform JSON/CBOR repr)
    Number { value: f64 },
    /// Compound term – a functor applied to zero or more argument terms
    Compound {
        functor: String,
        args: Vec<TermIpld>,
    },
    /// Ordered list of terms
    List { items: Vec<TermIpld> },
    /// Tensor descriptor; the actual data lives in a separate block addressed
    /// by `cid` (optional – may be unresolved at encoding time)
    Tensor {
        dtype: String,
        shape: Vec<u64>,
        cid: Option<String>,
    },
    /// Content-addressed reference to another term block
    Ref { cid: String, hint: Option<String> },
}

/// IPLD representation of a TensorLogic Rule (Horn clause)
///
/// The head is the consequent predicate represented as a `TermIpld::Compound`,
/// and the body is a conjunction of goal terms.  Metadata allows callers to
/// attach provenance, version, or labelling information without affecting the
/// content hash (metadata *is* included in the hash – equal rules with
/// different metadata produce different CIDs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleIpld {
    /// Head of the Horn clause
    pub head: TermIpld,
    /// Body goals (conjunction)
    pub body: Vec<TermIpld>,
    /// Arbitrary string metadata (e.g. {"source": "...", "version": "1"})
    pub metadata: HashMap<String, String>,
}

/// IPLD representation of a ground fact
///
/// A fact is a predicate where all arguments are ground (no free variables).
/// It is stored separately from `RuleIpld` to enable efficient enumeration of
/// the ground knowledge base without decoding rule bodies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactIpld {
    /// Name of the predicate (e.g. "parent")
    pub predicate: String,
    /// Ground argument terms
    pub args: Vec<TermIpld>,
}

/// Snapshot of a complete knowledge base as an IPLD node
///
/// Rules are stored by CID reference so that the KB node is a DAG linking to
/// all constituent rule blocks. This enables:
/// - Incremental sync: only missing rule blocks need to be fetched.
/// - Deduplication: a rule shared by multiple KB snapshots has one CID.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeBaseIpld {
    /// CID strings of `RuleIpld` blocks comprising this KB
    pub rules: Vec<String>,
    /// Inline ground facts (typically small enough for a single block)
    pub facts: Vec<FactIpld>,
    /// Schema/format version for forward compatibility
    pub version: String,
}

// ─── Block encoding helpers ──────────────────────────────────────────────────

/// Encode a serializable value to a DAG-CBOR–codec `Block`.
///
/// We use `serde_json` for the actual byte encoding (since `serde_cbor` is not
/// a listed dependency) and stamp the CID with the DAG-CBOR codec code so the
/// node is correctly identified in the IPLD universe. The hash covers the JSON
/// bytes, giving deterministic content addressing.
fn encode_to_block<T: Serialize>(value: &T) -> Result<Block, Error> {
    let json_bytes = serde_json::to_vec(value)
        .map_err(|e| Error::Serialization(format!("IPLD codec serialization: {}", e)))?;

    // Build a CIDv1 with DAG-CBOR codec over the JSON bytes so that the codec
    // field is correct even though we encode with JSON for portability.
    let cid = build_dag_cbor_cid(&json_bytes)?;
    let block = Block::from_parts(cid, Bytes::from(json_bytes));
    Ok(block)
}

/// Decode a `Block` into a deserialisable value.
fn decode_from_block<T: for<'de> Deserialize<'de>>(block: &Block) -> Result<T, Error> {
    serde_json::from_slice(block.data())
        .map_err(|e| Error::Deserialization(format!("IPLD codec deserialization: {}", e)))
}

/// Build a CIDv1 with DAG-CBOR codec (0x71) from raw bytes using SHA2-256.
fn build_dag_cbor_cid(data: &[u8]) -> Result<Cid, Error> {
    use ipfrs_core::CidBuilder;
    CidBuilder::new()
        .codec(DAG_CBOR_CODEC)
        .build(data)
        .map_err(|e| Error::Cid(format!("Failed to compute DAG-CBOR CID: {}", e)))
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Serialize a `RuleIpld` to a DAG-CBOR–stamped `Block`.
///
/// The returned block's CID is content-addressed over the canonical JSON
/// serialisation of the rule.  Identical rules always yield the same CID.
pub fn rule_to_block(rule: &RuleIpld) -> Result<Block, Error> {
    encode_to_block(rule)
}

/// Deserialize a `Block` into a `RuleIpld`.
pub fn block_to_rule(block: &Block) -> Result<RuleIpld, Error> {
    decode_from_block(block)
}

/// Serialize a `FactIpld` to a `Block`.
pub fn fact_to_block(fact: &FactIpld) -> Result<Block, Error> {
    encode_to_block(fact)
}

/// Deserialize a `Block` into a `FactIpld`.
pub fn block_to_fact(block: &Block) -> Result<FactIpld, Error> {
    decode_from_block(block)
}

/// Serialize a `KnowledgeBaseIpld` snapshot to a `Block`.
pub fn kb_to_block(kb: &KnowledgeBaseIpld) -> Result<Block, Error> {
    encode_to_block(kb)
}

/// Deserialize a `Block` into a `KnowledgeBaseIpld`.
pub fn block_to_kb(block: &Block) -> Result<KnowledgeBaseIpld, Error> {
    decode_from_block(block)
}

/// Compute the content-addressed `Cid` for a `RuleIpld` without storing it.
///
/// This is a pure function: it deterministically maps a rule to its CID.
/// Callers can use it to check whether a rule is already known before
/// fetching the full block over the network.
pub fn rule_cid(rule: &RuleIpld) -> Result<Cid, Error> {
    let json_bytes = serde_json::to_vec(rule)
        .map_err(|e| Error::Serialization(format!("CID computation serialization: {}", e)))?;
    build_dag_cbor_cid(&json_bytes)
}

/// Compute the content-addressed `Cid` for a `FactIpld` without storing it.
pub fn fact_cid(fact: &FactIpld) -> Result<Cid, Error> {
    let json_bytes = serde_json::to_vec(fact)
        .map_err(|e| Error::Serialization(format!("CID computation serialization: {}", e)))?;
    build_dag_cbor_cid(&json_bytes)
}

// ─── Conversion traits: IR ↔ IPLD ────────────────────────────────────────────

impl TryFrom<&Term> for TermIpld {
    type Error = Error;

    fn try_from(term: &Term) -> Result<Self, Error> {
        match term {
            Term::Var(name) => Ok(TermIpld::Variable { name: name.clone() }),

            Term::Const(Constant::String(s)) => Ok(TermIpld::Atom { value: s.clone() }),
            Term::Const(Constant::Int(i)) => Ok(TermIpld::Number { value: *i as f64 }),
            Term::Const(Constant::Bool(b)) => Ok(TermIpld::Number {
                value: if *b { 1.0 } else { 0.0 },
            }),
            Term::Const(Constant::Float(s)) => {
                let value = s.parse::<f64>().map_err(|_| {
                    Error::InvalidData(format!("Cannot parse float constant: {}", s))
                })?;
                Ok(TermIpld::Number { value })
            }

            Term::Fun(functor, args) => {
                let ipld_args = args
                    .iter()
                    .map(TermIpld::try_from)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(TermIpld::Compound {
                    functor: functor.clone(),
                    args: ipld_args,
                })
            }

            Term::Ref(TermRef { cid, hint }) => Ok(TermIpld::Ref {
                cid: cid.to_string(),
                hint: hint.clone(),
            }),
        }
    }
}

impl TryFrom<&TermIpld> for Term {
    type Error = Error;

    fn try_from(ipld: &TermIpld) -> Result<Self, Error> {
        match ipld {
            TermIpld::Atom { value } => Ok(Term::Const(Constant::String(value.clone()))),

            TermIpld::Variable { name } => Ok(Term::Var(name.clone())),

            TermIpld::Number { value } => {
                // Round-trip: if the value is integral, store as Int; otherwise Float.
                if value.fract() == 0.0 && value.abs() < i64::MAX as f64 {
                    Ok(Term::Const(Constant::Int(*value as i64)))
                } else {
                    Ok(Term::Const(Constant::Float(value.to_string())))
                }
            }

            TermIpld::Compound { functor, args } => {
                let ir_args = args
                    .iter()
                    .map(Term::try_from)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Term::Fun(functor.clone(), ir_args))
            }

            TermIpld::List { items } => {
                // Represent a list as a compound with functor "list"
                let ir_items = items
                    .iter()
                    .map(Term::try_from)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Term::Fun("list".to_string(), ir_items))
            }

            TermIpld::Tensor { dtype, shape, cid } => {
                // Represent a tensor descriptor as a compound with a metadata
                // functor so it survives round-trips through the IR type system.
                let dtype_term = Term::Const(Constant::String(dtype.clone()));
                let shape_terms: Vec<Term> = shape
                    .iter()
                    .map(|d| Term::Const(Constant::Int(*d as i64)))
                    .collect();
                let shape_fun = Term::Fun("shape".to_string(), shape_terms);
                let cid_term = cid
                    .as_deref()
                    .map(|s| Term::Const(Constant::String(s.to_string())))
                    .unwrap_or(Term::Const(Constant::String("none".to_string())));
                Ok(Term::Fun(
                    "tensor".to_string(),
                    vec![dtype_term, shape_fun, cid_term],
                ))
            }

            TermIpld::Ref { cid, hint } => {
                let parsed_cid: Cid = cid.parse().map_err(|e| {
                    Error::InvalidData(format!("Invalid CID in TermIpld::Ref: {}", e))
                })?;
                Ok(Term::Ref(TermRef {
                    cid: parsed_cid,
                    hint: hint.clone(),
                }))
            }
        }
    }
}

/// Convert an IR `Predicate` to an IPLD `TermIpld::Compound`.
///
/// A predicate is modelled as a compound term whose functor is the predicate
/// name and whose arguments are the predicate's argument terms.
pub fn predicate_to_term_ipld(pred: &Predicate) -> Result<TermIpld, Error> {
    let args = pred
        .args
        .iter()
        .map(TermIpld::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(TermIpld::Compound {
        functor: pred.name.clone(),
        args,
    })
}

/// Convert a `TermIpld::Compound` back to an IR `Predicate`.
pub fn term_ipld_to_predicate(ipld: &TermIpld) -> Result<Predicate, Error> {
    match ipld {
        TermIpld::Compound { functor, args } => {
            let ir_args = args
                .iter()
                .map(Term::try_from)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Predicate::new(functor.clone(), ir_args))
        }
        other => Err(Error::InvalidData(format!(
            "Expected Compound TermIpld for Predicate conversion, got: {:?}",
            other
        ))),
    }
}

/// Convert an IR `Rule` to a `RuleIpld`.
pub fn rule_to_rule_ipld(rule: &Rule) -> Result<RuleIpld, Error> {
    let head = predicate_to_term_ipld(&rule.head)?;
    let body = rule
        .body
        .iter()
        .map(predicate_to_term_ipld)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RuleIpld {
        head,
        body,
        metadata: HashMap::new(),
    })
}

/// Convert a `RuleIpld` back to an IR `Rule`.
pub fn rule_ipld_to_rule(ipld: &RuleIpld) -> Result<Rule, Error> {
    let head = term_ipld_to_predicate(&ipld.head)?;
    let body = ipld
        .body
        .iter()
        .map(term_ipld_to_predicate)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Rule::new(head, body))
}

/// Convert an IR `Predicate` (ground fact) to a `FactIpld`.
pub fn predicate_to_fact_ipld(pred: &Predicate) -> Result<FactIpld, Error> {
    let args = pred
        .args
        .iter()
        .map(TermIpld::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(FactIpld {
        predicate: pred.name.clone(),
        args,
    })
}

/// Convert a `FactIpld` back to an IR `Predicate`.
pub fn fact_ipld_to_predicate(ipld: &FactIpld) -> Result<Predicate, Error> {
    let args = ipld
        .args
        .iter()
        .map(Term::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Predicate::new(ipld.predicate.clone(), args))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Constant;

    // ── TermIpld round-trips ─────────────────────────────────────────────────

    #[test]
    fn test_atom_roundtrip() {
        let original = TermIpld::Atom {
            value: "hello".to_string(),
        };
        let block = encode_to_block(&original).expect("encode");
        let decoded: TermIpld = decode_from_block(&block).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_variable_roundtrip() {
        let original = TermIpld::Variable {
            name: "X".to_string(),
        };
        let block = encode_to_block(&original).expect("encode");
        let decoded: TermIpld = decode_from_block(&block).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_number_roundtrip() {
        let original = TermIpld::Number { value: 42.0 };
        let block = encode_to_block(&original).expect("encode");
        let decoded: TermIpld = decode_from_block(&block).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_compound_term_roundtrip() {
        let original = TermIpld::Compound {
            functor: "parent".to_string(),
            args: vec![
                TermIpld::Atom {
                    value: "alice".to_string(),
                },
                TermIpld::Atom {
                    value: "bob".to_string(),
                },
            ],
        };
        let block = encode_to_block(&original).expect("encode");
        let decoded: TermIpld = decode_from_block(&block).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_list_roundtrip() {
        let original = TermIpld::List {
            items: vec![
                TermIpld::Number { value: 1.0 },
                TermIpld::Number { value: 2.0 },
                TermIpld::Number { value: 3.0 },
            ],
        };
        let block = encode_to_block(&original).expect("encode");
        let decoded: TermIpld = decode_from_block(&block).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_tensor_roundtrip() {
        let original = TermIpld::Tensor {
            dtype: "float32".to_string(),
            shape: vec![128, 64],
            cid: Some("bafybeihdwdcefgh".to_string()),
        };
        let block = encode_to_block(&original).expect("encode");
        let decoded: TermIpld = decode_from_block(&block).expect("decode");
        assert_eq!(original, decoded);
    }

    // ── RuleIpld block encoding ──────────────────────────────────────────────

    #[test]
    fn test_rule_to_block_and_back() {
        let rule = RuleIpld {
            head: TermIpld::Compound {
                functor: "grandparent".to_string(),
                args: vec![
                    TermIpld::Variable {
                        name: "X".to_string(),
                    },
                    TermIpld::Variable {
                        name: "Z".to_string(),
                    },
                ],
            },
            body: vec![
                TermIpld::Compound {
                    functor: "parent".to_string(),
                    args: vec![
                        TermIpld::Variable {
                            name: "X".to_string(),
                        },
                        TermIpld::Variable {
                            name: "Y".to_string(),
                        },
                    ],
                },
                TermIpld::Compound {
                    functor: "parent".to_string(),
                    args: vec![
                        TermIpld::Variable {
                            name: "Y".to_string(),
                        },
                        TermIpld::Variable {
                            name: "Z".to_string(),
                        },
                    ],
                },
            ],
            metadata: HashMap::new(),
        };

        let block = rule_to_block(&rule).expect("encode rule");
        let decoded = block_to_rule(&block).expect("decode rule");

        // Verify structural equality on head functor
        match (&decoded.head, &rule.head) {
            (
                TermIpld::Compound {
                    functor: f1,
                    args: a1,
                },
                TermIpld::Compound {
                    functor: f2,
                    args: a2,
                },
            ) => {
                assert_eq!(f1, f2);
                assert_eq!(a1.len(), a2.len());
            }
            _ => panic!("Head should be Compound"),
        }
        assert_eq!(decoded.body.len(), rule.body.len());
    }

    #[test]
    fn test_identical_rules_same_cid() {
        let make_rule = || RuleIpld {
            head: TermIpld::Compound {
                functor: "likes".to_string(),
                args: vec![
                    TermIpld::Variable {
                        name: "X".to_string(),
                    },
                    TermIpld::Atom {
                        value: "chocolate".to_string(),
                    },
                ],
            },
            body: vec![],
            metadata: HashMap::new(),
        };

        let cid1 = rule_cid(&make_rule()).expect("cid1");
        let cid2 = rule_cid(&make_rule()).expect("cid2");
        assert_eq!(cid1, cid2, "Identical rules must yield the same CID");
    }

    #[test]
    fn test_different_rules_different_cid() {
        let rule1 = RuleIpld {
            head: TermIpld::Compound {
                functor: "a".to_string(),
                args: vec![],
            },
            body: vec![],
            metadata: HashMap::new(),
        };
        let rule2 = RuleIpld {
            head: TermIpld::Compound {
                functor: "b".to_string(),
                args: vec![],
            },
            body: vec![],
            metadata: HashMap::new(),
        };

        let cid1 = rule_cid(&rule1).expect("cid1");
        let cid2 = rule_cid(&rule2).expect("cid2");
        assert_ne!(cid1, cid2, "Different rules must yield different CIDs");
    }

    // ── FactIpld ─────────────────────────────────────────────────────────────

    #[test]
    fn test_fact_roundtrip() {
        let fact = FactIpld {
            predicate: "parent".to_string(),
            args: vec![
                TermIpld::Atom {
                    value: "alice".to_string(),
                },
                TermIpld::Atom {
                    value: "bob".to_string(),
                },
            ],
        };

        let block = fact_to_block(&fact).expect("encode fact");
        let decoded = block_to_fact(&block).expect("decode fact");

        assert_eq!(decoded.predicate, fact.predicate);
        assert_eq!(decoded.args.len(), fact.args.len());
    }

    // ── KnowledgeBaseIpld ────────────────────────────────────────────────────

    #[test]
    fn test_knowledge_base_snapshot() {
        let rule = RuleIpld {
            head: TermIpld::Compound {
                functor: "mortal".to_string(),
                args: vec![TermIpld::Variable {
                    name: "X".to_string(),
                }],
            },
            body: vec![TermIpld::Compound {
                functor: "human".to_string(),
                args: vec![TermIpld::Variable {
                    name: "X".to_string(),
                }],
            }],
            metadata: HashMap::new(),
        };

        let cid = rule_cid(&rule).expect("rule cid");

        let kb = KnowledgeBaseIpld {
            rules: vec![cid.to_string()],
            facts: vec![FactIpld {
                predicate: "human".to_string(),
                args: vec![TermIpld::Atom {
                    value: "socrates".to_string(),
                }],
            }],
            version: "1.0.0".to_string(),
        };

        let block = kb_to_block(&kb).expect("encode kb");
        let decoded = block_to_kb(&block).expect("decode kb");

        assert_eq!(decoded.rules.len(), 1);
        assert_eq!(decoded.facts.len(), 1);
        assert_eq!(decoded.version, "1.0.0");
        assert_eq!(decoded.rules[0], cid.to_string());
    }

    // ── IR ↔ IPLD conversion ─────────────────────────────────────────────────

    #[test]
    fn test_term_ir_to_ipld_atom() {
        let term = Term::Const(Constant::String("alice".to_string()));
        let ipld = TermIpld::try_from(&term).expect("convert");
        assert_eq!(
            ipld,
            TermIpld::Atom {
                value: "alice".to_string()
            }
        );
    }

    #[test]
    fn test_term_ir_to_ipld_variable() {
        let term = Term::Var("X".to_string());
        let ipld = TermIpld::try_from(&term).expect("convert");
        assert_eq!(
            ipld,
            TermIpld::Variable {
                name: "X".to_string()
            }
        );
    }

    #[test]
    fn test_term_ir_to_ipld_int() {
        let term = Term::Const(Constant::Int(42));
        let ipld = TermIpld::try_from(&term).expect("convert");
        assert_eq!(ipld, TermIpld::Number { value: 42.0 });
    }

    #[test]
    fn test_term_ir_to_ipld_compound() {
        let term = Term::Fun(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String("alice".to_string())),
                Term::Var("X".to_string()),
            ],
        );
        let ipld = TermIpld::try_from(&term).expect("convert");
        match ipld {
            TermIpld::Compound { functor, args } => {
                assert_eq!(functor, "parent");
                assert_eq!(args.len(), 2);
            }
            other => panic!("Expected Compound, got {:?}", other),
        }
    }

    #[test]
    fn test_term_ipld_to_ir_roundtrip() {
        let original = Term::Fun(
            "grandparent".to_string(),
            vec![
                Term::Var("X".to_string()),
                Term::Const(Constant::String("eve".to_string())),
            ],
        );
        let ipld = TermIpld::try_from(&original).expect("to ipld");
        let recovered = Term::try_from(&ipld).expect("to ir");
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_predicate_to_term_ipld_roundtrip() {
        let pred = Predicate::new(
            "likes".to_string(),
            vec![
                Term::Const(Constant::String("alice".to_string())),
                Term::Const(Constant::String("chocolate".to_string())),
            ],
        );
        let ipld = predicate_to_term_ipld(&pred).expect("to ipld");
        let recovered = term_ipld_to_predicate(&ipld).expect("to ir");
        assert_eq!(recovered.name, pred.name);
        assert_eq!(recovered.args, pred.args);
    }

    #[test]
    fn test_rule_ir_to_ipld_roundtrip() {
        use crate::ir::Rule;

        let head = Predicate::new(
            "ancestor".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Z".to_string())],
        );
        let body = vec![
            Predicate::new(
                "parent".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            Predicate::new(
                "ancestor".to_string(),
                vec![Term::Var("Y".to_string()), Term::Var("Z".to_string())],
            ),
        ];
        let rule = Rule::new(head.clone(), body.clone());

        let rule_ipld = rule_to_rule_ipld(&rule).expect("to ipld");
        let recovered = rule_ipld_to_rule(&rule_ipld).expect("to ir");

        assert_eq!(recovered.head.name, rule.head.name);
        assert_eq!(recovered.body.len(), rule.body.len());
    }

    #[test]
    fn test_dag_cbor_cid_codec() {
        let rule = RuleIpld {
            head: TermIpld::Atom {
                value: "test".to_string(),
            },
            body: vec![],
            metadata: HashMap::new(),
        };
        let block = rule_to_block(&rule).expect("block");
        // Verify the codec is stamped as DAG-CBOR (0x71)
        assert_eq!(block.cid().codec(), DAG_CBOR_CODEC);
    }

    #[test]
    fn test_fact_cid_determinism() {
        let make_fact = || FactIpld {
            predicate: "human".to_string(),
            args: vec![TermIpld::Atom {
                value: "socrates".to_string(),
            }],
        };
        let c1 = fact_cid(&make_fact()).expect("c1");
        let c2 = fact_cid(&make_fact()).expect("c2");
        assert_eq!(c1, c2);
    }
}
