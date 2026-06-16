//! Tensor Provenance Tracker
//!
//! Tracks the full provenance of tensor values and inference results, recording
//! which rules fired, which facts were used, and how values were derived.
//!
//! # Overview
//!
//! [`TensorProvenanceTracker`] accumulates [`ProvenanceRecord`] entries keyed by
//! both `record_id` (globally unique) and `tensor_id` (the tensor/value being
//! described). Each record carries a [`ProvenanceKind`] that explains how the
//! tensor value came to exist, a confidence score, and the simulation tick at
//! which the record was created.
//!
//! [`ProvenanceChain`] presents all records for a single tensor ordered
//! oldest-first, and exposes convenience accessors (root, tip, avg_confidence).
//!
//! [`ProvenanceStats`] gives high-level counters across the whole tracker.
//!
//! # Examples
//!
//! ```
//! use ipfrs_tensorlogic::provenance_tracker::{
//!     ProvenanceKind, TensorProvenanceTracker,
//! };
//!
//! let mut tracker = TensorProvenanceTracker::new();
//!
//! let id = tracker.record(1, ProvenanceKind::FactAssertion, 1.0, 0);
//! assert_eq!(id, 0);
//!
//! let chain = tracker.chain_for(1);
//! assert_eq!(chain.len(), 1);
//! assert!((chain.avg_confidence() - 1.0).abs() < f64::EPSILON);
//! ```

use std::collections::HashMap;

// ─── ProvenanceKind ──────────────────────────────────────────────────────────

/// Describes the origin or derivation mechanism for a tensor value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProvenanceKind {
    /// Value was derived by firing a named rule.
    RuleFired { rule_id: u64 },

    /// Value was directly asserted as a fact (ground truth).
    FactAssertion,

    /// Value is the result of a numbered inference step.
    InferenceStep { step: u32 },

    /// Value was imported from an external source (e.g., a dataset file).
    ExternalInput { source: String },

    /// Value was computed from one or more other tensor values.
    Derived { parent_ids: Vec<u64> },
}

// ─── ProvenanceRecord ────────────────────────────────────────────────────────

/// A single record describing how one tensor value was produced.
#[derive(Clone, Debug)]
pub struct ProvenanceRecord {
    /// Globally unique identifier for this record.
    pub record_id: u64,

    /// Identifier of the tensor/value this record describes.
    pub tensor_id: u64,

    /// How this value was derived.
    pub kind: ProvenanceKind,

    /// Simulation/inference tick at which this record was created.
    pub created_at_tick: u64,

    /// Confidence in the value, in [0.0, 1.0].
    pub confidence: f64,
}

impl ProvenanceRecord {
    /// Returns `true` when the value originates entirely from outside the
    /// inference engine (i.e., a fact assertion or external import).
    pub fn is_externally_sourced(&self) -> bool {
        matches!(
            self.kind,
            ProvenanceKind::FactAssertion | ProvenanceKind::ExternalInput { .. }
        )
    }
}

// ─── ProvenanceChain ─────────────────────────────────────────────────────────

/// Ordered sequence of provenance records for a single tensor, oldest first.
#[derive(Clone, Debug, Default)]
pub struct ProvenanceChain {
    /// Records ordered oldest → newest (insertion order).
    pub chain: Vec<ProvenanceRecord>,
}

impl ProvenanceChain {
    /// Returns a reference to the oldest record in the chain, or `None` if
    /// the chain is empty.
    pub fn root(&self) -> Option<&ProvenanceRecord> {
        self.chain.first()
    }

    /// Returns a reference to the newest record in the chain, or `None` if
    /// the chain is empty.
    pub fn tip(&self) -> Option<&ProvenanceRecord> {
        self.chain.last()
    }

    /// Returns the arithmetic mean of all record confidence values, or `0.0`
    /// when the chain is empty.
    pub fn avg_confidence(&self) -> f64 {
        if self.chain.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.chain.iter().map(|r| r.confidence).sum();
        sum / self.chain.len() as f64
    }

    /// Returns the number of records in the chain.
    pub fn len(&self) -> usize {
        self.chain.len()
    }

    /// Returns `true` when the chain contains no records.
    pub fn is_empty(&self) -> bool {
        self.chain.is_empty()
    }
}

// ─── ProvenanceStats ─────────────────────────────────────────────────────────

/// High-level statistics across the entire [`TensorProvenanceTracker`].
#[derive(Clone, Debug)]
pub struct ProvenanceStats {
    /// Total number of provenance records stored.
    pub total_records: usize,

    /// Number of distinct tensor IDs that have at least one record.
    pub unique_tensors: usize,

    /// Number of records whose kind is [`ProvenanceKind::FactAssertion`].
    pub fact_count: usize,

    /// Number of records whose kind is [`ProvenanceKind::RuleFired`].
    pub rule_fired_count: usize,

    /// Arithmetic mean of `confidence` across all records (0.0 if no records).
    pub avg_confidence: f64,
}

// ─── TensorProvenanceTracker ─────────────────────────────────────────────────

/// Tracks the full provenance of tensor values and inference results.
///
/// Records are keyed by a globally unique `record_id` and can also be looked up
/// by `tensor_id` (returning all records for that tensor in insertion order).
#[derive(Debug, Default)]
pub struct TensorProvenanceTracker {
    /// All records, keyed by `record_id`.
    pub records: HashMap<u64, ProvenanceRecord>,

    /// Maps each `tensor_id` to the ordered list of `record_id`s for that tensor.
    pub tensor_records: HashMap<u64, Vec<u64>>,

    /// Monotonically increasing counter used to assign record IDs.
    pub next_record_id: u64,
}

impl TensorProvenanceTracker {
    /// Creates a new, empty tracker.
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
            tensor_records: HashMap::new(),
            next_record_id: 0,
        }
    }

    /// Records a new provenance entry for `tensor_id`.
    ///
    /// Returns the newly assigned `record_id`.
    pub fn record(
        &mut self,
        tensor_id: u64,
        kind: ProvenanceKind,
        confidence: f64,
        tick: u64,
    ) -> u64 {
        let record_id = self.next_record_id;
        self.next_record_id += 1;

        let entry = ProvenanceRecord {
            record_id,
            tensor_id,
            kind,
            created_at_tick: tick,
            confidence,
        };

        self.records.insert(record_id, entry);
        self.tensor_records
            .entry(tensor_id)
            .or_default()
            .push(record_id);

        record_id
    }

    /// Returns the [`ProvenanceChain`] for `tensor_id` in insertion order.
    ///
    /// Returns an empty chain when `tensor_id` is not known.
    pub fn chain_for(&self, tensor_id: u64) -> ProvenanceChain {
        let chain = self
            .tensor_records
            .get(&tensor_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.records.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default();
        ProvenanceChain { chain }
    }

    /// Returns a reference to the record with the given `record_id`, or `None`.
    pub fn get_record(&self, record_id: u64) -> Option<&ProvenanceRecord> {
        self.records.get(&record_id)
    }

    /// Returns all records for `tensor_id` in insertion order.
    pub fn records_for_tensor(&self, tensor_id: u64) -> Vec<&ProvenanceRecord> {
        self.tensor_records
            .get(&tensor_id)
            .map(|ids| ids.iter().filter_map(|id| self.records.get(id)).collect())
            .unwrap_or_default()
    }

    /// Returns a sorted list of tensor IDs for which at least one record is
    /// externally sourced (i.e., [`ProvenanceRecord::is_externally_sourced`]
    /// returns `true`).
    pub fn externally_sourced_tensors(&self) -> Vec<u64> {
        let mut result: Vec<u64> = self
            .tensor_records
            .iter()
            .filter(|(_tid, ids)| {
                ids.iter()
                    .filter_map(|id| self.records.get(id))
                    .any(|r| r.is_externally_sourced())
            })
            .map(|(tid, _)| *tid)
            .collect();
        result.sort_unstable();
        result
    }

    /// Removes all records associated with `tensor_id`.
    ///
    /// Returns `true` on success, `false` if `tensor_id` was not found.
    pub fn delete_tensor(&mut self, tensor_id: u64) -> bool {
        match self.tensor_records.remove(&tensor_id) {
            None => false,
            Some(ids) => {
                for id in ids {
                    self.records.remove(&id);
                }
                true
            }
        }
    }

    /// Returns aggregate statistics across all stored records.
    pub fn stats(&self) -> ProvenanceStats {
        let total_records = self.records.len();
        let unique_tensors = self.tensor_records.len();

        let mut fact_count = 0usize;
        let mut rule_fired_count = 0usize;
        let mut confidence_sum = 0.0_f64;

        for record in self.records.values() {
            match &record.kind {
                ProvenanceKind::FactAssertion => fact_count += 1,
                ProvenanceKind::RuleFired { .. } => rule_fired_count += 1,
                _ => {}
            }
            confidence_sum += record.confidence;
        }

        let avg_confidence = if total_records == 0 {
            0.0
        } else {
            confidence_sum / total_records as f64
        };

        ProvenanceStats {
            total_records,
            unique_tensors,
            fact_count,
            rule_fired_count,
            avg_confidence,
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── construction ──────────────────────────────────────────────────────────

    #[test]
    fn new_starts_empty() {
        let tracker = TensorProvenanceTracker::new();
        assert!(tracker.records.is_empty());
        assert!(tracker.tensor_records.is_empty());
        assert_eq!(tracker.next_record_id, 0);
    }

    // ── record ────────────────────────────────────────────────────────────────

    #[test]
    fn record_stores_and_returns_id() {
        let mut t = TensorProvenanceTracker::new();
        let id = t.record(42, ProvenanceKind::FactAssertion, 1.0, 0);
        assert_eq!(id, 0);
        assert!(t.records.contains_key(&0));
    }

    #[test]
    fn record_increments_id() {
        let mut t = TensorProvenanceTracker::new();
        let a = t.record(1, ProvenanceKind::FactAssertion, 1.0, 0);
        let b = t.record(2, ProvenanceKind::FactAssertion, 1.0, 1);
        assert_eq!(a, 0);
        assert_eq!(b, 1);
    }

    #[test]
    fn record_appends_to_tensor_records() {
        let mut t = TensorProvenanceTracker::new();
        t.record(10, ProvenanceKind::FactAssertion, 1.0, 0);
        assert_eq!(t.tensor_records[&10], vec![0u64]);
    }

    #[test]
    fn multiple_records_for_same_tensor() {
        let mut t = TensorProvenanceTracker::new();
        t.record(5, ProvenanceKind::FactAssertion, 0.8, 0);
        t.record(5, ProvenanceKind::RuleFired { rule_id: 99 }, 0.9, 1);
        assert_eq!(t.tensor_records[&5], vec![0u64, 1u64]);
        assert_eq!(t.records.len(), 2);
    }

    // ── chain_for ─────────────────────────────────────────────────────────────

    #[test]
    fn chain_for_returns_in_insertion_order() {
        let mut t = TensorProvenanceTracker::new();
        t.record(7, ProvenanceKind::FactAssertion, 0.5, 0);
        t.record(7, ProvenanceKind::InferenceStep { step: 1 }, 0.7, 1);
        t.record(7, ProvenanceKind::RuleFired { rule_id: 3 }, 0.9, 2);
        let chain = t.chain_for(7);
        assert_eq!(chain.len(), 3);
        assert_eq!(chain.chain[0].record_id, 0);
        assert_eq!(chain.chain[1].record_id, 1);
        assert_eq!(chain.chain[2].record_id, 2);
    }

    #[test]
    fn chain_for_empty_chain_for_unknown_tensor() {
        let t = TensorProvenanceTracker::new();
        let chain = t.chain_for(999);
        assert!(chain.is_empty());
    }

    // ── ProvenanceChain ───────────────────────────────────────────────────────

    #[test]
    fn provenance_chain_root_and_tip() {
        let mut t = TensorProvenanceTracker::new();
        t.record(3, ProvenanceKind::FactAssertion, 0.4, 0);
        t.record(3, ProvenanceKind::RuleFired { rule_id: 1 }, 0.6, 1);
        let chain = t.chain_for(3);
        assert_eq!(chain.root().map(|r| r.record_id), Some(0));
        assert_eq!(chain.tip().map(|r| r.record_id), Some(1));
    }

    #[test]
    fn provenance_chain_root_none_when_empty() {
        let chain = ProvenanceChain::default();
        assert!(chain.root().is_none());
        assert!(chain.tip().is_none());
    }

    #[test]
    fn provenance_chain_avg_confidence() {
        let mut t = TensorProvenanceTracker::new();
        t.record(8, ProvenanceKind::FactAssertion, 0.4, 0);
        t.record(8, ProvenanceKind::InferenceStep { step: 2 }, 0.6, 1);
        let chain = t.chain_for(8);
        let avg = chain.avg_confidence();
        assert!((avg - 0.5).abs() < 1e-10);
    }

    #[test]
    fn provenance_chain_avg_confidence_empty() {
        let chain = ProvenanceChain::default();
        assert_eq!(chain.avg_confidence(), 0.0);
    }

    #[test]
    fn provenance_chain_len_and_is_empty() {
        let mut t = TensorProvenanceTracker::new();
        let empty = t.chain_for(0);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());

        t.record(0, ProvenanceKind::FactAssertion, 1.0, 0);
        let non_empty = t.chain_for(0);
        assert_eq!(non_empty.len(), 1);
        assert!(!non_empty.is_empty());
    }

    // ── get_record ────────────────────────────────────────────────────────────

    #[test]
    fn get_record_some() {
        let mut t = TensorProvenanceTracker::new();
        let id = t.record(1, ProvenanceKind::FactAssertion, 1.0, 5);
        let rec = t.get_record(id);
        assert!(rec.is_some());
        assert_eq!(rec.expect("test: should succeed").tensor_id, 1);
        assert_eq!(rec.expect("test: should succeed").created_at_tick, 5);
    }

    #[test]
    fn get_record_none_for_unknown() {
        let t = TensorProvenanceTracker::new();
        assert!(t.get_record(9999).is_none());
    }

    // ── records_for_tensor ────────────────────────────────────────────────────

    #[test]
    fn records_for_tensor_insertion_order() {
        let mut t = TensorProvenanceTracker::new();
        t.record(20, ProvenanceKind::FactAssertion, 0.3, 0);
        t.record(20, ProvenanceKind::RuleFired { rule_id: 7 }, 0.7, 1);
        let recs = t.records_for_tensor(20);
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].record_id, 0);
        assert_eq!(recs[1].record_id, 1);
    }

    #[test]
    fn records_for_tensor_empty_for_unknown() {
        let t = TensorProvenanceTracker::new();
        assert!(t.records_for_tensor(404).is_empty());
    }

    // ── is_externally_sourced ─────────────────────────────────────────────────

    #[test]
    fn is_externally_sourced_fact_assertion_true() {
        let rec = ProvenanceRecord {
            record_id: 0,
            tensor_id: 0,
            kind: ProvenanceKind::FactAssertion,
            created_at_tick: 0,
            confidence: 1.0,
        };
        assert!(rec.is_externally_sourced());
    }

    #[test]
    fn is_externally_sourced_external_input_true() {
        let rec = ProvenanceRecord {
            record_id: 0,
            tensor_id: 0,
            kind: ProvenanceKind::ExternalInput {
                source: "file.csv".to_string(),
            },
            created_at_tick: 0,
            confidence: 0.9,
        };
        assert!(rec.is_externally_sourced());
    }

    #[test]
    fn is_externally_sourced_rule_fired_false() {
        let rec = ProvenanceRecord {
            record_id: 0,
            tensor_id: 0,
            kind: ProvenanceKind::RuleFired { rule_id: 1 },
            created_at_tick: 0,
            confidence: 0.8,
        };
        assert!(!rec.is_externally_sourced());
    }

    #[test]
    fn is_externally_sourced_inference_step_false() {
        let rec = ProvenanceRecord {
            record_id: 0,
            tensor_id: 0,
            kind: ProvenanceKind::InferenceStep { step: 3 },
            created_at_tick: 0,
            confidence: 0.7,
        };
        assert!(!rec.is_externally_sourced());
    }

    #[test]
    fn is_externally_sourced_derived_false() {
        let rec = ProvenanceRecord {
            record_id: 0,
            tensor_id: 0,
            kind: ProvenanceKind::Derived {
                parent_ids: vec![1, 2],
            },
            created_at_tick: 0,
            confidence: 0.6,
        };
        assert!(!rec.is_externally_sourced());
    }

    // ── externally_sourced_tensors ────────────────────────────────────────────

    #[test]
    fn externally_sourced_tensors_correct_and_sorted() {
        let mut t = TensorProvenanceTracker::new();
        // tensor 5: external
        t.record(5, ProvenanceKind::FactAssertion, 1.0, 0);
        // tensor 2: external via ExternalInput
        t.record(
            2,
            ProvenanceKind::ExternalInput {
                source: "db".to_string(),
            },
            0.9,
            1,
        );
        // tensor 9: NOT external
        t.record(9, ProvenanceKind::RuleFired { rule_id: 0 }, 0.5, 2);
        // tensor 1: mixed; one external record → qualifies
        t.record(1, ProvenanceKind::InferenceStep { step: 0 }, 0.3, 3);
        t.record(1, ProvenanceKind::FactAssertion, 0.8, 4);

        let ext = t.externally_sourced_tensors();
        assert_eq!(ext, vec![1u64, 2u64, 5u64]);
    }

    // ── delete_tensor ─────────────────────────────────────────────────────────

    #[test]
    fn delete_tensor_removes_all_records() {
        let mut t = TensorProvenanceTracker::new();
        t.record(4, ProvenanceKind::FactAssertion, 1.0, 0);
        t.record(4, ProvenanceKind::RuleFired { rule_id: 2 }, 0.5, 1);
        assert_eq!(t.records.len(), 2);

        let ok = t.delete_tensor(4);
        assert!(ok);
        assert!(t.records.is_empty());
        assert!(!t.tensor_records.contains_key(&4));
    }

    #[test]
    fn delete_tensor_returns_false_for_unknown() {
        let mut t = TensorProvenanceTracker::new();
        assert!(!t.delete_tensor(123));
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn stats_total_records_and_unique_tensors() {
        let mut t = TensorProvenanceTracker::new();
        t.record(1, ProvenanceKind::FactAssertion, 1.0, 0);
        t.record(1, ProvenanceKind::RuleFired { rule_id: 0 }, 0.8, 1);
        t.record(2, ProvenanceKind::FactAssertion, 0.5, 2);
        let s = t.stats();
        assert_eq!(s.total_records, 3);
        assert_eq!(s.unique_tensors, 2);
    }

    #[test]
    fn stats_fact_count_and_rule_fired_count() {
        let mut t = TensorProvenanceTracker::new();
        t.record(1, ProvenanceKind::FactAssertion, 1.0, 0);
        t.record(2, ProvenanceKind::FactAssertion, 1.0, 1);
        t.record(3, ProvenanceKind::RuleFired { rule_id: 7 }, 0.9, 2);
        t.record(4, ProvenanceKind::InferenceStep { step: 0 }, 0.6, 3);
        let s = t.stats();
        assert_eq!(s.fact_count, 2);
        assert_eq!(s.rule_fired_count, 1);
    }

    #[test]
    fn stats_avg_confidence() {
        let mut t = TensorProvenanceTracker::new();
        t.record(1, ProvenanceKind::FactAssertion, 0.8, 0);
        t.record(2, ProvenanceKind::RuleFired { rule_id: 0 }, 0.4, 1);
        let s = t.stats();
        assert!((s.avg_confidence - 0.6).abs() < 1e-10);
    }

    #[test]
    fn stats_empty_tracker() {
        let t = TensorProvenanceTracker::new();
        let s = t.stats();
        assert_eq!(s.total_records, 0);
        assert_eq!(s.unique_tensors, 0);
        assert_eq!(s.fact_count, 0);
        assert_eq!(s.rule_fired_count, 0);
        assert_eq!(s.avg_confidence, 0.0);
    }
}
