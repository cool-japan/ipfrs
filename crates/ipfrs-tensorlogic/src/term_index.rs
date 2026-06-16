//! Inverted index over TensorLogic terms for fast predicate/constant/variable lookup.
//!
//! [`TermIndexBuilder`] maintains a posting-list index keyed by [`TermType`].
//! Each posting list is a `Vec<TermRef>` that records which rule (or fact) and
//! at which position a given term appears.  The index supports incremental
//! construction (`index_term`), point lookups (`lookup`, `lookup_predicate`,
//! `lookup_constant`), derived queries (`rules_for_predicate`), mutation
//! (`remove_rule`, `clear`), and statistics (`stats`).

use std::collections::HashMap;

// ── TermPosition ─────────────────────────────────────────────────────────────

/// Describes where in a rule (or as a standalone fact) a term appears.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TermPosition {
    /// The term appears in the rule head.
    Head,
    /// The term appears in the body at the given clause index.
    Body(usize),
    /// The term is a standalone fact (not part of a rule).
    Fact,
}

// ── TermType ─────────────────────────────────────────────────────────────────

/// The kind/value of a term used as an index key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TermType {
    /// A predicate name (functor).
    Predicate(String),
    /// A string constant.
    Constant(String),
    /// A logic variable name.
    Variable(String),
    /// A numeric value stored as the bit-pattern of an `f64` so that the type
    /// remains `Hash + Eq`.  Use [`TermType::from_f64`] and
    /// [`TermType::as_f64`] for ergonomic construction/access.
    Numeric(u64),
}

impl TermType {
    /// Construct a [`TermType::Numeric`] from an `f64`.
    #[inline]
    pub fn from_f64(v: f64) -> Self {
        Self::Numeric(v.to_bits())
    }

    /// Return the `f64` value stored in a [`TermType::Numeric`], or `None`
    /// for other variants.
    #[inline]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Numeric(bits) => Some(f64::from_bits(*bits)),
            _ => None,
        }
    }
}

// ── TermRef ───────────────────────────────────────────────────────────────────

/// A single posting-list entry: records which rule (or fact) mentions a term
/// and at which position.
///
/// Note: this `TermRef` is specific to the `term_index` module and is
/// *distinct* from [`crate::ir::TermRef`], which is a CID-based reference used
/// in the IR layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TermRef {
    /// Identifier of the rule or fact that contains this term.
    pub rule_id: u64,
    /// Position within the rule/fact where the term appears.
    pub position: TermPosition,
    /// The kind/value of the term.
    pub term_type: TermType,
}

// ── IndexStats ────────────────────────────────────────────────────────────────

/// Summary statistics for a [`TermIndexBuilder`].
#[derive(Clone, Debug, PartialEq)]
pub struct IndexStats {
    /// Number of unique term keys in the index.
    pub total_terms: usize,
    /// Sum of all posting-list lengths.
    pub total_refs: usize,
}

impl IndexStats {
    /// Average number of posting-list entries per unique term key.
    ///
    /// Returns `0.0` when the index is empty.
    pub fn average_posting_len(&self) -> f64 {
        if self.total_terms == 0 {
            0.0
        } else {
            self.total_refs as f64 / self.total_terms as f64
        }
    }
}

// ── TermIndexBuilder ──────────────────────────────────────────────────────────

/// Builds and queries an inverted index over TensorLogic terms.
///
/// The index maps each [`TermType`] to the list of [`TermRef`]s (posting list)
/// that identify every rule/fact position where that term occurs.
///
/// # Example
///
/// ```
/// use ipfrs_tensorlogic::term_index::{TermIndexBuilder, TermPosition, TermType};
///
/// let mut builder = TermIndexBuilder::new();
///
/// builder.index_term(
///     TermType::Predicate("parent".to_string()),
///     1,
///     TermPosition::Head,
/// );
///
/// let refs = builder.lookup_predicate("parent");
/// assert_eq!(refs.len(), 1);
/// assert_eq!(refs[0].rule_id, 1);
/// ```
#[derive(Debug, Default)]
pub struct TermIndexBuilder {
    /// The underlying inverted index.
    pub index: HashMap<TermType, Vec<TermRef>>,
}

impl TermIndexBuilder {
    /// Create a new, empty [`TermIndexBuilder`].
    pub fn new() -> Self {
        Self {
            index: HashMap::new(),
        }
    }

    /// Add a term occurrence to the index.
    ///
    /// Appends a [`TermRef`] to the posting list for `term`.
    pub fn index_term(&mut self, term: TermType, rule_id: u64, position: TermPosition) {
        let entry = self.index.entry(term.clone()).or_default();
        entry.push(TermRef {
            rule_id,
            position,
            term_type: term,
        });
    }

    /// Return the posting list for `term`, or an empty slice if absent.
    pub fn lookup(&self, term: &TermType) -> &[TermRef] {
        self.index.get(term).map(Vec::as_slice).unwrap_or_default()
    }

    /// Return the posting list for `Predicate(name)`, or an empty slice.
    pub fn lookup_predicate(&self, name: &str) -> &[TermRef] {
        self.lookup(&TermType::Predicate(name.to_string()))
    }

    /// Return the posting list for `Constant(name)`, or an empty slice.
    pub fn lookup_constant(&self, name: &str) -> &[TermRef] {
        self.lookup(&TermType::Constant(name.to_string()))
    }

    /// Return a deduplicated, sorted list of rule IDs that mention
    /// `Predicate(name)`.
    pub fn rules_for_predicate(&self, name: &str) -> Vec<u64> {
        let mut ids: Vec<u64> = self
            .lookup_predicate(name)
            .iter()
            .map(|r| r.rule_id)
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Remove all [`TermRef`]s belonging to `rule_id` from every posting list.
    ///
    /// Posting lists that become empty after removal are dropped from the index.
    pub fn remove_rule(&mut self, rule_id: u64) {
        self.index.retain(|_, refs| {
            refs.retain(|r| r.rule_id != rule_id);
            !refs.is_empty()
        });
    }

    /// Return statistics about the current state of the index.
    pub fn stats(&self) -> IndexStats {
        let total_terms = self.index.len();
        let total_refs = self.index.values().map(Vec::len).sum();
        IndexStats {
            total_terms,
            total_refs,
        }
    }

    /// Remove all entries from the index.
    pub fn clear(&mut self) {
        self.index.clear();
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a small builder with two predicates and a constant.
    fn sample_builder() -> TermIndexBuilder {
        let mut b = TermIndexBuilder::new();
        b.index_term(
            TermType::Predicate("parent".to_string()),
            1,
            TermPosition::Head,
        );
        b.index_term(
            TermType::Predicate("parent".to_string()),
            2,
            TermPosition::Body(0),
        );
        b.index_term(
            TermType::Predicate("child".to_string()),
            2,
            TermPosition::Head,
        );
        b.index_term(
            TermType::Constant("alice".to_string()),
            1,
            TermPosition::Body(0),
        );
        b
    }

    // ── 1. index_term builds the posting list ────────────────────────────────

    #[test]
    fn test_index_term_single_entry() {
        let mut b = TermIndexBuilder::new();
        b.index_term(
            TermType::Predicate("foo".to_string()),
            10,
            TermPosition::Head,
        );
        assert_eq!(b.index.len(), 1);
        let list = b.lookup_predicate("foo");
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].rule_id, 10);
        assert_eq!(list[0].position, TermPosition::Head);
    }

    #[test]
    fn test_index_term_multiple_entries_same_key() {
        let mut b = TermIndexBuilder::new();
        b.index_term(TermType::Predicate("p".to_string()), 1, TermPosition::Head);
        b.index_term(TermType::Predicate("p".to_string()), 2, TermPosition::Head);
        b.index_term(
            TermType::Predicate("p".to_string()),
            3,
            TermPosition::Body(1),
        );
        let list = b.lookup_predicate("p");
        assert_eq!(list.len(), 3);
    }

    // ── 2. lookup returns refs ────────────────────────────────────────────────

    #[test]
    fn test_lookup_returns_correct_refs() {
        let b = sample_builder();
        let list = b.lookup(&TermType::Predicate("parent".to_string()));
        assert_eq!(list.len(), 2);
        assert!(list
            .iter()
            .any(|r| r.rule_id == 1 && r.position == TermPosition::Head));
        assert!(list
            .iter()
            .any(|r| r.rule_id == 2 && r.position == TermPosition::Body(0)));
    }

    #[test]
    fn test_lookup_missing_returns_empty() {
        let b = sample_builder();
        let list = b.lookup(&TermType::Predicate("nonexistent".to_string()));
        assert!(list.is_empty());
    }

    // ── 3. lookup_predicate ───────────────────────────────────────────────────

    #[test]
    fn test_lookup_predicate_found() {
        let b = sample_builder();
        assert_eq!(b.lookup_predicate("child").len(), 1);
    }

    #[test]
    fn test_lookup_predicate_not_found() {
        let b = sample_builder();
        assert!(b.lookup_predicate("grandparent").is_empty());
    }

    // ── 4. lookup_constant ────────────────────────────────────────────────────

    #[test]
    fn test_lookup_constant_found() {
        let b = sample_builder();
        let list = b.lookup_constant("alice");
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].rule_id, 1);
        assert_eq!(list[0].position, TermPosition::Body(0));
    }

    #[test]
    fn test_lookup_constant_not_found() {
        let b = sample_builder();
        assert!(b.lookup_constant("bob").is_empty());
    }

    // ── 5. rules_for_predicate deduplicates ───────────────────────────────────

    #[test]
    fn test_rules_for_predicate_deduplicates() {
        let mut b = TermIndexBuilder::new();
        // Rule 5 mentions "ancestor" in both head and body — should appear once.
        b.index_term(
            TermType::Predicate("ancestor".to_string()),
            5,
            TermPosition::Head,
        );
        b.index_term(
            TermType::Predicate("ancestor".to_string()),
            5,
            TermPosition::Body(0),
        );
        b.index_term(
            TermType::Predicate("ancestor".to_string()),
            7,
            TermPosition::Head,
        );

        let ids = b.rules_for_predicate("ancestor");
        assert_eq!(ids, vec![5, 7]);
    }

    #[test]
    fn test_rules_for_predicate_sorted() {
        let mut b = TermIndexBuilder::new();
        b.index_term(TermType::Predicate("r".to_string()), 10, TermPosition::Head);
        b.index_term(
            TermType::Predicate("r".to_string()),
            3,
            TermPosition::Body(0),
        );
        b.index_term(TermType::Predicate("r".to_string()), 7, TermPosition::Head);

        let ids = b.rules_for_predicate("r");
        assert_eq!(ids, vec![3, 7, 10]);
    }

    #[test]
    fn test_rules_for_predicate_empty() {
        let b = TermIndexBuilder::new();
        assert!(b.rules_for_predicate("missing").is_empty());
    }

    // ── 6. remove_rule removes from all lists and drops empty ─────────────────

    #[test]
    fn test_remove_rule_removes_refs() {
        let mut b = sample_builder();
        // "parent" has entries for rules 1 and 2; remove rule 1.
        b.remove_rule(1);
        let list = b.lookup_predicate("parent");
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].rule_id, 2);
    }

    #[test]
    fn test_remove_rule_drops_empty_posting_list() {
        let mut b = sample_builder();
        // "alice" constant is only referenced by rule 1; removing rule 1 drops it.
        b.remove_rule(1);
        assert!(!b
            .index
            .contains_key(&TermType::Constant("alice".to_string())));
    }

    #[test]
    fn test_remove_rule_nonexistent_is_noop() {
        let mut b = sample_builder();
        let before = b.stats();
        b.remove_rule(999);
        assert_eq!(b.stats(), before);
    }

    // ── 7. stats ──────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_counts() {
        let b = sample_builder();
        let s = b.stats();
        // Keys: "parent", "child", "alice" → 3 unique terms.
        assert_eq!(s.total_terms, 3);
        // Refs: parent×2, child×1, alice×1 → 4.
        assert_eq!(s.total_refs, 4);
    }

    #[test]
    fn test_stats_empty() {
        let b = TermIndexBuilder::new();
        let s = b.stats();
        assert_eq!(s.total_terms, 0);
        assert_eq!(s.total_refs, 0);
    }

    // ── 8. average_posting_len ────────────────────────────────────────────────

    #[test]
    fn test_average_posting_len() {
        let b = sample_builder();
        let s = b.stats();
        // 4 refs / 3 terms ≈ 1.333…
        let avg = s.average_posting_len();
        assert!((avg - (4.0 / 3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_average_posting_len_empty() {
        let s = IndexStats {
            total_terms: 0,
            total_refs: 0,
        };
        assert_eq!(s.average_posting_len(), 0.0);
    }

    // ── 9. clear empties the index ────────────────────────────────────────────

    #[test]
    fn test_clear_empties_index() {
        let mut b = sample_builder();
        b.clear();
        assert!(b.index.is_empty());
        let s = b.stats();
        assert_eq!(s.total_terms, 0);
        assert_eq!(s.total_refs, 0);
    }

    // ── 10. multiple terms for the same rule ──────────────────────────────────

    #[test]
    fn test_multiple_terms_same_rule() {
        let mut b = TermIndexBuilder::new();
        let rule_id = 42;
        b.index_term(
            TermType::Predicate("path".to_string()),
            rule_id,
            TermPosition::Head,
        );
        b.index_term(
            TermType::Predicate("edge".to_string()),
            rule_id,
            TermPosition::Body(0),
        );
        b.index_term(
            TermType::Variable("X".to_string()),
            rule_id,
            TermPosition::Body(0),
        );
        b.index_term(
            TermType::Variable("Y".to_string()),
            rule_id,
            TermPosition::Body(0),
        );

        assert_eq!(b.lookup_predicate("path").len(), 1);
        assert_eq!(b.lookup_predicate("edge").len(), 1);
        assert_eq!(b.lookup(&TermType::Variable("X".to_string())).len(), 1);
        assert_eq!(b.lookup(&TermType::Variable("Y".to_string())).len(), 1);

        // Removing the rule clears all four entries.
        b.remove_rule(rule_id);
        assert!(b.index.is_empty());
    }

    // ── 11. Numeric term ──────────────────────────────────────────────────────

    #[test]
    fn test_numeric_term_index_and_lookup() {
        let mut b = TermIndexBuilder::new();
        let num = TermType::from_f64(std::f64::consts::PI);
        b.index_term(num.clone(), 99, TermPosition::Fact);

        let list = b.lookup(&num);
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].rule_id, 99);
        assert_eq!(list[0].position, TermPosition::Fact);
        assert!(
            (list[0].term_type.as_f64().expect("test: should succeed") - std::f64::consts::PI)
                .abs()
                < 1e-10
        );
    }

    #[test]
    fn test_numeric_term_distinct_values() {
        let mut b = TermIndexBuilder::new();
        b.index_term(TermType::from_f64(1.0), 1, TermPosition::Fact);
        b.index_term(TermType::from_f64(2.0), 2, TermPosition::Fact);
        b.index_term(TermType::from_f64(1.0), 3, TermPosition::Fact);

        // 1.0 and 2.0 are distinct keys.
        assert_eq!(b.index.len(), 2);
        assert_eq!(b.lookup(&TermType::from_f64(1.0)).len(), 2);
        assert_eq!(b.lookup(&TermType::from_f64(2.0)).len(), 1);
    }

    // ── 12. Fact position ─────────────────────────────────────────────────────

    #[test]
    fn test_fact_position() {
        let mut b = TermIndexBuilder::new();
        b.index_term(
            TermType::Predicate("likes".to_string()),
            0,
            TermPosition::Fact,
        );
        let list = b.lookup_predicate("likes");
        assert_eq!(list[0].position, TermPosition::Fact);
    }

    // ── 13. Body position variant ─────────────────────────────────────────────

    #[test]
    fn test_body_position_index() {
        let mut b = TermIndexBuilder::new();
        b.index_term(
            TermType::Predicate("q".to_string()),
            7,
            TermPosition::Body(3),
        );
        let list = b.lookup_predicate("q");
        assert_eq!(list[0].position, TermPosition::Body(3));
    }

    // ── 14. term_type stored correctly in TermRef ─────────────────────────────

    #[test]
    fn test_term_ref_stores_term_type() {
        let mut b = TermIndexBuilder::new();
        b.index_term(
            TermType::Constant("hello".to_string()),
            5,
            TermPosition::Head,
        );
        let list = b.lookup_constant("hello");
        assert_eq!(list[0].term_type, TermType::Constant("hello".to_string()));
    }

    // ── 15. remove_rule on shared posting list ───────────────────────────────

    #[test]
    fn test_remove_rule_partial_posting_list_retained() {
        let mut b = TermIndexBuilder::new();
        b.index_term(TermType::Predicate("p".to_string()), 1, TermPosition::Head);
        b.index_term(TermType::Predicate("p".to_string()), 2, TermPosition::Head);
        b.index_term(TermType::Predicate("p".to_string()), 3, TermPosition::Head);

        b.remove_rule(2);

        let ids = b.rules_for_predicate("p");
        assert_eq!(ids, vec![1, 3]);
    }

    // ── 16. stats after remove ────────────────────────────────────────────────

    #[test]
    fn test_stats_after_remove() {
        let mut b = sample_builder();
        // Before: 3 terms, 4 refs.
        b.remove_rule(1);
        // Rule 1 had: parent(head) and constant(alice). Both removed.
        // "alice" posting list becomes empty and is dropped.
        // "parent" posting list still has rule 2 → remains.
        let s = b.stats();
        assert_eq!(s.total_terms, 2); // "parent", "child"
        assert_eq!(s.total_refs, 2); // parent/rule2 + child/rule2
    }

    // ── 17. new() constructs empty builder ────────────────────────────────────

    #[test]
    fn test_new_is_empty() {
        let b = TermIndexBuilder::new();
        assert!(b.index.is_empty());
        let s = b.stats();
        assert_eq!(s.total_terms, 0);
        assert_eq!(s.total_refs, 0);
        assert_eq!(s.average_posting_len(), 0.0);
    }
}
