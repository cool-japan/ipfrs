//! TensorStateSnapshot — captures and restores complete TensorLogic runtime state
//! for migration, debugging, and distributed coordination purposes.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// FNV-1a helper
// ---------------------------------------------------------------------------

/// Compute FNV-1a 64-bit hash of a byte slice.
pub fn fnv1a_u64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 14_695_981_039_346_656_037_u64;
    const PRIME: u64 = 1_099_511_628_211_u64;
    let mut hash = OFFSET_BASIS;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// SnapshotField
// ---------------------------------------------------------------------------

/// A logical field (partition) that may be included in a snapshot.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SnapshotField {
    /// Rule set included.
    Rules,
    /// Fact base included.
    Facts,
    /// Raw tensor data included.
    TensorValues,
    /// System metadata included.
    Metadata,
    /// Active session data included.
    Sessions,
}

// ---------------------------------------------------------------------------
// FieldData
// ---------------------------------------------------------------------------

/// Metadata describing one field captured inside a [`StateSnapshot`].
#[derive(Clone, Debug, PartialEq)]
pub struct FieldData {
    /// Which logical field this record describes.
    pub field: SnapshotField,
    /// Serialised byte size of the field payload.
    pub size_bytes: u64,
    /// FNV-1a checksum of the field-name bytes.
    pub checksum: u64,
    /// Number of logical records in the field (rules, facts, tensors, …).
    pub record_count: usize,
}

impl FieldData {
    /// Construct a [`FieldData`], computing the checksum automatically from
    /// the canonical string name of the field.
    pub fn new(field: SnapshotField, size_bytes: u64, record_count: usize) -> Self {
        let checksum = fnv1a_u64(field_name_bytes(&field));
        Self {
            field,
            size_bytes,
            checksum,
            record_count,
        }
    }
}

/// Return the canonical ASCII name bytes for a [`SnapshotField`].
fn field_name_bytes(field: &SnapshotField) -> &'static [u8] {
    match field {
        SnapshotField::Rules => b"Rules",
        SnapshotField::Facts => b"Facts",
        SnapshotField::TensorValues => b"TensorValues",
        SnapshotField::Metadata => b"Metadata",
        SnapshotField::Sessions => b"Sessions",
    }
}

// ---------------------------------------------------------------------------
// StateSnapshot
// ---------------------------------------------------------------------------

/// A complete, point-in-time snapshot of TensorLogic runtime state.
#[derive(Clone, Debug)]
pub struct StateSnapshot {
    /// Unique, monotonically increasing identifier.
    pub snapshot_id: u64,
    /// Unix epoch seconds when the snapshot was taken.
    pub created_at_secs: u64,
    /// Identifier of the node that created the snapshot.
    pub node_id: String,
    /// Fields captured in this snapshot.
    pub fields: Vec<FieldData>,
    /// Schema / format version (default 1).
    pub format_version: u32,
}

impl StateSnapshot {
    /// Sum of all field payload sizes in bytes.
    pub fn total_size(&self) -> u64 {
        self.fields.iter().map(|f| f.size_bytes).sum()
    }

    /// Returns `true` when the snapshot contains `field`.
    pub fn has_field(&self, field: &SnapshotField) -> bool {
        self.fields.iter().any(|f| &f.field == field)
    }

    /// Number of distinct fields present in the snapshot.
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }
}

// ---------------------------------------------------------------------------
// SnapshotDelta
// ---------------------------------------------------------------------------

/// Difference between two [`StateSnapshot`]s.
#[derive(Clone, Debug)]
pub struct SnapshotDelta {
    /// Snapshot used as the baseline.
    pub base_snapshot_id: u64,
    /// Snapshot that was compared against the baseline.
    pub new_snapshot_id: u64,
    /// Fields present in the newer snapshot but absent from the baseline.
    pub added_fields: Vec<SnapshotField>,
    /// Fields present in the baseline but absent from the newer snapshot.
    pub removed_fields: Vec<SnapshotField>,
    /// Signed byte-size difference (`new.total_size() - old.total_size()`).
    pub size_delta_bytes: i64,
}

impl SnapshotDelta {
    /// Returns `true` when there are no added or removed fields.
    pub fn is_empty(&self) -> bool {
        self.added_fields.is_empty() && self.removed_fields.is_empty()
    }
}

// ---------------------------------------------------------------------------
// StateSnapshotStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for a [`TensorStateSnapshot`] manager.
///
/// Named `StateSnapshotStats` to avoid collision with `SnapshotManagerStats`
/// in the storage module.
#[derive(Clone, Debug)]
pub struct StateSnapshotStats {
    /// Total number of snapshots currently retained.
    pub total_snapshots: usize,
    /// Combined byte size across all retained snapshots.
    pub total_size_bytes: u64,
    /// `snapshot_id` of the oldest retained snapshot, or `None` if empty.
    pub oldest_snapshot_id: Option<u64>,
    /// `snapshot_id` of the most-recently-captured snapshot, or `None` if empty.
    pub newest_snapshot_id: Option<u64>,
}

// ---------------------------------------------------------------------------
// TensorStateSnapshot
// ---------------------------------------------------------------------------

/// Manages captures and retrieval of [`StateSnapshot`]s.
pub struct TensorStateSnapshot {
    /// All retained snapshots keyed by their `snapshot_id`.
    pub snapshots: HashMap<u64, StateSnapshot>,
    /// Counter used to assign the next unique id.
    pub next_id: u64,
}

impl TensorStateSnapshot {
    /// Create a new, empty manager.
    pub fn new() -> Self {
        Self {
            snapshots: HashMap::new(),
            next_id: 1,
        }
    }

    /// Capture a snapshot for `node_id` containing `fields` at `now_secs`.
    ///
    /// Returns the assigned `snapshot_id`.
    pub fn capture(&mut self, node_id: &str, fields: Vec<FieldData>, now_secs: u64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let snapshot = StateSnapshot {
            snapshot_id: id,
            created_at_secs: now_secs,
            node_id: node_id.to_owned(),
            fields,
            format_version: 1,
        };
        self.snapshots.insert(id, snapshot);
        id
    }

    /// Retrieve a snapshot by id.
    pub fn get(&self, id: u64) -> Option<&StateSnapshot> {
        self.snapshots.get(&id)
    }

    /// Delete a snapshot by id.  Returns `true` when the snapshot existed.
    pub fn delete(&mut self, id: u64) -> bool {
        self.snapshots.remove(&id).is_some()
    }

    /// Compute the difference between snapshot `old_id` and snapshot `new_id`.
    ///
    /// Returns `None` if either id is not found.
    pub fn diff(&self, old_id: u64, new_id: u64) -> Option<SnapshotDelta> {
        let old = self.snapshots.get(&old_id)?;
        let new = self.snapshots.get(&new_id)?;

        let old_fields: std::collections::HashSet<&SnapshotField> =
            old.fields.iter().map(|f| &f.field).collect();
        let new_fields: std::collections::HashSet<&SnapshotField> =
            new.fields.iter().map(|f| &f.field).collect();

        let added_fields: Vec<SnapshotField> = new_fields
            .difference(&old_fields)
            .map(|f| (*f).clone())
            .collect();
        let removed_fields: Vec<SnapshotField> = old_fields
            .difference(&new_fields)
            .map(|f| (*f).clone())
            .collect();

        let size_delta_bytes = new.total_size() as i64 - old.total_size() as i64;

        Some(SnapshotDelta {
            base_snapshot_id: old_id,
            new_snapshot_id: new_id,
            added_fields,
            removed_fields,
            size_delta_bytes,
        })
    }

    /// Return a reference to the snapshot with the highest `snapshot_id`.
    pub fn latest(&self) -> Option<&StateSnapshot> {
        self.snapshots.values().max_by_key(|s| s.snapshot_id)
    }

    /// Aggregate statistics over all retained snapshots.
    pub fn stats(&self) -> StateSnapshotStats {
        let total_snapshots = self.snapshots.len();
        let total_size_bytes = self.snapshots.values().map(|s| s.total_size()).sum();
        let oldest_snapshot_id = self.snapshots.keys().copied().min();
        let newest_snapshot_id = self.snapshots.keys().copied().max();
        StateSnapshotStats {
            total_snapshots,
            total_size_bytes,
            oldest_snapshot_id,
            newest_snapshot_id,
        }
    }
}

impl Default for TensorStateSnapshot {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn rules_field(size: u64) -> FieldData {
        FieldData::new(SnapshotField::Rules, size, 10)
    }

    fn facts_field(size: u64) -> FieldData {
        FieldData::new(SnapshotField::Facts, size, 20)
    }

    fn tensor_field(size: u64) -> FieldData {
        FieldData::new(SnapshotField::TensorValues, size, 5)
    }

    fn metadata_field(size: u64) -> FieldData {
        FieldData::new(SnapshotField::Metadata, size, 3)
    }

    fn sessions_field(size: u64) -> FieldData {
        FieldData::new(SnapshotField::Sessions, size, 2)
    }

    // ------------------------------------------------------------------
    // TensorStateSnapshot::new
    // ------------------------------------------------------------------

    #[test]
    fn test_new_starts_empty() {
        let mgr = TensorStateSnapshot::new();
        assert!(mgr.snapshots.is_empty());
        assert_eq!(mgr.next_id, 1);
    }

    // ------------------------------------------------------------------
    // capture / get
    // ------------------------------------------------------------------

    #[test]
    fn test_capture_stores_snapshot() {
        let mut mgr = TensorStateSnapshot::new();
        let id = mgr.capture("node-a", vec![rules_field(512)], 1_000);
        let snap = mgr.get(id).expect("snapshot must exist after capture");
        assert_eq!(snap.snapshot_id, id);
        assert_eq!(snap.node_id, "node-a");
        assert_eq!(snap.created_at_secs, 1_000);
    }

    #[test]
    fn test_capture_returns_incrementing_ids() {
        let mut mgr = TensorStateSnapshot::new();
        let id1 = mgr.capture("node-a", vec![rules_field(100)], 1);
        let id2 = mgr.capture("node-a", vec![rules_field(200)], 2);
        let id3 = mgr.capture("node-a", vec![rules_field(300)], 3);
        assert!(id1 < id2);
        assert!(id2 < id3);
    }

    #[test]
    fn test_get_some() {
        let mut mgr = TensorStateSnapshot::new();
        let id = mgr.capture("node-b", vec![facts_field(256)], 42);
        assert!(mgr.get(id).is_some());
    }

    #[test]
    fn test_get_none_for_unknown_id() {
        let mgr = TensorStateSnapshot::new();
        assert!(mgr.get(999).is_none());
    }

    // ------------------------------------------------------------------
    // delete
    // ------------------------------------------------------------------

    #[test]
    fn test_delete_returns_true_when_exists() {
        let mut mgr = TensorStateSnapshot::new();
        let id = mgr.capture("node-c", vec![rules_field(64)], 10);
        assert!(mgr.delete(id));
    }

    #[test]
    fn test_delete_returns_false_when_missing() {
        let mut mgr = TensorStateSnapshot::new();
        assert!(!mgr.delete(777));
    }

    #[test]
    fn test_delete_removes_snapshot() {
        let mut mgr = TensorStateSnapshot::new();
        let id = mgr.capture("node-d", vec![rules_field(128)], 20);
        mgr.delete(id);
        assert!(mgr.get(id).is_none());
    }

    // ------------------------------------------------------------------
    // has_field
    // ------------------------------------------------------------------

    #[test]
    fn test_has_field_true() {
        let mut mgr = TensorStateSnapshot::new();
        let id = mgr.capture("node-e", vec![facts_field(100)], 5);
        let snap = mgr.get(id).expect("test: should succeed");
        assert!(snap.has_field(&SnapshotField::Facts));
    }

    #[test]
    fn test_has_field_false() {
        let mut mgr = TensorStateSnapshot::new();
        let id = mgr.capture("node-e", vec![facts_field(100)], 5);
        let snap = mgr.get(id).expect("test: should succeed");
        assert!(!snap.has_field(&SnapshotField::Rules));
    }

    // ------------------------------------------------------------------
    // total_size / field_count
    // ------------------------------------------------------------------

    #[test]
    fn test_total_size_sum_of_fields() {
        let mut mgr = TensorStateSnapshot::new();
        let fields = vec![rules_field(100), facts_field(200), tensor_field(50)];
        let id = mgr.capture("node-f", fields, 0);
        let snap = mgr.get(id).expect("test: should succeed");
        assert_eq!(snap.total_size(), 350);
    }

    #[test]
    fn test_field_count_correct() {
        let mut mgr = TensorStateSnapshot::new();
        let fields = vec![rules_field(10), facts_field(20), metadata_field(5)];
        let id = mgr.capture("node-g", fields, 0);
        let snap = mgr.get(id).expect("test: should succeed");
        assert_eq!(snap.field_count(), 3);
    }

    // ------------------------------------------------------------------
    // diff
    // ------------------------------------------------------------------

    #[test]
    fn test_diff_added_fields_detected() {
        let mut mgr = TensorStateSnapshot::new();
        let id1 = mgr.capture("n", vec![rules_field(100)], 1);
        let id2 = mgr.capture("n", vec![rules_field(100), facts_field(50)], 2);
        let delta = mgr.diff(id1, id2).expect("diff must return Some");
        assert!(delta.added_fields.contains(&SnapshotField::Facts));
        assert!(delta.removed_fields.is_empty());
    }

    #[test]
    fn test_diff_removed_fields_detected() {
        let mut mgr = TensorStateSnapshot::new();
        let id1 = mgr.capture("n", vec![rules_field(100), facts_field(50)], 1);
        let id2 = mgr.capture("n", vec![rules_field(100)], 2);
        let delta = mgr.diff(id1, id2).expect("diff must return Some");
        assert!(delta.removed_fields.contains(&SnapshotField::Facts));
        assert!(delta.added_fields.is_empty());
    }

    #[test]
    fn test_diff_size_delta_positive() {
        let mut mgr = TensorStateSnapshot::new();
        let id1 = mgr.capture("n", vec![rules_field(100)], 1);
        let id2 = mgr.capture("n", vec![rules_field(100), facts_field(200)], 2);
        let delta = mgr.diff(id1, id2).expect("test: should succeed");
        assert_eq!(delta.size_delta_bytes, 200);
    }

    #[test]
    fn test_diff_size_delta_negative() {
        let mut mgr = TensorStateSnapshot::new();
        let id1 = mgr.capture("n", vec![rules_field(300)], 1);
        let id2 = mgr.capture("n", vec![rules_field(100)], 2);
        let delta = mgr.diff(id1, id2).expect("test: should succeed");
        assert_eq!(delta.size_delta_bytes, -200);
    }

    #[test]
    fn test_diff_returns_none_for_unknown_old_id() {
        let mut mgr = TensorStateSnapshot::new();
        let id = mgr.capture("n", vec![rules_field(10)], 1);
        assert!(mgr.diff(999, id).is_none());
    }

    #[test]
    fn test_diff_returns_none_for_unknown_new_id() {
        let mut mgr = TensorStateSnapshot::new();
        let id = mgr.capture("n", vec![rules_field(10)], 1);
        assert!(mgr.diff(id, 999).is_none());
    }

    #[test]
    fn test_diff_is_empty_when_no_changes() {
        let mut mgr = TensorStateSnapshot::new();
        let id1 = mgr.capture("n", vec![rules_field(100)], 1);
        let id2 = mgr.capture("n", vec![rules_field(100)], 2);
        let delta = mgr.diff(id1, id2).expect("test: should succeed");
        assert!(delta.is_empty());
    }

    #[test]
    fn test_diff_is_empty_false_when_changes() {
        let mut mgr = TensorStateSnapshot::new();
        let id1 = mgr.capture("n", vec![rules_field(100)], 1);
        let id2 = mgr.capture("n", vec![facts_field(100)], 2);
        let delta = mgr.diff(id1, id2).expect("test: should succeed");
        assert!(!delta.is_empty());
    }

    // ------------------------------------------------------------------
    // latest
    // ------------------------------------------------------------------

    #[test]
    fn test_latest_returns_highest_id() {
        let mut mgr = TensorStateSnapshot::new();
        mgr.capture("n", vec![rules_field(10)], 1);
        mgr.capture("n", vec![rules_field(20)], 2);
        let id3 = mgr.capture("n", vec![rules_field(30)], 3);
        let latest = mgr.latest().expect("latest must be Some");
        assert_eq!(latest.snapshot_id, id3);
    }

    #[test]
    fn test_latest_none_when_empty() {
        let mgr = TensorStateSnapshot::new();
        assert!(mgr.latest().is_none());
    }

    // ------------------------------------------------------------------
    // stats
    // ------------------------------------------------------------------

    #[test]
    fn test_stats_total_snapshots() {
        let mut mgr = TensorStateSnapshot::new();
        mgr.capture("n", vec![rules_field(10)], 1);
        mgr.capture("n", vec![rules_field(10)], 2);
        assert_eq!(mgr.stats().total_snapshots, 2);
    }

    #[test]
    fn test_stats_total_size_bytes() {
        let mut mgr = TensorStateSnapshot::new();
        mgr.capture("n", vec![rules_field(100)], 1);
        mgr.capture("n", vec![facts_field(200)], 2);
        assert_eq!(mgr.stats().total_size_bytes, 300);
    }

    #[test]
    fn test_stats_oldest_newest_snapshot_id() {
        let mut mgr = TensorStateSnapshot::new();
        let id1 = mgr.capture("n", vec![rules_field(10)], 1);
        mgr.capture("n", vec![rules_field(10)], 2);
        let id3 = mgr.capture("n", vec![rules_field(10)], 3);
        let stats = mgr.stats();
        assert_eq!(stats.oldest_snapshot_id, Some(id1));
        assert_eq!(stats.newest_snapshot_id, Some(id3));
    }

    // ------------------------------------------------------------------
    // fnv1a_u64
    // ------------------------------------------------------------------

    #[test]
    fn test_fnv1a_u64_deterministic() {
        let a = fnv1a_u64(b"Rules");
        let b = fnv1a_u64(b"Rules");
        assert_eq!(a, b);
    }

    #[test]
    fn test_fnv1a_u64_different_inputs_differ() {
        let a = fnv1a_u64(b"Rules");
        let b = fnv1a_u64(b"Facts");
        assert_ne!(a, b);
    }

    // ------------------------------------------------------------------
    // FieldData checksum
    // ------------------------------------------------------------------

    #[test]
    fn test_field_data_checksum_matches_fnv1a_of_name() {
        let fd = FieldData::new(SnapshotField::Rules, 512, 10);
        assert_eq!(fd.checksum, fnv1a_u64(b"Rules"));
    }

    #[test]
    fn test_all_fields_have_distinct_checksums() {
        let fields = [
            SnapshotField::Rules,
            SnapshotField::Facts,
            SnapshotField::TensorValues,
            SnapshotField::Metadata,
            SnapshotField::Sessions,
        ];
        let checksums: Vec<u64> = fields
            .iter()
            .map(|f| FieldData::new(f.clone(), 0, 0).checksum)
            .collect();
        let unique: std::collections::HashSet<u64> = checksums.iter().copied().collect();
        assert_eq!(
            unique.len(),
            fields.len(),
            "all field checksums must be distinct"
        );
    }

    // ------------------------------------------------------------------
    // format_version default
    // ------------------------------------------------------------------

    #[test]
    fn test_snapshot_format_version_default_one() {
        let mut mgr = TensorStateSnapshot::new();
        let id = mgr.capture("n", vec![sessions_field(64)], 0);
        assert_eq!(mgr.get(id).expect("test: should succeed").format_version, 1);
    }
}
