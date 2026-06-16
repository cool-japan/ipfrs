//! Arrow IPC schema evolution support for TensorSwap protocol.
//!
//! This module provides:
//! - [`SchemaVersion`]: a versioned identifier for an Arrow schema.
//! - [`SchemaRegistry`]: a registry that tracks multiple named schema versions
//!   and validates compatibility when schemas evolve.
//! - [`SchemaEvolutionFrame`]: the wire frame used to signal a schema change
//!   mid-stream inside a TensorSwap session.
//! - [`EvolutionStrategy`]: the policy that governs how schemas may evolve.
//!
//! # Compatibility rules
//!
//! Two schema versions are *compatible* (reader can decode writer data) when:
//! 1. They share the same name.
//! 2. Every field present in the *writer* schema exists with the same name and
//!    data type in the *reader* schema.
//! 3. New fields added by the reader (absent in the writer) must be nullable so
//!    that missing values can be represented as null.

use arrow::datatypes::{DataType, Field, Schema};
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// FNV-1a implementation (pure Rust, no external dep)
// ---------------------------------------------------------------------------

/// Compute a 64-bit FNV-1a fingerprint over an arbitrary byte slice.
fn fnv1a_64(data: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;
    let mut hash = OFFSET_BASIS;
    for byte in data {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// SchemaVersion
// ---------------------------------------------------------------------------

/// A versioned schema identifier that enables fast equality and compatibility
/// checks via a FNV-1a fingerprint of field names and data types.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SchemaVersion {
    /// Logical name of the schema (e.g. `"embeddings"`, `"attention_layer"`).
    pub name: String,
    /// Monotonically increasing version number within the same named schema.
    pub version: u32,
    /// FNV-1a hash of every `"<field_name>:<data_type>"` string concatenated
    /// together, in field declaration order, for fast inequality detection.
    pub fingerprint: u64,
}

impl SchemaVersion {
    /// Build a `SchemaVersion` from a name and an Arrow [`Schema`].
    ///
    /// The version starts at 1 for newly created versions; use
    /// [`SchemaRegistry::register`] / [`SchemaRegistry::upgrade_schema`] to
    /// obtain properly sequenced versions.
    pub fn new(name: &str, schema: &Schema) -> Self {
        Self {
            name: name.to_owned(),
            version: 1,
            fingerprint: Self::fingerprint_of(schema),
        }
    }

    /// Compute the FNV-1a fingerprint for a given [`Schema`].
    ///
    /// The fingerprint covers field names and their serialised data-type
    /// strings in declaration order.
    pub fn fingerprint_of(schema: &Schema) -> u64 {
        let mut buf = String::new();
        for field in schema.fields() {
            buf.push_str(field.name());
            buf.push(':');
            buf.push_str(&format!("{:?}", field.data_type()));
            buf.push(';');
        }
        fnv1a_64(buf.as_bytes())
    }

    /// Returns `true` when `self` is compatible with (can be read by) `other`.
    ///
    /// Compatibility requires:
    /// - Same schema name.
    /// - `self.version <= other.version` (older writer readable by newer reader).
    pub fn is_compatible_with(&self, other: &SchemaVersion) -> bool {
        self.name == other.name && self.version <= other.version
    }
}

// ---------------------------------------------------------------------------
// SchemaError
// ---------------------------------------------------------------------------

/// Errors that can occur during schema registration or evolution.
#[derive(Debug, thiserror::Error)]
pub enum SchemaError {
    /// The requested schema name is not registered.
    #[error("Schema '{0}' not found")]
    NotFound(String),

    /// The proposed schema change violates the active evolution strategy.
    #[error("Incompatible schema change: {0}")]
    Incompatible(String),

    /// A schema with this version number is already registered.
    #[error("Schema version {0} already registered")]
    AlreadyRegistered(u32),
}

// ---------------------------------------------------------------------------
// EvolutionStrategy
// ---------------------------------------------------------------------------

/// Governs which kinds of schema changes are permitted during an ongoing
/// TensorSwap session.
#[derive(Debug, Clone)]
pub enum EvolutionStrategy {
    /// Only nullable columns may be appended to the schema; existing fields
    /// must remain unchanged.  Always backward-compatible.
    AddColumnsOnly,

    /// Columns may be renamed via an explicit old→new mapping.  All renames
    /// must be listed; unlisted fields must be identical.
    RenameColumns {
        /// Map from old field name to new field name.
        mapping: HashMap<String, String>,
    },

    /// The schema may change arbitrarily.  If the remote peer does not
    /// advertise support for `FullReplacement`, reconnection is required.
    FullReplacement,
}

// ---------------------------------------------------------------------------
// SchemaRegistry
// ---------------------------------------------------------------------------

/// Registry that maps versioned [`SchemaVersion`] keys to Arrow [`Schema`]
/// objects and enforces evolution constraints.
pub struct SchemaRegistry {
    /// Primary store: version key → schema.
    schemas: HashMap<SchemaVersion, Arc<Schema>>,
    /// Tracks the latest version number for each named schema.
    latest: HashMap<String, u32>,
}

impl Default for SchemaRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SchemaRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            schemas: HashMap::new(),
            latest: HashMap::new(),
        }
    }

    /// Register a schema under `name`, automatically assigning the next
    /// version number.  Returns the [`SchemaVersion`] that was assigned.
    ///
    /// If this is the first registration for `name` the version is set to 1.
    pub fn register(&mut self, name: &str, schema: Arc<Schema>) -> SchemaVersion {
        let version = self.latest.get(name).copied().unwrap_or(0) + 1;
        let sv = SchemaVersion {
            name: name.to_owned(),
            version,
            fingerprint: SchemaVersion::fingerprint_of(&schema),
        };
        self.schemas.insert(sv.clone(), schema);
        self.latest.insert(name.to_owned(), version);
        sv
    }

    /// Look up the [`Schema`] for a specific [`SchemaVersion`].
    pub fn get(&self, version: &SchemaVersion) -> Option<Arc<Schema>> {
        self.schemas.get(version).cloned()
    }

    /// Return the [`SchemaVersion`] with the highest version number for
    /// `name`, or `None` if the name has never been registered.
    pub fn latest_version(&self, name: &str) -> Option<SchemaVersion> {
        let ver = *self.latest.get(name)?;
        self.schemas
            .keys()
            .find(|sv| sv.name == name && sv.version == ver)
            .cloned()
    }

    /// Attempt to register a new version of `name` using the
    /// [`EvolutionStrategy::AddColumnsOnly`] rule: only nullable fields may be
    /// added; no existing field may be removed or have its type changed.
    ///
    /// For richer strategies call [`Self::upgrade_schema_with_strategy`].
    pub fn upgrade_schema(
        &mut self,
        name: &str,
        new_schema: Arc<Schema>,
    ) -> Result<SchemaVersion, SchemaError> {
        self.upgrade_schema_with_strategy(name, new_schema, &EvolutionStrategy::AddColumnsOnly)
    }

    /// Upgrade a schema using an explicit [`EvolutionStrategy`].
    ///
    /// - [`EvolutionStrategy::AddColumnsOnly`]: new fields must be nullable;
    ///   existing fields must be unchanged.
    /// - [`EvolutionStrategy::RenameColumns`]: applies the rename mapping then
    ///   validates that all other fields are unchanged.
    /// - [`EvolutionStrategy::FullReplacement`]: always succeeds (the caller
    ///   is responsible for ensuring the peer supports this).
    pub fn upgrade_schema_with_strategy(
        &mut self,
        name: &str,
        new_schema: Arc<Schema>,
        strategy: &EvolutionStrategy,
    ) -> Result<SchemaVersion, SchemaError> {
        // Retrieve the current latest schema for comparison.
        let current_sv = self
            .latest_version(name)
            .ok_or_else(|| SchemaError::NotFound(name.to_owned()))?;

        let current_schema = self
            .get(&current_sv)
            .ok_or_else(|| SchemaError::NotFound(name.to_owned()))?;

        match strategy {
            EvolutionStrategy::AddColumnsOnly => {
                validate_add_columns_only(&current_schema, &new_schema)?;
            }
            EvolutionStrategy::RenameColumns { mapping } => {
                validate_rename_columns(&current_schema, &new_schema, mapping)?;
            }
            EvolutionStrategy::FullReplacement => {
                // No structural validation; any schema is accepted.
            }
        }

        // Assign the next version and store.
        Ok(self.register(name, new_schema))
    }

    /// Determine whether data written with `writer_version` can be decoded by
    /// a reader holding `reader_version`.
    ///
    /// This performs a structural check on the stored schemas in addition to
    /// the version number check in [`SchemaVersion::is_compatible_with`].
    pub fn can_read_with(
        &self,
        writer_version: &SchemaVersion,
        reader_version: &SchemaVersion,
    ) -> bool {
        if !writer_version.is_compatible_with(reader_version) {
            return false;
        }

        // Obtain both schemas; if either is missing we cannot confirm.
        let (Some(writer_schema), Some(reader_schema)) =
            (self.get(writer_version), self.get(reader_version))
        else {
            return false;
        };

        // Every field the writer produced must be present and type-compatible
        // in the reader schema.
        for writer_field in writer_schema.fields() {
            match reader_schema.field_with_name(writer_field.name()) {
                Ok(reader_field) => {
                    if !types_compatible(writer_field.data_type(), reader_field.data_type()) {
                        return false;
                    }
                }
                Err(_) => {
                    // Writer field missing from reader schema — incompatible.
                    return false;
                }
            }
        }

        true
    }
}

// ---------------------------------------------------------------------------
// SchemaEvolutionFrame
// ---------------------------------------------------------------------------

/// Wire frame that signals a schema change mid-stream in a TensorSwap session.
///
/// When a sender wants to start emitting data using a different Arrow schema it
/// first serialises a `SchemaEvolutionFrame` and transmits it to the receiver.
/// The receiver must acknowledge the new schema before the sender continues
/// with Arrow IPC batches encoded against `new_version`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SchemaEvolutionFrame {
    /// Identifier of the TensorSwap session this evolution applies to.
    pub session_id: String,
    /// The schema version that was active before the change.
    pub old_version: SchemaVersion,
    /// The schema version that will be used after the change.
    pub new_version: SchemaVersion,
    /// Serialised Arrow IPC `Schema` message bytes for `new_version`.
    ///
    /// Encoded using `arrow::ipc::writer::schema_to_bytes` so that the
    /// receiver can reconstruct the full [`Schema`] without registry access.
    pub schema_ipc: Vec<u8>,
}

impl SchemaEvolutionFrame {
    /// Construct a `SchemaEvolutionFrame`, serialising `new_schema` into Arrow
    /// IPC format automatically.
    pub fn new(
        session_id: impl Into<String>,
        old_version: SchemaVersion,
        new_version: SchemaVersion,
        new_schema: &Schema,
    ) -> Self {
        let schema_ipc = schema_to_ipc_bytes(new_schema);
        Self {
            session_id: session_id.into(),
            old_version,
            new_version,
            schema_ipc,
        }
    }

    /// Serialise this frame to JSON bytes suitable for transmission.
    pub fn to_bytes(&self) -> Result<Vec<u8>, SchemaError> {
        serde_json::to_vec(self)
            .map_err(|e| SchemaError::Incompatible(format!("serialisation error: {e}")))
    }

    /// Deserialise a `SchemaEvolutionFrame` from JSON bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SchemaError> {
        serde_json::from_slice(bytes)
            .map_err(|e| SchemaError::Incompatible(format!("deserialisation error: {e}")))
    }

    /// Recover the Arrow [`Schema`] embedded in `schema_ipc`.
    pub fn decode_schema(&self) -> Result<Arc<Schema>, SchemaError> {
        ipc_bytes_to_schema(&self.schema_ipc)
    }
}

// ---------------------------------------------------------------------------
// Arrow IPC helpers
// ---------------------------------------------------------------------------

/// Serialise an Arrow [`Schema`] to IPC `Schema` message bytes.
///
/// The bytes include the 4-byte continuation marker and 4-byte length prefix
/// that Arrow IPC streaming format prepends to every message.
pub fn schema_to_ipc_bytes(schema: &Schema) -> Vec<u8> {
    use arrow::ipc::writer::{DictionaryTracker, IpcDataGenerator, IpcWriteOptions};

    let opts = IpcWriteOptions::default();
    let gen = IpcDataGenerator {};
    let mut tracker = DictionaryTracker::new(false);
    let encoded = gen.schema_to_bytes_with_dictionary_tracker(schema, &mut tracker, &opts);
    // `ipc_message` is the flatbuffer bytes for the Message; prepend the
    // standard IPC continuation marker (4×0xFF) and the 32-bit length.
    let msg = &encoded.ipc_message;
    let len = msg.len() as u32;
    let mut out = Vec::with_capacity(8 + msg.len());
    out.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(msg);
    out
}

/// Deserialise an Arrow [`Schema`] from IPC `Schema` message bytes.
///
/// Accepts bytes produced by [`schema_to_ipc_bytes`] (with or without the
/// 8-byte continuation+length prefix).
pub fn ipc_bytes_to_schema(bytes: &[u8]) -> Result<Arc<Schema>, SchemaError> {
    use arrow::ipc::convert::fb_to_schema;
    use arrow::ipc::root_as_message;

    // The IPC format prepends a 4-byte continuation marker (0xFF×4) followed
    // by a 4-byte little-endian message length.  Strip them when present.
    let payload = if bytes.len() >= 8 && bytes[0..4] == [0xFF, 0xFF, 0xFF, 0xFF] {
        &bytes[8..]
    } else {
        bytes
    };

    let message = root_as_message(payload)
        .map_err(|e| SchemaError::Incompatible(format!("invalid Arrow IPC message: {e}")))?;

    let schema_ref = message.header_as_schema().ok_or_else(|| {
        SchemaError::Incompatible("IPC message does not contain a Schema header".to_owned())
    })?;

    Ok(Arc::new(fb_to_schema(schema_ref)))
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Validate an [`EvolutionStrategy::AddColumnsOnly`] transition.
///
/// Rules:
/// 1. All fields present in `old` must exist with the same name and data type
///    in `new`.
/// 2. Any field that appears only in `new` must be nullable.
fn validate_add_columns_only(old: &Schema, new: &Schema) -> Result<(), SchemaError> {
    // Rule 1: every old field must survive.
    for old_field in old.fields() {
        match new.field_with_name(old_field.name()) {
            Ok(new_field) => {
                if !types_compatible(old_field.data_type(), new_field.data_type()) {
                    return Err(SchemaError::Incompatible(format!(
                        "field '{}' changed type from {:?} to {:?}",
                        old_field.name(),
                        old_field.data_type(),
                        new_field.data_type(),
                    )));
                }
            }
            Err(_) => {
                return Err(SchemaError::Incompatible(format!(
                    "existing field '{}' was removed",
                    old_field.name()
                )));
            }
        }
    }

    // Rule 2: new fields must be nullable.
    for new_field in new.fields() {
        if old.field_with_name(new_field.name()).is_err() && !new_field.is_nullable() {
            return Err(SchemaError::Incompatible(format!(
                "new field '{}' must be nullable when using AddColumnsOnly strategy",
                new_field.name()
            )));
        }
    }

    Ok(())
}

/// Validate an [`EvolutionStrategy::RenameColumns`] transition.
///
/// After applying the rename mapping the resulting set of fields must be
/// structurally identical to `new` (same names, same types).
fn validate_rename_columns(
    old: &Schema,
    new: &Schema,
    mapping: &HashMap<String, String>,
) -> Result<(), SchemaError> {
    // Build a renamed view of the old schema.
    let renamed_fields: Vec<Arc<Field>> = old
        .fields()
        .iter()
        .map(|f| {
            let new_name = mapping
                .get(f.name())
                .cloned()
                .unwrap_or_else(|| f.name().clone());
            Arc::new(Field::new(new_name, f.data_type().clone(), f.is_nullable()))
        })
        .collect();

    // Every field in `new` must appear in the renamed view with matching type.
    for new_field in new.fields() {
        let found = renamed_fields
            .iter()
            .find(|rf| rf.name() == new_field.name());
        match found {
            Some(rf) => {
                if !types_compatible(rf.data_type(), new_field.data_type()) {
                    return Err(SchemaError::Incompatible(format!(
                        "field '{}' changed type after rename",
                        new_field.name()
                    )));
                }
            }
            None => {
                // A brand-new field is only allowed if it is nullable
                // (AddColumnsOnly rule re-applied after renaming).
                if !new_field.is_nullable() {
                    return Err(SchemaError::Incompatible(format!(
                        "new non-nullable field '{}' not covered by rename mapping",
                        new_field.name()
                    )));
                }
            }
        }
    }

    // Every renamed field must still exist in `new`.
    for rf in &renamed_fields {
        if new.field_with_name(rf.name()).is_err() {
            return Err(SchemaError::Incompatible(format!(
                "renamed field '{}' is absent from the new schema",
                rf.name()
            )));
        }
    }

    Ok(())
}

/// Return `true` when the two [`DataType`]s are considered equivalent for the
/// purpose of read compatibility.  Currently requires exact equality; extend
/// this function for widening casts if needed.
fn types_compatible(writer: &DataType, reader: &DataType) -> bool {
    writer == reader
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn make_schema(fields: Vec<(&str, DataType, bool)>) -> Arc<Schema> {
        let arrow_fields: Vec<Field> = fields
            .into_iter()
            .map(|(name, dt, nullable)| Field::new(name, dt, nullable))
            .collect();
        Arc::new(Schema::new(arrow_fields))
    }

    // -----------------------------------------------------------------------
    // SchemaVersion fingerprint tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_schema_version_fingerprint_deterministic() {
        let schema = make_schema(vec![
            ("id", DataType::Int64, false),
            ("embedding", DataType::Float32, false),
        ]);
        let fp1 = SchemaVersion::fingerprint_of(&schema);
        let fp2 = SchemaVersion::fingerprint_of(&schema);
        assert_eq!(fp1, fp2, "fingerprint must be deterministic");
    }

    #[test]
    fn test_schema_version_different_schemas() {
        let schema_a = make_schema(vec![("id", DataType::Int64, false)]);
        let schema_b = make_schema(vec![("id", DataType::Int32, false)]);
        let fp_a = SchemaVersion::fingerprint_of(&schema_a);
        let fp_b = SchemaVersion::fingerprint_of(&schema_b);
        assert_ne!(
            fp_a, fp_b,
            "different schemas must produce different fingerprints"
        );
    }

    // -----------------------------------------------------------------------
    // SchemaRegistry: register and get
    // -----------------------------------------------------------------------

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = SchemaRegistry::new();
        let schema = make_schema(vec![("x", DataType::Float32, false)]);
        let sv = reg.register("model", schema.clone());
        assert_eq!(sv.name, "model");
        assert_eq!(sv.version, 1);
        let retrieved = reg.get(&sv).expect("schema should be retrievable");
        assert_eq!(retrieved.fields().len(), 1);
    }

    // -----------------------------------------------------------------------
    // SchemaRegistry: upgrade – add nullable column (compatible)
    // -----------------------------------------------------------------------

    #[test]
    fn test_registry_upgrade_add_column() {
        let mut reg = SchemaRegistry::new();
        let schema_v1 = make_schema(vec![("x", DataType::Float32, false)]);
        reg.register("model", schema_v1);

        let schema_v2 = make_schema(vec![
            ("x", DataType::Float32, false),
            ("bias", DataType::Float32, true), // nullable — compatible
        ]);
        let sv2 = reg
            .upgrade_schema("model", schema_v2)
            .expect("adding nullable column must succeed");
        assert_eq!(sv2.version, 2);
    }

    // -----------------------------------------------------------------------
    // SchemaRegistry: upgrade – incompatible (non-nullable new column)
    // -----------------------------------------------------------------------

    #[test]
    fn test_registry_upgrade_incompatible() {
        let mut reg = SchemaRegistry::new();
        let schema_v1 = make_schema(vec![("x", DataType::Float32, false)]);
        reg.register("model", schema_v1);

        // Removing an existing field is incompatible.
        let schema_bad = make_schema(vec![
            ("y", DataType::Float32, false), // "x" is gone — not allowed
        ]);
        let result = reg.upgrade_schema("model", schema_bad);
        assert!(
            matches!(result, Err(SchemaError::Incompatible(_))),
            "removing a field must be incompatible"
        );
    }

    // -----------------------------------------------------------------------
    // SchemaRegistry: can_read_with
    // -----------------------------------------------------------------------

    #[test]
    fn test_can_read_with() {
        let mut reg = SchemaRegistry::new();
        let schema_v1 = make_schema(vec![("x", DataType::Float32, false)]);
        let sv1 = reg.register("model", schema_v1);

        let schema_v2 = make_schema(vec![
            ("x", DataType::Float32, false),
            ("y", DataType::Float32, true),
        ]);
        let sv2 = reg
            .upgrade_schema("model", schema_v2)
            .expect("test: upgrade schema");

        // A reader at v2 can read data written by v1 (older writer).
        assert!(
            reg.can_read_with(&sv1, &sv2),
            "newer reader must be able to read older writer data"
        );

        // A reader at v1 cannot read data written by v2 (newer writer has extra field).
        assert!(
            !reg.can_read_with(&sv2, &sv1),
            "older reader must NOT be able to read newer writer data with extra fields"
        );
    }

    // -----------------------------------------------------------------------
    // SchemaEvolutionFrame: serialise / deserialise round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_schema_evolution_frame_roundtrip() {
        let mut reg = SchemaRegistry::new();
        let schema_v1 = make_schema(vec![("a", DataType::Int32, false)]);
        let sv1 = reg.register("tensors", schema_v1);

        let schema_v2 = make_schema(vec![
            ("a", DataType::Int32, false),
            ("b", DataType::Float64, true),
        ]);
        let sv2 = reg
            .upgrade_schema("tensors", schema_v2.clone())
            .expect("test: upgrade schema");

        let frame = SchemaEvolutionFrame::new("session-42", sv1.clone(), sv2.clone(), &schema_v2);

        let bytes = frame.to_bytes().expect("serialisation must succeed");
        let decoded =
            SchemaEvolutionFrame::from_bytes(&bytes).expect("deserialisation must succeed");

        assert_eq!(decoded.session_id, "session-42");
        assert_eq!(decoded.old_version, sv1);
        assert_eq!(decoded.new_version, sv2);

        let recovered = decoded.decode_schema().expect("IPC decode must succeed");
        assert_eq!(recovered.fields().len(), 2);
        assert_eq!(recovered.field(0).name(), "a");
        assert_eq!(recovered.field(1).name(), "b");
    }

    // -----------------------------------------------------------------------
    // latest_version
    // -----------------------------------------------------------------------

    #[test]
    fn test_latest_version() {
        let mut reg = SchemaRegistry::new();
        assert!(reg.latest_version("missing").is_none());

        let s = make_schema(vec![("v", DataType::Utf8, true)]);
        reg.register("ns", s.clone());
        let sv = reg.latest_version("ns").expect("test: get latest version");
        assert_eq!(sv.version, 1);

        let s2 = make_schema(vec![
            ("v", DataType::Utf8, true),
            ("w", DataType::Int8, true),
        ]);
        reg.upgrade_schema("ns", s2).expect("test: upgrade schema");
        let sv2 = reg.latest_version("ns").expect("test: get latest version");
        assert_eq!(sv2.version, 2);
    }

    // -----------------------------------------------------------------------
    // RenameColumns strategy
    // -----------------------------------------------------------------------

    #[test]
    fn test_upgrade_rename_columns() {
        let mut reg = SchemaRegistry::new();
        let schema_v1 = make_schema(vec![("old_name", DataType::Float32, false)]);
        reg.register("layer", schema_v1);

        let schema_v2 = make_schema(vec![("new_name", DataType::Float32, false)]);
        let mut map = HashMap::new();
        map.insert("old_name".to_owned(), "new_name".to_owned());
        let result = reg.upgrade_schema_with_strategy(
            "layer",
            schema_v2,
            &EvolutionStrategy::RenameColumns { mapping: map },
        );
        assert!(result.is_ok(), "valid rename must succeed: {result:?}");
    }

    // -----------------------------------------------------------------------
    // FullReplacement strategy
    // -----------------------------------------------------------------------

    #[test]
    fn test_upgrade_full_replacement() {
        let mut reg = SchemaRegistry::new();
        let schema_v1 = make_schema(vec![("a", DataType::Int64, false)]);
        reg.register("data", schema_v1);

        // Completely different schema — only allowed under FullReplacement.
        let schema_v2 = make_schema(vec![("z", DataType::Boolean, false)]);
        let result = reg.upgrade_schema_with_strategy(
            "data",
            schema_v2,
            &EvolutionStrategy::FullReplacement,
        );
        assert!(result.is_ok(), "FullReplacement must accept any new schema");
    }
}
