//! Online Arrow IPC schema migration for TensorSwap protocol.
//!
//! This module provides versioned schema migration: field-level evolution
//! operations can be applied to a live stream without reconnecting.
//!
//! # Overview
//!
//! - [`FieldMigration`] — atomic field-level change (add, drop, rename, nullability).
//! - [`SchemaMigration`] — an ordered list of [`FieldMigration`] steps that
//!   advance a schema from one version to the next.
//! - [`SchemaEvolutionManager`] — concurrent registry that applies migrations
//!   to the current version on demand.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

// ---------------------------------------------------------------------------
// FieldDefault
// ---------------------------------------------------------------------------

/// Default value to fill in when a new nullable or typed field is added.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum FieldDefault {
    /// Numeric zero (0 / 0.0 / false depending on target type).
    Zero,
    /// Numeric one (1 / 1.0 / true depending on target type).
    One,
    /// SQL-style NULL / Arrow null.
    Null,
    /// Arbitrary string literal (e.g. for dictionary/utf8 columns).
    StringValue(String),
}

// ---------------------------------------------------------------------------
// FieldMigration
// ---------------------------------------------------------------------------

/// A single field-level evolution operation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum FieldMigration {
    /// Append a new field to the schema.
    ///
    /// Existing records will have `default_value` substituted for the missing
    /// column.
    AddField {
        name: String,
        data_type: String,
        nullable: bool,
        default_value: FieldDefault,
    },

    /// Remove a field entirely; any data in that column is discarded.
    DropField { name: String },

    /// Rename a field in-place without changing its type or nullability.
    RenameField { from: String, to: String },

    /// Change the nullability of an existing field.
    SetNullable { name: String, nullable: bool },
}

// ---------------------------------------------------------------------------
// SchemaMigration
// ---------------------------------------------------------------------------

/// A versioned migration script: an ordered sequence of [`FieldMigration`]
/// operations that advance the schema from `from_version` to `to_version`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SchemaMigration {
    /// The schema version this migration starts from.
    pub from_version: u64,
    /// The schema version this migration produces.
    pub to_version: u64,
    /// Human-readable description of the migration.
    pub description: String,
    /// Ordered list of field-level operations to apply.
    pub operations: Vec<FieldMigration>,
}

impl SchemaMigration {
    /// Construct a new migration.
    pub fn new(
        from: u64,
        to: u64,
        description: impl Into<String>,
        ops: Vec<FieldMigration>,
    ) -> Self {
        Self {
            from_version: from,
            to_version: to,
            description: description.into(),
            operations: ops,
        }
    }

    /// Returns `true` when this migration moves the schema *forward* (i.e.
    /// `to_version > from_version`).
    pub fn is_forward(&self) -> bool {
        self.to_version > self.from_version
    }

    /// Collect the names of all fields that are *added* by this migration.
    pub fn field_names_added(&self) -> Vec<&str> {
        self.operations
            .iter()
            .filter_map(|op| {
                if let FieldMigration::AddField { name, .. } = op {
                    Some(name.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Collect the names of all fields that are *removed* by this migration.
    pub fn field_names_removed(&self) -> Vec<&str> {
        self.operations
            .iter()
            .filter_map(|op| {
                if let FieldMigration::DropField { name } = op {
                    Some(name.as_str())
                } else {
                    None
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// MigrationError
// ---------------------------------------------------------------------------

/// Errors returned by [`SchemaEvolutionManager`].
#[derive(Debug)]
pub enum MigrationError {
    /// There is no registered migration path between the two versions.
    NoPath { from: u64, to: u64 },
    /// The current version is already the requested target.
    AlreadyAtVersion { version: u64 },
    /// Downgrade requests are not supported.
    Downgrade { from: u64, to: u64 },
}

impl std::fmt::Display for MigrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MigrationError::NoPath { from, to } => {
                write!(f, "no migration path from version {from} to version {to}")
            }
            MigrationError::AlreadyAtVersion { version } => {
                write!(f, "schema is already at version {version}")
            }
            MigrationError::Downgrade { from, to } => {
                write!(f, "downgrade from version {from} to {to} is not supported")
            }
        }
    }
}

impl std::error::Error for MigrationError {}

// ---------------------------------------------------------------------------
// SchemaEvolutionManager
// ---------------------------------------------------------------------------

/// Thread-safe registry that applies versioned [`SchemaMigration`]s to an
/// ongoing stream without reconnecting.
///
/// # Concurrency
///
/// The migration map is protected by a [`parking_lot::RwLock`].  Version
/// counters use atomic operations so read-heavy workloads need no exclusive
/// locks.
pub struct SchemaEvolutionManager {
    /// from_version → migration to apply to advance one step.
    migrations: RwLock<HashMap<u64, SchemaMigration>>,
    /// The version this manager was last migrated to.
    current_version: AtomicU64,
    /// Cumulative count of successful migrations applied.
    applied_count: AtomicU64,
}

impl SchemaEvolutionManager {
    /// Create a new manager starting at `initial_version`.
    ///
    /// Returns an [`Arc`]-wrapped instance so it can be shared across tasks.
    pub fn new(initial_version: u64) -> Arc<Self> {
        Arc::new(Self {
            migrations: RwLock::new(HashMap::new()),
            current_version: AtomicU64::new(initial_version),
            applied_count: AtomicU64::new(0),
        })
    }

    /// Register a migration.
    ///
    /// Returns `false` if a migration for `migration.from_version` is already
    /// registered (duplicate registration is silently rejected).
    pub fn register_migration(&self, migration: SchemaMigration) -> bool {
        let mut map = self.migrations.write();
        if map.contains_key(&migration.from_version) {
            return false;
        }
        map.insert(migration.from_version, migration);
        true
    }

    /// Compute the ordered sequence of migrations needed to go from
    /// `from_version` to `to_version`.
    ///
    /// Only *forward* paths (strictly increasing version numbers) are
    /// supported.  Returns `None` when no complete path exists.
    pub fn migration_path(
        &self,
        from_version: u64,
        to_version: u64,
    ) -> Option<Vec<SchemaMigration>> {
        if from_version == to_version {
            return Some(Vec::new());
        }
        if from_version > to_version {
            return None;
        }

        let map = self.migrations.read();
        let mut path = Vec::new();
        let mut current = from_version;

        while current < to_version {
            let step = map.get(&current)?;
            if step.to_version <= current {
                // guard against non-advancing or backward step
                return None;
            }
            path.push(step.clone());
            current = step.to_version;
        }

        if current == to_version {
            Some(path)
        } else {
            None
        }
    }

    /// Apply all migrations necessary to reach `target_version`.
    ///
    /// # Errors
    ///
    /// - [`MigrationError::AlreadyAtVersion`] — current version equals target.
    /// - [`MigrationError::Downgrade`] — target is less than current version.
    /// - [`MigrationError::NoPath`] — no registered path leads to target.
    pub fn migrate_to(&self, target_version: u64) -> Result<Vec<SchemaMigration>, MigrationError> {
        let current = self.current_version.load(Ordering::Acquire);

        if current == target_version {
            return Err(MigrationError::AlreadyAtVersion {
                version: target_version,
            });
        }

        if target_version < current {
            return Err(MigrationError::Downgrade {
                from: current,
                to: target_version,
            });
        }

        let path = self
            .migration_path(current, target_version)
            .ok_or(MigrationError::NoPath {
                from: current,
                to: target_version,
            })?;

        let steps = path.len() as u64;

        // Advance current version atomically (single writer, so a plain store
        // after the read is safe here because migrate_to uses an exclusive
        // logical sequence).
        self.current_version
            .store(target_version, Ordering::Release);
        self.applied_count.fetch_add(steps, Ordering::Relaxed);

        Ok(path)
    }

    /// Return the current schema version.
    pub fn current_version(&self) -> u64 {
        self.current_version.load(Ordering::Acquire)
    }

    /// Return the total number of migrations applied since creation.
    pub fn applied_count(&self) -> u64 {
        self.applied_count.load(Ordering::Relaxed)
    }

    /// Return `true` when a migration path from `from_version` to
    /// `to_version` is available.
    pub fn can_migrate(&self, from_version: u64, to_version: u64) -> bool {
        self.migration_path(from_version, to_version).is_some()
    }

    /// Return all registered `from_version` values, in ascending order.
    pub fn registered_versions(&self) -> Vec<u64> {
        let map = self.migrations.read();
        let mut versions: Vec<u64> = map.keys().copied().collect();
        versions.sort_unstable();
        versions
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_add_op(name: &str) -> FieldMigration {
        FieldMigration::AddField {
            name: name.to_owned(),
            data_type: "Int64".to_owned(),
            nullable: true,
            default_value: FieldDefault::Null,
        }
    }

    fn make_drop_op(name: &str) -> FieldMigration {
        FieldMigration::DropField {
            name: name.to_owned(),
        }
    }

    fn simple_migration(from: u64, to: u64, op: FieldMigration) -> SchemaMigration {
        SchemaMigration::new(from, to, format!("v{from}->v{to}"), vec![op])
    }

    #[test]
    fn test_register_migration() {
        let mgr = SchemaEvolutionManager::new(1);
        let m = simple_migration(1, 2, make_add_op("score"));
        assert!(mgr.register_migration(m));
        assert_eq!(mgr.registered_versions(), vec![1]);
    }

    #[test]
    fn test_register_duplicate_returns_false() {
        let mgr = SchemaEvolutionManager::new(1);
        let m1 = simple_migration(1, 2, make_add_op("a"));
        let m2 = simple_migration(1, 3, make_add_op("b"));
        assert!(mgr.register_migration(m1));
        // Second registration for from_version=1 must be rejected.
        assert!(!mgr.register_migration(m2));
    }

    #[test]
    fn test_migration_path_direct() {
        let mgr = SchemaEvolutionManager::new(1);
        mgr.register_migration(simple_migration(1, 2, make_add_op("x")));

        let path = mgr.migration_path(1, 2).expect("path should exist");
        assert_eq!(path.len(), 1);
        assert_eq!(path[0].from_version, 1);
        assert_eq!(path[0].to_version, 2);
    }

    #[test]
    fn test_migration_path_chained() {
        let mgr = SchemaEvolutionManager::new(1);
        mgr.register_migration(simple_migration(1, 2, make_add_op("a")));
        mgr.register_migration(simple_migration(2, 3, make_add_op("b")));

        let path = mgr.migration_path(1, 3).expect("chained path should exist");
        assert_eq!(path.len(), 2);
        assert_eq!(path[0].from_version, 1);
        assert_eq!(path[1].from_version, 2);
    }

    #[test]
    fn test_migration_path_missing() {
        let mgr = SchemaEvolutionManager::new(1);
        mgr.register_migration(simple_migration(1, 2, make_add_op("a")));
        // No migration registered for v2→v3
        assert!(mgr.migration_path(1, 3).is_none());
    }

    #[test]
    fn test_migrate_to_success() {
        let mgr = SchemaEvolutionManager::new(1);
        mgr.register_migration(simple_migration(1, 2, make_add_op("score")));

        let applied = mgr.migrate_to(2).expect("migration should succeed");
        assert_eq!(applied.len(), 1);
        assert_eq!(mgr.current_version(), 2);
    }

    #[test]
    fn test_migrate_to_already_at_version() {
        let mgr = SchemaEvolutionManager::new(2);
        let err = mgr.migrate_to(2).expect_err("should fail");
        assert!(matches!(
            err,
            MigrationError::AlreadyAtVersion { version: 2 }
        ));
    }

    #[test]
    fn test_migrate_to_downgrade() {
        let mgr = SchemaEvolutionManager::new(5);
        let err = mgr.migrate_to(3).expect_err("downgrade should fail");
        assert!(matches!(err, MigrationError::Downgrade { from: 5, to: 3 }));
    }

    #[test]
    fn test_migrate_increments_applied_count() {
        let mgr = SchemaEvolutionManager::new(1);
        mgr.register_migration(simple_migration(1, 2, make_add_op("a")));
        mgr.register_migration(simple_migration(2, 3, make_add_op("b")));

        assert_eq!(mgr.applied_count(), 0);
        mgr.migrate_to(3).expect("ok");
        assert_eq!(mgr.applied_count(), 2);
    }

    #[test]
    fn test_can_migrate() {
        let mgr = SchemaEvolutionManager::new(1);
        mgr.register_migration(simple_migration(1, 2, make_add_op("x")));

        assert!(mgr.can_migrate(1, 2));
        assert!(!mgr.can_migrate(1, 3));
        assert!(!mgr.can_migrate(2, 1)); // backward
    }

    #[test]
    fn test_field_names_added_removed() {
        let ops = vec![
            make_add_op("new_col"),
            make_drop_op("old_col"),
            FieldMigration::RenameField {
                from: "alpha".to_owned(),
                to: "beta".to_owned(),
            },
        ];
        let m = SchemaMigration::new(1, 2, "mixed", ops);
        assert_eq!(m.field_names_added(), vec!["new_col"]);
        assert_eq!(m.field_names_removed(), vec!["old_col"]);
    }

    #[test]
    fn test_registered_versions() {
        let mgr = SchemaEvolutionManager::new(1);
        mgr.register_migration(simple_migration(3, 4, make_add_op("z")));
        mgr.register_migration(simple_migration(1, 2, make_add_op("a")));
        mgr.register_migration(simple_migration(2, 3, make_add_op("b")));

        assert_eq!(mgr.registered_versions(), vec![1, 2, 3]);
    }

    #[test]
    fn test_is_forward() {
        let forward = SchemaMigration::new(1, 2, "fwd", vec![]);
        assert!(forward.is_forward());
        let same = SchemaMigration::new(2, 2, "same", vec![]);
        assert!(!same.is_forward());
    }
}
