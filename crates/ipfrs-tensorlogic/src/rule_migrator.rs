//! Rule version migration for TensorLogic rule sets.
//!
//! Provides the infrastructure for migrating rule set schemas between declared
//! versions (`V1`, `V2`, `V3`), applying ordered transformation steps, and
//! validating compatibility in both upgrade and downgrade directions.
//!
//! # Overview
//!
//! ```
//! use ipfrs_tensorlogic::rule_migrator::{
//!     MigrationStep, MigrationTransform, RuleSchemaVersion, RuleVersionMigrator,
//! };
//!
//! let mut migrator = RuleVersionMigrator::new();
//!
//! migrator.register_step(MigrationStep {
//!     from_version: RuleSchemaVersion::V1,
//!     to_version:   RuleSchemaVersion::V2,
//!     transforms: vec![
//!         MigrationTransform::RenameField {
//!             from: "head".to_string(),
//!             to:   "conclusion".to_string(),
//!         },
//!     ],
//!     description: "Rename head → conclusion".to_string(),
//! });
//!
//! let result = migrator.migrate(RuleSchemaVersion::V1, RuleSchemaVersion::V2);
//! assert!(result.is_success());
//! ```

// ---------------------------------------------------------------------------
// RuleSchemaVersion
// ---------------------------------------------------------------------------

/// Declared schema versions for TensorLogic rule sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RuleSchemaVersion {
    /// Initial rule-set schema.
    V1 = 1,
    /// Second-generation schema (renamed fields, added metadata).
    V2 = 2,
    /// Third-generation schema (type system overhaul).
    V3 = 3,
}

// ---------------------------------------------------------------------------
// MigrationTransform
// ---------------------------------------------------------------------------

/// An atomic transformation applied to a rule set during migration.
#[derive(Clone, Debug, PartialEq)]
pub enum MigrationTransform {
    /// Rename a field across all rules in the set.
    RenameField {
        /// Original field name.
        from: String,
        /// Target field name.
        to: String,
    },
    /// Insert a new field with a default value into all rules.
    AddField {
        /// Name of the new field.
        name: String,
        /// Serialised default value for the field.
        default_value: String,
    },
    /// Remove a field from all rules (data may be lost — lossy).
    RemoveField {
        /// Name of the field to remove.
        name: String,
    },
    /// Convert the type of an existing field (may be lossy).
    ConvertType {
        /// Field whose type is changed.
        field: String,
        /// Source type descriptor (informational).
        from_type: String,
        /// Destination type descriptor (informational).
        to_type: String,
    },
}

// ---------------------------------------------------------------------------
// MigrationStep
// ---------------------------------------------------------------------------

/// A single, directed migration step between two adjacent (or distant) schema
/// versions, carrying the ordered list of transformations to apply.
#[derive(Clone, Debug, PartialEq)]
pub struct MigrationStep {
    /// The version the rule set must currently be at.
    pub from_version: RuleSchemaVersion,
    /// The version the rule set will be at after this step completes.
    pub to_version: RuleSchemaVersion,
    /// Ordered list of transformations applied during this step.
    pub transforms: Vec<MigrationTransform>,
    /// Human-readable description of the migration step.
    pub description: String,
}

// ---------------------------------------------------------------------------
// MigrationPlan
// ---------------------------------------------------------------------------

/// An ordered sequence of migration steps that connects a source version to a
/// target version.
#[derive(Clone, Debug, PartialEq)]
pub struct MigrationPlan {
    /// Ordered path from source to target version.
    pub steps: Vec<MigrationStep>,
}

impl MigrationPlan {
    /// Total number of transforms across all steps in this plan.
    pub fn total_transforms(&self) -> usize {
        self.steps.iter().map(|s| s.transforms.len()).sum()
    }

    /// Returns `true` when the plan has no steps (source == target).
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

// ---------------------------------------------------------------------------
// MigrationResult
// ---------------------------------------------------------------------------

/// The outcome of executing a [`MigrationPlan`].
#[derive(Clone, Debug, PartialEq)]
pub struct MigrationResult {
    /// Number of migration steps that were applied.
    pub applied_steps: usize,
    /// Total number of transforms executed across all applied steps.
    pub transforms_applied: usize,
    /// Diagnostic warnings produced during migration (e.g. lossy type
    /// conversions, downgrade data loss notices).
    pub warnings: Vec<String>,
}

impl MigrationResult {
    /// Returns `true` when the migration is considered successful.
    ///
    /// Success means either:
    /// - at least one step was applied, or
    /// - the source was already at the target version (zero steps needed).
    pub fn is_success(&self) -> bool {
        self.applied_steps > 0 || self.warnings.iter().all(|w| !w.starts_with("FATAL"))
    }
}

// ---------------------------------------------------------------------------
// RuleVersionMigrator
// ---------------------------------------------------------------------------

/// Registry and executor for rule-set schema migrations.
///
/// Steps are registered individually and queried on demand to build a
/// [`MigrationPlan`].  Migration is simulated (no actual rule-set bytes are
/// transformed; the struct tracks counts and emits warnings for lossy
/// operations).
pub struct RuleVersionMigrator {
    /// All registered migration steps.
    steps: Vec<MigrationStep>,
}

impl RuleVersionMigrator {
    /// Create an empty migrator with no registered steps.
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Register a migration step.
    ///
    /// Steps may be registered in any order; `plan` will sort and chain them
    /// as required.
    pub fn register_step(&mut self, step: MigrationStep) {
        self.steps.push(step);
    }

    /// Build an ordered migration plan from `from` to `to`.
    ///
    /// * If `from == to`: returns `Some(MigrationPlan { steps: vec![] })`.
    /// * If `from < to` (upgrade): attempts to find a sequential chain of
    ///   registered steps that covers every version hop between `from` and
    ///   `to`.
    /// * If `from > to` (downgrade): attempts the same but in reverse; a
    ///   downgrade warning is embedded in the first step's description (the
    ///   warning is surfaced as a `MigrationResult::warnings` entry when
    ///   `migrate` is called).
    /// * Returns `None` when no complete path exists.
    pub fn plan(&self, from: RuleSchemaVersion, to: RuleSchemaVersion) -> Option<MigrationPlan> {
        if from == to {
            return Some(MigrationPlan { steps: vec![] });
        }

        if from < to {
            // Upgrade path: collect steps in ascending order.
            let path = self.find_path(from, to, false)?;
            Some(MigrationPlan { steps: path })
        } else {
            // Downgrade path: collect steps in descending order.
            let path = self.find_path(from, to, true)?;
            Some(MigrationPlan { steps: path })
        }
    }

    /// Find an ordered chain of steps from `from` to `to`.
    ///
    /// When `reverse` is `true` the search walks in the downgrade direction
    /// (higher → lower version numbers).
    fn find_path(
        &self,
        from: RuleSchemaVersion,
        to: RuleSchemaVersion,
        reverse: bool,
    ) -> Option<Vec<MigrationStep>> {
        // Enumerate every version in the ordered range between `from` and `to`.
        // All versions in the enum, sorted ascending.
        const ALL_VERSIONS: [RuleSchemaVersion; 3] = [
            RuleSchemaVersion::V1,
            RuleSchemaVersion::V2,
            RuleSchemaVersion::V3,
        ];

        // Build the sequence of (current, next) version hops we need to cover.
        let hops: Vec<(RuleSchemaVersion, RuleSchemaVersion)> = if !reverse {
            // Upgrade: walk ascending from `from` to `to`.
            ALL_VERSIONS
                .windows(2)
                .filter_map(|w| {
                    let (a, b) = (w[0], w[1]);
                    if a >= from && b <= to {
                        Some((a, b))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            // Downgrade: walk descending from `from` to `to`.
            let mut desc: Vec<(RuleSchemaVersion, RuleSchemaVersion)> = ALL_VERSIONS
                .windows(2)
                .filter_map(|w| {
                    let (a, b) = (w[0], w[1]);
                    // In downgrade mode we need steps that go b→a (high→low).
                    // We include the pair when a >= to and b <= from.
                    if a >= to && b <= from {
                        Some((b, a)) // reversed direction
                    } else {
                        None
                    }
                })
                .collect();
            desc.reverse(); // highest-to-lowest order
            desc
        };

        if hops.is_empty() {
            return None;
        }

        // For each hop, locate the first registered step that covers it.
        let mut plan_steps: Vec<MigrationStep> = Vec::with_capacity(hops.len());
        for (hop_from, hop_to) in hops {
            let found = self
                .steps
                .iter()
                .find(|s| s.from_version == hop_from && s.to_version == hop_to)
                .cloned();

            match found {
                Some(step) => plan_steps.push(step),
                None => return None, // path is broken
            }
        }

        Some(plan_steps)
    }

    /// Execute a migration from `from` to `to` and return the result.
    ///
    /// The migration is *simulated*: no actual rule-set data is mutated.  The
    /// result carries counts of applied steps and transforms, plus any
    /// warnings generated (e.g. for `ConvertType` or downgrade operations).
    pub fn migrate(&self, from: RuleSchemaVersion, to: RuleSchemaVersion) -> MigrationResult {
        let plan = match self.plan(from, to) {
            Some(p) => p,
            None => {
                return MigrationResult {
                    applied_steps: 0,
                    transforms_applied: 0,
                    warnings: vec![format!(
                        "No migration path found from {:?} to {:?}",
                        from, to
                    )],
                }
            }
        };

        // Same-version: trivially successful with no work done.
        if plan.is_empty() {
            return MigrationResult {
                applied_steps: 0,
                transforms_applied: 0,
                warnings: vec![],
            };
        }

        let mut warnings: Vec<String> = Vec::new();

        // Emit a global downgrade notice when migrating to an older version.
        if from > to {
            warnings.push(format!(
                "Downgrade from {:?} to {:?}: migration may be lossy — fields removed in newer \
                 versions cannot be recovered",
                from, to
            ));
        }

        let mut transforms_applied: usize = 0;

        for step in &plan.steps {
            for transform in &step.transforms {
                match transform {
                    MigrationTransform::ConvertType {
                        field,
                        from_type,
                        to_type,
                    } => {
                        warnings.push(format!(
                            "ConvertType on field '{}': '{}' → '{}' may be lossy",
                            field, from_type, to_type
                        ));
                    }
                    MigrationTransform::RemoveField { name } => {
                        warnings.push(format!(
                            "RemoveField '{}': data will be permanently discarded",
                            name
                        ));
                    }
                    MigrationTransform::RenameField { .. }
                    | MigrationTransform::AddField { .. } => {
                        // Non-lossy transforms; no warning required.
                    }
                }
                transforms_applied += 1;
            }
        }

        MigrationResult {
            applied_steps: plan.steps.len(),
            transforms_applied,
            warnings,
        }
    }

    /// Return all registered (from, to) version pairs.
    pub fn registered_paths(&self) -> Vec<(RuleSchemaVersion, RuleSchemaVersion)> {
        self.steps
            .iter()
            .map(|s| (s.from_version, s.to_version))
            .collect()
    }
}

impl Default for RuleVersionMigrator {
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

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_v1_v2_step() -> MigrationStep {
        MigrationStep {
            from_version: RuleSchemaVersion::V1,
            to_version: RuleSchemaVersion::V2,
            transforms: vec![
                MigrationTransform::RenameField {
                    from: "head".to_string(),
                    to: "conclusion".to_string(),
                },
                MigrationTransform::AddField {
                    name: "metadata".to_string(),
                    default_value: "{}".to_string(),
                },
            ],
            description: "V1 → V2: rename head, add metadata".to_string(),
        }
    }

    fn make_v2_v3_step() -> MigrationStep {
        MigrationStep {
            from_version: RuleSchemaVersion::V2,
            to_version: RuleSchemaVersion::V3,
            transforms: vec![
                MigrationTransform::ConvertType {
                    field: "priority".to_string(),
                    from_type: "i32".to_string(),
                    to_type: "f64".to_string(),
                },
                MigrationTransform::RemoveField {
                    name: "legacy_tag".to_string(),
                },
            ],
            description: "V2 → V3: convert priority type, remove legacy_tag".to_string(),
        }
    }

    fn make_v3_v2_step() -> MigrationStep {
        MigrationStep {
            from_version: RuleSchemaVersion::V3,
            to_version: RuleSchemaVersion::V2,
            transforms: vec![MigrationTransform::ConvertType {
                field: "priority".to_string(),
                from_type: "f64".to_string(),
                to_type: "i32".to_string(),
            }],
            description: "V3 → V2 downgrade: convert priority back".to_string(),
        }
    }

    fn make_v2_v1_step() -> MigrationStep {
        MigrationStep {
            from_version: RuleSchemaVersion::V2,
            to_version: RuleSchemaVersion::V1,
            transforms: vec![MigrationTransform::RenameField {
                from: "conclusion".to_string(),
                to: "head".to_string(),
            }],
            description: "V2 → V1 downgrade: rename conclusion back".to_string(),
        }
    }

    fn migrator_with_upgrade_steps() -> RuleVersionMigrator {
        let mut m = RuleVersionMigrator::new();
        m.register_step(make_v1_v2_step());
        m.register_step(make_v2_v3_step());
        m
    }

    // -----------------------------------------------------------------------
    // 1. plan — same version returns empty plan
    // -----------------------------------------------------------------------

    #[test]
    fn test_plan_same_version_is_empty() {
        let m = migrator_with_upgrade_steps();
        let plan = m
            .plan(RuleSchemaVersion::V1, RuleSchemaVersion::V1)
            .expect("test: should succeed");
        assert!(plan.is_empty());
        assert_eq!(plan.total_transforms(), 0);
    }

    #[test]
    fn test_plan_same_version_v2() {
        let m = migrator_with_upgrade_steps();
        let plan = m
            .plan(RuleSchemaVersion::V2, RuleSchemaVersion::V2)
            .expect("test: should succeed");
        assert!(plan.is_empty());
    }

    #[test]
    fn test_plan_same_version_v3() {
        let m = migrator_with_upgrade_steps();
        let plan = m
            .plan(RuleSchemaVersion::V3, RuleSchemaVersion::V3)
            .expect("test: should succeed");
        assert!(plan.is_empty());
    }

    // -----------------------------------------------------------------------
    // 2. plan — V1 → V2
    // -----------------------------------------------------------------------

    #[test]
    fn test_plan_v1_to_v2() {
        let m = migrator_with_upgrade_steps();
        let plan = m
            .plan(RuleSchemaVersion::V1, RuleSchemaVersion::V2)
            .expect("test: should succeed");
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].from_version, RuleSchemaVersion::V1);
        assert_eq!(plan.steps[0].to_version, RuleSchemaVersion::V2);
        assert_eq!(plan.total_transforms(), 2);
    }

    // -----------------------------------------------------------------------
    // 3. plan — V1 → V3 via V2
    // -----------------------------------------------------------------------

    #[test]
    fn test_plan_v1_to_v3_via_v2() {
        let m = migrator_with_upgrade_steps();
        let plan = m
            .plan(RuleSchemaVersion::V1, RuleSchemaVersion::V3)
            .expect("test: should succeed");
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.steps[0].from_version, RuleSchemaVersion::V1);
        assert_eq!(plan.steps[0].to_version, RuleSchemaVersion::V2);
        assert_eq!(plan.steps[1].from_version, RuleSchemaVersion::V2);
        assert_eq!(plan.steps[1].to_version, RuleSchemaVersion::V3);
        // 2 transforms in V1→V2, 2 transforms in V2→V3
        assert_eq!(plan.total_transforms(), 4);
    }

    // -----------------------------------------------------------------------
    // 4. plan — missing step returns None
    // -----------------------------------------------------------------------

    #[test]
    fn test_plan_missing_step_returns_none() {
        // Only V1→V2 is registered; V2→V3 is absent.
        let mut m = RuleVersionMigrator::new();
        m.register_step(make_v1_v2_step());
        let plan = m.plan(RuleSchemaVersion::V1, RuleSchemaVersion::V3);
        assert!(plan.is_none());
    }

    #[test]
    fn test_plan_empty_migrator_returns_none_for_upgrade() {
        let m = RuleVersionMigrator::new();
        assert!(m
            .plan(RuleSchemaVersion::V1, RuleSchemaVersion::V2)
            .is_none());
    }

    // -----------------------------------------------------------------------
    // 5. plan — downgrade V3 → V1
    // -----------------------------------------------------------------------

    #[test]
    fn test_plan_downgrade_v3_to_v1() {
        let mut m = RuleVersionMigrator::new();
        m.register_step(make_v3_v2_step());
        m.register_step(make_v2_v1_step());

        let plan = m
            .plan(RuleSchemaVersion::V3, RuleSchemaVersion::V1)
            .expect("test: should succeed");
        assert_eq!(plan.steps.len(), 2);
        // First hop: V3→V2, second hop: V2→V1
        assert_eq!(plan.steps[0].from_version, RuleSchemaVersion::V3);
        assert_eq!(plan.steps[0].to_version, RuleSchemaVersion::V2);
        assert_eq!(plan.steps[1].from_version, RuleSchemaVersion::V2);
        assert_eq!(plan.steps[1].to_version, RuleSchemaVersion::V1);
    }

    #[test]
    fn test_plan_downgrade_missing_step_returns_none() {
        let mut m = RuleVersionMigrator::new();
        // Only V3→V2 is registered; V2→V1 is absent.
        m.register_step(make_v3_v2_step());
        let plan = m.plan(RuleSchemaVersion::V3, RuleSchemaVersion::V1);
        assert!(plan.is_none());
    }

    // -----------------------------------------------------------------------
    // 6. register_step
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_step_increases_registered_paths() {
        let mut m = RuleVersionMigrator::new();
        assert!(m.registered_paths().is_empty());

        m.register_step(make_v1_v2_step());
        assert_eq!(m.registered_paths().len(), 1);

        m.register_step(make_v2_v3_step());
        assert_eq!(m.registered_paths().len(), 2);
    }

    // -----------------------------------------------------------------------
    // 7. registered_paths
    // -----------------------------------------------------------------------

    #[test]
    fn test_registered_paths_content() {
        let m = migrator_with_upgrade_steps();
        let paths = m.registered_paths();
        assert!(paths.contains(&(RuleSchemaVersion::V1, RuleSchemaVersion::V2)));
        assert!(paths.contains(&(RuleSchemaVersion::V2, RuleSchemaVersion::V3)));
    }

    // -----------------------------------------------------------------------
    // 8. migrate — total_transforms reflected in result
    // -----------------------------------------------------------------------

    #[test]
    fn test_migrate_total_transforms() {
        let m = migrator_with_upgrade_steps();
        let result = m.migrate(RuleSchemaVersion::V1, RuleSchemaVersion::V3);
        // V1→V2 has 2, V2→V3 has 2 → total 4
        assert_eq!(result.transforms_applied, 4);
        assert_eq!(result.applied_steps, 2);
    }

    // -----------------------------------------------------------------------
    // 9. migrate — ConvertType emits warning
    // -----------------------------------------------------------------------

    #[test]
    fn test_migrate_convert_type_emits_warning() {
        let m = migrator_with_upgrade_steps();
        // V2→V3 contains a ConvertType transform.
        let result = m.migrate(RuleSchemaVersion::V2, RuleSchemaVersion::V3);
        let has_convert_warning = result
            .warnings
            .iter()
            .any(|w| w.contains("ConvertType") && w.contains("priority"));
        assert!(
            has_convert_warning,
            "Expected ConvertType warning, got: {:?}",
            result.warnings
        );
    }

    // -----------------------------------------------------------------------
    // 10. migrate — RemoveField emits warning
    // -----------------------------------------------------------------------

    #[test]
    fn test_migrate_remove_field_emits_warning() {
        let m = migrator_with_upgrade_steps();
        let result = m.migrate(RuleSchemaVersion::V2, RuleSchemaVersion::V3);
        let has_remove_warning = result
            .warnings
            .iter()
            .any(|w| w.contains("RemoveField") && w.contains("legacy_tag"));
        assert!(
            has_remove_warning,
            "Expected RemoveField warning, got: {:?}",
            result.warnings
        );
    }

    // -----------------------------------------------------------------------
    // 11. migrate — downgrade emits downgrade warning
    // -----------------------------------------------------------------------

    #[test]
    fn test_migrate_downgrade_emits_warning() {
        let mut m = RuleVersionMigrator::new();
        m.register_step(make_v3_v2_step());
        m.register_step(make_v2_v1_step());

        let result = m.migrate(RuleSchemaVersion::V3, RuleSchemaVersion::V1);
        let has_downgrade = result
            .warnings
            .iter()
            .any(|w| w.to_lowercase().contains("downgrade"));
        assert!(
            has_downgrade,
            "Expected downgrade warning, got: {:?}",
            result.warnings
        );
    }

    // -----------------------------------------------------------------------
    // 12. MigrationPlan::total_transforms
    // -----------------------------------------------------------------------

    #[test]
    fn test_migration_plan_total_transforms() {
        let step1 = make_v1_v2_step(); // 2 transforms
        let step2 = make_v2_v3_step(); // 2 transforms
        let plan = MigrationPlan {
            steps: vec![step1, step2],
        };
        assert_eq!(plan.total_transforms(), 4);
    }

    // -----------------------------------------------------------------------
    // 13. MigrationPlan::is_empty
    // -----------------------------------------------------------------------

    #[test]
    fn test_migration_plan_is_empty_true() {
        let plan = MigrationPlan { steps: vec![] };
        assert!(plan.is_empty());
    }

    #[test]
    fn test_migration_plan_is_empty_false() {
        let plan = MigrationPlan {
            steps: vec![make_v1_v2_step()],
        };
        assert!(!plan.is_empty());
    }

    // -----------------------------------------------------------------------
    // 14. MigrationResult::is_success — true cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_migration_result_is_success_with_steps() {
        let result = MigrationResult {
            applied_steps: 1,
            transforms_applied: 2,
            warnings: vec![],
        };
        assert!(result.is_success());
    }

    #[test]
    fn test_migration_result_is_success_same_version() {
        // Same-version: 0 steps applied, no warnings → still success.
        let m = migrator_with_upgrade_steps();
        let result = m.migrate(RuleSchemaVersion::V2, RuleSchemaVersion::V2);
        assert!(result.is_success());
        assert_eq!(result.applied_steps, 0);
    }

    // -----------------------------------------------------------------------
    // 15. MigrationResult::is_success — false case (no plan)
    // -----------------------------------------------------------------------

    #[test]
    fn test_migration_result_no_plan_has_zero_steps() {
        let m = RuleVersionMigrator::new();
        // No steps registered → plan returns None → applied_steps == 0
        let result = m.migrate(RuleSchemaVersion::V1, RuleSchemaVersion::V3);
        assert_eq!(result.applied_steps, 0);
        assert!(
            !result.warnings.is_empty(),
            "should warn about missing path"
        );
    }

    // -----------------------------------------------------------------------
    // 16. migrate — step-by-step V1→V2 accuracy
    // -----------------------------------------------------------------------

    #[test]
    fn test_migrate_v1_to_v2_single_step() {
        let m = migrator_with_upgrade_steps();
        let result = m.migrate(RuleSchemaVersion::V1, RuleSchemaVersion::V2);
        assert_eq!(result.applied_steps, 1);
        assert_eq!(result.transforms_applied, 2);
        // RenameField + AddField are non-lossy → no warnings
        assert!(result.warnings.is_empty());
    }

    // -----------------------------------------------------------------------
    // 17. RuleSchemaVersion ordering
    // -----------------------------------------------------------------------

    #[test]
    fn test_schema_version_ordering() {
        assert!(RuleSchemaVersion::V1 < RuleSchemaVersion::V2);
        assert!(RuleSchemaVersion::V2 < RuleSchemaVersion::V3);
        assert!(RuleSchemaVersion::V1 < RuleSchemaVersion::V3);
        assert_eq!(RuleSchemaVersion::V2, RuleSchemaVersion::V2);
    }

    // -----------------------------------------------------------------------
    // 18. Duplicate registration does not break planning
    // -----------------------------------------------------------------------

    #[test]
    fn test_duplicate_step_registration_uses_first_match() {
        let mut m = RuleVersionMigrator::new();
        m.register_step(make_v1_v2_step());
        // Register a second (different) V1→V2 step — planner should use the first.
        m.register_step(MigrationStep {
            from_version: RuleSchemaVersion::V1,
            to_version: RuleSchemaVersion::V2,
            transforms: vec![],
            description: "duplicate V1→V2".to_string(),
        });
        let plan = m
            .plan(RuleSchemaVersion::V1, RuleSchemaVersion::V2)
            .expect("test: should succeed");
        assert_eq!(plan.steps.len(), 1);
        // First registered step is used.
        assert!(!plan.steps[0].description.contains("duplicate"));
    }
}
