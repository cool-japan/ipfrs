//! TensorDiffEngine — structural and numeric diff between tensor snapshots.
//!
//! Used for change-detection in federated learning checkpoints: given two sets of
//! named tensors (old and new), compute per-tensor diffs covering additions,
//! removals, shape changes, and numeric value changes.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// DiffKind
// ---------------------------------------------------------------------------

/// Describes what changed (or didn't) for a single named tensor.
#[derive(Debug, Clone, PartialEq)]
pub enum DiffKind {
    /// Key present in new snapshot, absent in old.
    Added,
    /// Key present in old snapshot, absent in new.
    Removed,
    /// Shapes differ between old and new.
    ShapeChanged {
        old_shape: Vec<usize>,
        new_shape: Vec<usize>,
    },
    /// Shapes match but numeric values differ beyond threshold.
    ValueChanged {
        max_abs_diff: f32,
        mean_abs_diff: f32,
        changed_elements: usize,
    },
    /// Shapes match and all values are within threshold.
    Unchanged,
}

// ---------------------------------------------------------------------------
// TensorSnapshot
// ---------------------------------------------------------------------------

/// A named, flat snapshot of a single tensor.
#[derive(Debug, Clone)]
pub struct TensorSnapshot {
    /// Tensor name / checkpoint key.
    pub name: String,
    /// Shape dimensions, e.g. `[128, 256]` for a 2-D tensor.
    pub shape: Vec<usize>,
    /// Flattened row-major element data.
    pub data: Vec<f32>,
}

impl TensorSnapshot {
    /// Create a new snapshot.
    pub fn new(name: impl Into<String>, shape: Vec<usize>, data: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            shape,
            data,
        }
    }

    /// Number of elements: product of shape dimensions, or `data.len()` when
    /// `shape` is empty (scalar / rank-0 tensors).
    pub fn numel(&self) -> usize {
        if self.shape.is_empty() {
            self.data.len()
        } else {
            self.shape.iter().product()
        }
    }

    /// Returns `true` when `self` and `other` have identical shapes and can
    /// therefore be compared element-wise.
    pub fn is_compatible(&self, other: &TensorSnapshot) -> bool {
        self.shape == other.shape
    }
}

// ---------------------------------------------------------------------------
// TensorDiff
// ---------------------------------------------------------------------------

/// Diff result for one named tensor.
#[derive(Debug, Clone)]
pub struct TensorDiff {
    /// Tensor name.
    pub name: String,
    /// Nature of the change.
    pub kind: DiffKind,
}

impl TensorDiff {
    /// Returns `true` when this diff is considered "significant" relative to
    /// `threshold`.
    ///
    /// - `ValueChanged` → significant when `max_abs_diff > threshold`
    /// - `ShapeChanged`, `Added`, `Removed` → always significant
    /// - `Unchanged` → never significant
    pub fn is_significant(&self, threshold: f32) -> bool {
        match &self.kind {
            DiffKind::ValueChanged { max_abs_diff, .. } => *max_abs_diff > threshold,
            DiffKind::ShapeChanged { .. } | DiffKind::Added | DiffKind::Removed => true,
            DiffKind::Unchanged => false,
        }
    }
}

// ---------------------------------------------------------------------------
// DiffSummary
// ---------------------------------------------------------------------------

/// Aggregated statistics over a slice of [`TensorDiff`] values.
#[derive(Debug, Clone, Default)]
pub struct DiffSummary {
    pub added: usize,
    pub removed: usize,
    pub shape_changed: usize,
    pub value_changed: usize,
    pub unchanged: usize,
    pub total_changed_elements: usize,
}

impl DiffSummary {
    /// Returns `true` when at least one tensor was added, removed, or changed.
    pub fn has_changes(&self) -> bool {
        self.added > 0 || self.removed > 0 || self.shape_changed > 0 || self.value_changed > 0
    }
}

// ---------------------------------------------------------------------------
// TensorDiffEngine
// ---------------------------------------------------------------------------

/// Engine that computes structural and numeric diffs between tensor snapshots.
pub struct TensorDiffEngine {
    /// Element-wise absolute difference threshold below which a value is
    /// considered unchanged.
    pub threshold: f32,
}

impl TensorDiffEngine {
    /// Create a new engine with the given numeric threshold.
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Diff two individual tensors that share the same name.
    ///
    /// Shape mismatch → `ShapeChanged`.
    /// Element-wise: count elements where `|new[i] - old[i]| > threshold`,
    /// accumulate max and sum.  If count == 0 → `Unchanged`, else `ValueChanged`.
    pub fn diff_tensors(&self, old: &TensorSnapshot, new: &TensorSnapshot) -> TensorDiff {
        if old.shape != new.shape {
            return TensorDiff {
                name: new.name.clone(),
                kind: DiffKind::ShapeChanged {
                    old_shape: old.shape.clone(),
                    new_shape: new.shape.clone(),
                },
            };
        }

        let len = old.data.len().max(new.data.len());
        let mut max_abs: f32 = 0.0;
        let mut sum_abs: f32 = 0.0;
        let mut changed: usize = 0;

        for i in 0..len {
            let o = old.data.get(i).copied().unwrap_or(0.0);
            let n = new.data.get(i).copied().unwrap_or(0.0);
            let d = (n - o).abs();
            if d > max_abs {
                max_abs = d;
            }
            sum_abs += d;
            if d > self.threshold {
                changed += 1;
            }
        }

        let kind = if changed == 0 {
            DiffKind::Unchanged
        } else {
            let mean_abs = if len > 0 { sum_abs / len as f32 } else { 0.0 };
            DiffKind::ValueChanged {
                max_abs_diff: max_abs,
                mean_abs_diff: mean_abs,
                changed_elements: changed,
            }
        };

        TensorDiff {
            name: new.name.clone(),
            kind,
        }
    }

    /// Diff two full checkpoint snapshots (slices of tensors).
    ///
    /// Results are sorted by name for deterministic output.
    pub fn diff_snapshots(
        &self,
        old_set: &[TensorSnapshot],
        new_set: &[TensorSnapshot],
    ) -> Vec<TensorDiff> {
        let old_map: HashMap<&str, &TensorSnapshot> =
            old_set.iter().map(|t| (t.name.as_str(), t)).collect();
        let new_map: HashMap<&str, &TensorSnapshot> =
            new_set.iter().map(|t| (t.name.as_str(), t)).collect();

        let mut diffs: Vec<TensorDiff> = Vec::new();

        // Process tensors present in old
        for (name, old_tensor) in &old_map {
            match new_map.get(name) {
                None => diffs.push(TensorDiff {
                    name: (*name).to_string(),
                    kind: DiffKind::Removed,
                }),
                Some(new_tensor) => {
                    diffs.push(self.diff_tensors(old_tensor, new_tensor));
                }
            }
        }

        // Tensors only in new
        for name in new_map.keys() {
            if !old_map.contains_key(name) {
                diffs.push(TensorDiff {
                    name: (*name).to_string(),
                    kind: DiffKind::Added,
                });
            }
        }

        diffs.sort_by(|a, b| a.name.cmp(&b.name));
        diffs
    }

    /// Aggregate a slice of diffs into a [`DiffSummary`].
    pub fn summarize(&self, diffs: &[TensorDiff]) -> DiffSummary {
        let mut summary = DiffSummary::default();
        for diff in diffs {
            match &diff.kind {
                DiffKind::Added => summary.added += 1,
                DiffKind::Removed => summary.removed += 1,
                DiffKind::ShapeChanged { .. } => summary.shape_changed += 1,
                DiffKind::ValueChanged {
                    changed_elements, ..
                } => {
                    summary.value_changed += 1;
                    summary.total_changed_elements += changed_elements;
                }
                DiffKind::Unchanged => summary.unchanged += 1,
            }
        }
        summary
    }

    /// Filter diffs to only those that are significant at `self.threshold`.
    pub fn significant_diffs<'a>(&self, diffs: &'a [TensorDiff]) -> Vec<&'a TensorDiff> {
        diffs
            .iter()
            .filter(|d| d.is_significant(self.threshold))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snap(name: &str, shape: Vec<usize>, data: Vec<f32>) -> TensorSnapshot {
        TensorSnapshot::new(name, shape, data)
    }

    // 1. new() stores threshold
    #[test]
    fn test_engine_new_threshold() {
        let engine = TensorDiffEngine::new(1e-4);
        assert!((engine.threshold - 1e-4_f32).abs() < f32::EPSILON);
    }

    // 2. numel() for 2-D shape
    #[test]
    fn test_numel_2d() {
        let snap = make_snap("w", vec![128, 256], vec![0.0; 128 * 256]);
        assert_eq!(snap.numel(), 128 * 256);
    }

    // 3. numel() empty shape falls back to data.len()
    #[test]
    fn test_numel_empty_shape() {
        let snap = make_snap("scalar", vec![], vec![1.0, 2.0, 3.0]);
        assert_eq!(snap.numel(), 3);
    }

    // 4. is_compatible: matching shapes → true
    #[test]
    fn test_is_compatible_matching() {
        let a = make_snap("a", vec![4, 4], vec![0.0; 16]);
        let b = make_snap("b", vec![4, 4], vec![1.0; 16]);
        assert!(a.is_compatible(&b));
    }

    // 5. is_compatible: mismatched → false
    #[test]
    fn test_is_compatible_mismatch() {
        let a = make_snap("a", vec![4, 4], vec![0.0; 16]);
        let b = make_snap("b", vec![4, 8], vec![0.0; 32]);
        assert!(!a.is_compatible(&b));
    }

    // 6. diff_tensors: identical → Unchanged
    #[test]
    fn test_diff_tensors_identical_unchanged() {
        let engine = TensorDiffEngine::new(1e-6);
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let old = make_snap("t", vec![4], data.clone());
        let new = make_snap("t", vec![4], data);
        let diff = engine.diff_tensors(&old, &new);
        assert_eq!(diff.kind, DiffKind::Unchanged);
    }

    // 7. diff_tensors: small delta below threshold → Unchanged
    #[test]
    fn test_diff_tensors_small_delta_unchanged() {
        let threshold = 1e-5_f32;
        let engine = TensorDiffEngine::new(threshold);
        let old = make_snap("t", vec![3], vec![1.0, 2.0, 3.0]);
        let new = make_snap("t", vec![3], vec![1.0 + 1e-7, 2.0, 3.0]);
        let diff = engine.diff_tensors(&old, &new);
        assert_eq!(diff.kind, DiffKind::Unchanged);
    }

    // 8. diff_tensors: large delta above threshold → ValueChanged with correct max/mean
    #[test]
    fn test_diff_tensors_value_changed_max_mean() {
        let engine = TensorDiffEngine::new(1e-6);
        let old = make_snap("t", vec![2], vec![0.0, 0.0]);
        let new = make_snap("t", vec![2], vec![1.0, 3.0]);
        let diff = engine.diff_tensors(&old, &new);
        match diff.kind {
            DiffKind::ValueChanged {
                max_abs_diff,
                mean_abs_diff,
                changed_elements,
            } => {
                assert!((max_abs_diff - 3.0).abs() < 1e-5, "max={max_abs_diff}");
                assert!((mean_abs_diff - 2.0).abs() < 1e-5, "mean={mean_abs_diff}");
                assert_eq!(changed_elements, 2);
            }
            other => panic!("Expected ValueChanged, got {other:?}"),
        }
    }

    // 9. diff_tensors: shape mismatch → ShapeChanged
    #[test]
    fn test_diff_tensors_shape_mismatch() {
        let engine = TensorDiffEngine::new(1e-6);
        let old = make_snap("t", vec![2, 2], vec![0.0; 4]);
        let new = make_snap("t", vec![4], vec![0.0; 4]);
        let diff = engine.diff_tensors(&old, &new);
        match diff.kind {
            DiffKind::ShapeChanged {
                old_shape,
                new_shape,
            } => {
                assert_eq!(old_shape, vec![2, 2]);
                assert_eq!(new_shape, vec![4]);
            }
            other => panic!("Expected ShapeChanged, got {other:?}"),
        }
    }

    // 10. diff_tensors: mean_abs_diff computed correctly for 4 elements
    #[test]
    fn test_diff_tensors_mean_abs_diff_four_elements() {
        let engine = TensorDiffEngine::new(1e-6);
        // diffs: 1, 2, 3, 4 → mean = 2.5
        let old = make_snap("t", vec![4], vec![0.0, 0.0, 0.0, 0.0]);
        let new = make_snap("t", vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let diff = engine.diff_tensors(&old, &new);
        match diff.kind {
            DiffKind::ValueChanged { mean_abs_diff, .. } => {
                assert!((mean_abs_diff - 2.5).abs() < 1e-5, "mean={mean_abs_diff}");
            }
            other => panic!("Expected ValueChanged, got {other:?}"),
        }
    }

    // 11. diff_snapshots: added tensor detected
    #[test]
    fn test_diff_snapshots_added() {
        let engine = TensorDiffEngine::new(1e-6);
        let old_set: Vec<TensorSnapshot> = vec![];
        let new_set = vec![make_snap("layer.weight", vec![2], vec![1.0, 2.0])];
        let diffs = engine.diff_snapshots(&old_set, &new_set);
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].name, "layer.weight");
        assert_eq!(diffs[0].kind, DiffKind::Added);
    }

    // 12. diff_snapshots: removed tensor detected
    #[test]
    fn test_diff_snapshots_removed() {
        let engine = TensorDiffEngine::new(1e-6);
        let old_set = vec![make_snap("layer.weight", vec![2], vec![1.0, 2.0])];
        let new_set: Vec<TensorSnapshot> = vec![];
        let diffs = engine.diff_snapshots(&old_set, &new_set);
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].name, "layer.weight");
        assert_eq!(diffs[0].kind, DiffKind::Removed);
    }

    // 13. diff_snapshots: unchanged tensor detected
    #[test]
    fn test_diff_snapshots_unchanged() {
        let engine = TensorDiffEngine::new(1e-6);
        let snap = make_snap("w", vec![3], vec![1.0, 2.0, 3.0]);
        let old_set = vec![snap.clone()];
        let new_set = vec![snap];
        let diffs = engine.diff_snapshots(&old_set, &new_set);
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].kind, DiffKind::Unchanged);
    }

    // 14. diff_snapshots: changed tensor detected
    #[test]
    fn test_diff_snapshots_changed() {
        let engine = TensorDiffEngine::new(1e-6);
        let old_set = vec![make_snap("w", vec![2], vec![0.0, 0.0])];
        let new_set = vec![make_snap("w", vec![2], vec![1.0, 1.0])];
        let diffs = engine.diff_snapshots(&old_set, &new_set);
        assert_eq!(diffs.len(), 1);
        assert!(matches!(diffs[0].kind, DiffKind::ValueChanged { .. }));
    }

    // 15. diff_snapshots: output sorted by name
    #[test]
    fn test_diff_snapshots_sorted_by_name() {
        let engine = TensorDiffEngine::new(1e-6);
        let old_set = vec![
            make_snap("zebra", vec![1], vec![1.0]),
            make_snap("apple", vec![1], vec![2.0]),
            make_snap("mango", vec![1], vec![3.0]),
        ];
        let new_set = vec![
            make_snap("mango", vec![1], vec![3.0]),
            make_snap("zebra", vec![1], vec![1.0]),
            make_snap("apple", vec![1], vec![2.0]),
        ];
        let diffs = engine.diff_snapshots(&old_set, &new_set);
        let names: Vec<&str> = diffs.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, vec!["apple", "mango", "zebra"]);
    }

    // 16. summarize: counts correct
    #[test]
    fn test_summarize_counts() {
        let engine = TensorDiffEngine::new(1e-6);
        let diffs = vec![
            TensorDiff {
                name: "a".into(),
                kind: DiffKind::Added,
            },
            TensorDiff {
                name: "b".into(),
                kind: DiffKind::Removed,
            },
            TensorDiff {
                name: "c".into(),
                kind: DiffKind::ShapeChanged {
                    old_shape: vec![2],
                    new_shape: vec![4],
                },
            },
            TensorDiff {
                name: "d".into(),
                kind: DiffKind::ValueChanged {
                    max_abs_diff: 0.5,
                    mean_abs_diff: 0.25,
                    changed_elements: 7,
                },
            },
            TensorDiff {
                name: "e".into(),
                kind: DiffKind::Unchanged,
            },
        ];
        let summary = engine.summarize(&diffs);
        assert_eq!(summary.added, 1);
        assert_eq!(summary.removed, 1);
        assert_eq!(summary.shape_changed, 1);
        assert_eq!(summary.value_changed, 1);
        assert_eq!(summary.unchanged, 1);
        assert_eq!(summary.total_changed_elements, 7);
        assert!(summary.has_changes());
    }

    // 17. significant_diffs filters by threshold
    #[test]
    fn test_significant_diffs_filters() {
        let engine = TensorDiffEngine::new(0.1);
        let diffs = vec![
            TensorDiff {
                name: "big".into(),
                kind: DiffKind::ValueChanged {
                    max_abs_diff: 0.5,
                    mean_abs_diff: 0.3,
                    changed_elements: 3,
                },
            },
            TensorDiff {
                name: "small".into(),
                kind: DiffKind::ValueChanged {
                    max_abs_diff: 0.05,
                    mean_abs_diff: 0.02,
                    changed_elements: 1,
                },
            },
            TensorDiff {
                name: "added".into(),
                kind: DiffKind::Added,
            },
        ];
        let sig = engine.significant_diffs(&diffs);
        // "big" (0.5 > 0.1) and "added" are significant; "small" (0.05 <= 0.1) is not
        let names: Vec<&str> = sig.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"big"));
        assert!(names.contains(&"added"));
        assert!(!names.contains(&"small"));
    }

    // 18. is_significant: Unchanged always false
    #[test]
    fn test_is_significant_unchanged_false() {
        let diff = TensorDiff {
            name: "t".into(),
            kind: DiffKind::Unchanged,
        };
        assert!(!diff.is_significant(0.0));
        assert!(!diff.is_significant(1e-10));
        assert!(!diff.is_significant(f32::MAX));
    }

    // Bonus: has_changes() false when all unchanged
    #[test]
    fn test_has_changes_false_when_all_unchanged() {
        let engine = TensorDiffEngine::new(1e-6);
        let diffs = vec![TensorDiff {
            name: "t".into(),
            kind: DiffKind::Unchanged,
        }];
        let summary = engine.summarize(&diffs);
        assert!(!summary.has_changes());
    }
}
