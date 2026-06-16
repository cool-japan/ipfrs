//! `TensorKernelRegistry` — a registry of computational kernels (named tensor
//! operations with metadata), enabling dynamic kernel lookup, versioning, and
//! capability-based selection.
//!
//! # Overview
//!
//! The registry stores [`KernelDescriptor`] entries keyed by a monotonically
//! increasing `u64` identifier. Callers may:
//!
//! - [`TensorKernelRegistry::register`] — add a new kernel and obtain its id.
//! - [`TensorKernelRegistry::lookup`] — query by name/precision/target/tag.
//! - [`TensorKernelRegistry::best_for`] — select the most capable kernel for a
//!   given name+precision combination (Simd > Cpu > Gpu > Generic, then highest
//!   version).
//! - [`TensorKernelRegistry::remove`] / [`TensorKernelRegistry::get`] — remove
//!   or retrieve individual entries.
//! - [`TensorKernelRegistry::stats`] — observe aggregate counters.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// KernelPrecision
// ---------------------------------------------------------------------------

/// Numeric precision of a kernel's computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum KernelPrecision {
    /// 16-bit IEEE 754 half precision float.
    F16,
    /// 32-bit IEEE 754 single precision float.
    F32,
    /// 64-bit IEEE 754 double precision float.
    F64,
    /// 8-bit signed integer.
    I8,
    /// 32-bit signed integer.
    I32,
}

// ---------------------------------------------------------------------------
// KernelTarget
// ---------------------------------------------------------------------------

/// Execution target for a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelTarget {
    /// Scalar CPU implementation.
    Cpu,
    /// GPU (CUDA / OpenCL / Vulkan) implementation.
    Gpu,
    /// Vectorised CPU implementation (SIMD/AVX/NEON/…).
    Simd,
    /// Portable fallback — no architecture-specific instructions.
    Generic,
}

// ---------------------------------------------------------------------------
// KernelDescriptor
// ---------------------------------------------------------------------------

/// Metadata record for a single computational kernel.
#[derive(Clone, Debug)]
pub struct KernelDescriptor {
    /// Unique stable identifier assigned by the registry.
    pub kernel_id: u64,
    /// Human-readable operation name, e.g. `"matmul"`, `"relu"`, `"softmax"`.
    pub name: String,
    /// Monotonically increasing version counter for the same operation.
    pub version: u32,
    /// Numeric precision used by this kernel.
    pub precision: KernelPrecision,
    /// Execution target for this kernel.
    pub target: KernelTarget,
    /// Approximate number of floating-point operations per output element.
    pub flops_per_element: f64,
    /// Searchable tags (e.g. `["fused"`, `"blas"`, `"experimental"]`).
    pub tags: Vec<String>,
}

// ---------------------------------------------------------------------------
// KernelQuery
// ---------------------------------------------------------------------------

/// Filter criteria for [`TensorKernelRegistry::lookup`].
///
/// All `Some` fields must match; `None` fields are ignored (wildcard).
#[derive(Clone, Debug, Default)]
pub struct KernelQuery {
    /// Case-insensitive substring match against [`KernelDescriptor::name`].
    pub name: Option<String>,
    /// Exact match against [`KernelDescriptor::precision`].
    pub precision: Option<KernelPrecision>,
    /// Exact match against [`KernelDescriptor::target`].
    pub target: Option<KernelTarget>,
    /// Exact match against any element of [`KernelDescriptor::tags`].
    pub tag: Option<String>,
}

// ---------------------------------------------------------------------------
// KernelRegistryStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for a [`TensorKernelRegistry`].
#[derive(Clone, Debug, Default)]
pub struct KernelRegistryStats {
    /// Total number of kernels currently registered.
    pub total_kernels: usize,
    /// Number of registered kernels per [`KernelTarget`].
    pub by_target: HashMap<KernelTarget, usize>,
    /// Number of registered kernels per [`KernelPrecision`].
    pub by_precision: HashMap<KernelPrecision, usize>,
    /// Running count of [`lookup`](TensorKernelRegistry::lookup) and
    /// [`best_for`](TensorKernelRegistry::best_for) calls.
    pub total_lookups: u64,
}

// ---------------------------------------------------------------------------
// TensorKernelRegistry
// ---------------------------------------------------------------------------

/// Registry of computational kernels with dynamic lookup and versioning.
pub struct TensorKernelRegistry {
    kernels: HashMap<u64, KernelDescriptor>,
    next_id: u64,
    stats: KernelRegistryStats,
}

impl Default for TensorKernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorKernelRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            kernels: HashMap::new(),
            next_id: 1,
            stats: KernelRegistryStats::default(),
        }
    }

    // -----------------------------------------------------------------------
    // Mutation helpers
    // -----------------------------------------------------------------------

    /// Rebuild the `by_target` and `by_precision` histogram maps from scratch.
    ///
    /// Called after every register/remove to keep stats consistent.
    fn rebuild_histograms(&mut self) {
        let mut by_target: HashMap<KernelTarget, usize> = HashMap::new();
        let mut by_precision: HashMap<KernelPrecision, usize> = HashMap::new();
        for desc in self.kernels.values() {
            *by_target.entry(desc.target).or_insert(0) += 1;
            *by_precision.entry(desc.precision).or_insert(0) += 1;
        }
        self.stats.by_target = by_target;
        self.stats.by_precision = by_precision;
        self.stats.total_kernels = self.kernels.len();
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Register a new kernel and return its assigned `kernel_id`.
    ///
    /// # Arguments
    ///
    /// * `name`             — Operation name (e.g. `"matmul"`).
    /// * `version`          — Caller-supplied version number.
    /// * `precision`        — Numeric precision.
    /// * `target`           — Execution target.
    /// * `flops_per_element`— Approximate FLOPs per output element.
    /// * `tags`             — Searchable tag strings.
    pub fn register(
        &mut self,
        name: String,
        version: u32,
        precision: KernelPrecision,
        target: KernelTarget,
        flops_per_element: f64,
        tags: Vec<String>,
    ) -> u64 {
        let kernel_id = self.next_id;
        self.next_id += 1;

        let descriptor = KernelDescriptor {
            kernel_id,
            name,
            version,
            precision,
            target,
            flops_per_element,
            tags,
        };
        self.kernels.insert(kernel_id, descriptor);
        self.rebuild_histograms();
        kernel_id
    }

    /// Query the registry and return matching [`KernelDescriptor`] references.
    ///
    /// All non-`None` query fields must match. Results are sorted by
    /// `(name asc, version desc, kernel_id asc)`.
    ///
    /// Increments [`KernelRegistryStats::total_lookups`].
    pub fn lookup(&mut self, query: &KernelQuery) -> Vec<&KernelDescriptor> {
        self.stats.total_lookups += 1;

        let mut matches: Vec<&KernelDescriptor> = self
            .kernels
            .values()
            .filter(|desc| {
                // Name: case-insensitive substring
                if let Some(ref name_filter) = query.name {
                    let lc_name = desc.name.to_lowercase();
                    let lc_filter = name_filter.to_lowercase();
                    if !lc_name.contains(lc_filter.as_str()) {
                        return false;
                    }
                }
                // Precision: exact
                if let Some(prec) = query.precision {
                    if desc.precision != prec {
                        return false;
                    }
                }
                // Target: exact
                if let Some(tgt) = query.target {
                    if desc.target != tgt {
                        return false;
                    }
                }
                // Tag: exact match against any tag in the descriptor
                if let Some(ref tag_filter) = query.tag {
                    if !desc.tags.iter().any(|t| t == tag_filter) {
                        return false;
                    }
                }
                true
            })
            .collect();

        // Sort: name asc, then version desc, then kernel_id asc
        matches.sort_by(|a, b| {
            a.name
                .cmp(&b.name)
                .then_with(|| b.version.cmp(&a.version))
                .then_with(|| a.kernel_id.cmp(&b.kernel_id))
        });

        matches
    }

    /// Select the best kernel for `name` + `precision`.
    ///
    /// Target preference: `Simd > Cpu > Gpu > Generic`. Within the same target,
    /// the highest `version` wins.
    ///
    /// Increments [`KernelRegistryStats::total_lookups`].
    pub fn best_for(
        &mut self,
        name: &str,
        precision: KernelPrecision,
    ) -> Option<&KernelDescriptor> {
        self.stats.total_lookups += 1;

        // Collect IDs of candidates first to avoid borrow issues.
        let candidate_ids: Vec<u64> = self
            .kernels
            .values()
            .filter(|desc| desc.name.eq_ignore_ascii_case(name) && desc.precision == precision)
            .map(|desc| desc.kernel_id)
            .collect();

        if candidate_ids.is_empty() {
            return None;
        }

        // Map target to priority (lower = better).
        let target_priority = |t: KernelTarget| match t {
            KernelTarget::Simd => 0u8,
            KernelTarget::Cpu => 1,
            KernelTarget::Gpu => 2,
            KernelTarget::Generic => 3,
        };

        // Pick the best by (target priority asc, version desc).
        let best_id = candidate_ids.into_iter().min_by(|&a_id, &b_id| {
            let a = &self.kernels[&a_id];
            let b = &self.kernels[&b_id];
            target_priority(a.target)
                .cmp(&target_priority(b.target))
                .then_with(|| b.version.cmp(&a.version)) // higher version first
        });

        best_id.and_then(|id| self.kernels.get(&id))
    }

    /// Remove a kernel by id.
    ///
    /// Returns `true` if the kernel existed and was removed, `false` otherwise.
    pub fn remove(&mut self, kernel_id: u64) -> bool {
        let removed = self.kernels.remove(&kernel_id).is_some();
        if removed {
            self.rebuild_histograms();
        }
        removed
    }

    /// Retrieve a kernel by id without modifying stats.
    pub fn get(&self, kernel_id: u64) -> Option<&KernelDescriptor> {
        self.kernels.get(&kernel_id)
    }

    /// Return a reference to the current registry statistics.
    pub fn stats(&self) -> &KernelRegistryStats {
        &self.stats
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: register a kernel with default flops and no tags.
    fn reg_simple(
        r: &mut TensorKernelRegistry,
        name: &str,
        version: u32,
        precision: KernelPrecision,
        target: KernelTarget,
    ) -> u64 {
        r.register(name.to_string(), version, precision, target, 1.0, vec![])
    }

    /// Helper: register a kernel with tags.
    fn reg_tagged(
        r: &mut TensorKernelRegistry,
        name: &str,
        version: u32,
        precision: KernelPrecision,
        target: KernelTarget,
        tags: Vec<&str>,
    ) -> u64 {
        r.register(
            name.to_string(),
            version,
            precision,
            target,
            1.0,
            tags.into_iter().map(String::from).collect(),
        )
    }

    // -----------------------------------------------------------------------
    // register
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_creates_kernel() {
        let mut r = TensorKernelRegistry::new();
        let id = reg_simple(&mut r, "matmul", 1, KernelPrecision::F32, KernelTarget::Cpu);
        assert!(r.get(id).is_some());
    }

    #[test]
    fn test_register_assigns_unique_ids() {
        let mut r = TensorKernelRegistry::new();
        let id1 = reg_simple(&mut r, "relu", 1, KernelPrecision::F32, KernelTarget::Cpu);
        let id2 = reg_simple(&mut r, "relu", 1, KernelPrecision::F32, KernelTarget::Gpu);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_register_increments_total_kernels() {
        let mut r = TensorKernelRegistry::new();
        reg_simple(&mut r, "relu", 1, KernelPrecision::F32, KernelTarget::Cpu);
        reg_simple(
            &mut r,
            "sigmoid",
            1,
            KernelPrecision::F64,
            KernelTarget::Gpu,
        );
        assert_eq!(r.stats().total_kernels, 2);
    }

    #[test]
    fn test_register_stores_correct_metadata() {
        let mut r = TensorKernelRegistry::new();
        let id = r.register(
            "softmax".to_string(),
            3,
            KernelPrecision::F16,
            KernelTarget::Simd,
            4.0,
            vec!["fused".to_string(), "stable".to_string()],
        );
        let desc = r.get(id).expect("kernel must exist");
        assert_eq!(desc.name, "softmax");
        assert_eq!(desc.version, 3);
        assert_eq!(desc.precision, KernelPrecision::F16);
        assert_eq!(desc.target, KernelTarget::Simd);
        assert!((desc.flops_per_element - 4.0).abs() < f64::EPSILON);
        assert_eq!(desc.tags, vec!["fused", "stable"]);
    }

    // -----------------------------------------------------------------------
    // lookup — individual filters
    // -----------------------------------------------------------------------

    #[test]
    fn test_lookup_by_name_substring() {
        let mut r = TensorKernelRegistry::new();
        reg_simple(&mut r, "matmul", 1, KernelPrecision::F32, KernelTarget::Cpu);
        reg_simple(&mut r, "relu", 1, KernelPrecision::F32, KernelTarget::Cpu);
        let q = KernelQuery {
            name: Some("mat".to_string()),
            ..Default::default()
        };
        let results = r.lookup(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "matmul");
    }

    #[test]
    fn test_lookup_by_name_case_insensitive() {
        let mut r = TensorKernelRegistry::new();
        reg_simple(&mut r, "MatMul", 1, KernelPrecision::F32, KernelTarget::Cpu);
        let q = KernelQuery {
            name: Some("matmul".to_string()),
            ..Default::default()
        };
        let results = r.lookup(&q);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_lookup_by_precision() {
        let mut r = TensorKernelRegistry::new();
        reg_simple(&mut r, "conv", 1, KernelPrecision::F16, KernelTarget::Cpu);
        reg_simple(&mut r, "conv", 1, KernelPrecision::F32, KernelTarget::Cpu);
        reg_simple(&mut r, "conv", 1, KernelPrecision::F64, KernelTarget::Cpu);
        let q = KernelQuery {
            precision: Some(KernelPrecision::F32),
            ..Default::default()
        };
        let results = r.lookup(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].precision, KernelPrecision::F32);
    }

    #[test]
    fn test_lookup_by_target_simd() {
        let mut r = TensorKernelRegistry::new();
        reg_simple(&mut r, "relu", 1, KernelPrecision::F32, KernelTarget::Cpu);
        reg_simple(&mut r, "relu", 1, KernelPrecision::F32, KernelTarget::Simd);
        let q = KernelQuery {
            target: Some(KernelTarget::Simd),
            ..Default::default()
        };
        let results = r.lookup(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].target, KernelTarget::Simd);
    }

    #[test]
    fn test_lookup_by_target_gpu() {
        let mut r = TensorKernelRegistry::new();
        reg_simple(&mut r, "gemm", 1, KernelPrecision::F32, KernelTarget::Gpu);
        reg_simple(&mut r, "gemm", 1, KernelPrecision::F32, KernelTarget::Cpu);
        let q = KernelQuery {
            target: Some(KernelTarget::Gpu),
            ..Default::default()
        };
        let results = r.lookup(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].target, KernelTarget::Gpu);
    }

    #[test]
    fn test_lookup_by_tag_exact_match() {
        let mut r = TensorKernelRegistry::new();
        reg_tagged(
            &mut r,
            "softmax",
            1,
            KernelPrecision::F32,
            KernelTarget::Cpu,
            vec!["stable", "blas"],
        );
        reg_tagged(
            &mut r,
            "relu",
            1,
            KernelPrecision::F32,
            KernelTarget::Cpu,
            vec!["experimental"],
        );
        let q = KernelQuery {
            tag: Some("stable".to_string()),
            ..Default::default()
        };
        let results = r.lookup(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "softmax");
    }

    #[test]
    fn test_lookup_tag_no_partial_match() {
        let mut r = TensorKernelRegistry::new();
        reg_tagged(
            &mut r,
            "relu",
            1,
            KernelPrecision::F32,
            KernelTarget::Cpu,
            vec!["stable"],
        );
        let q = KernelQuery {
            tag: Some("stab".to_string()), // not an exact match
            ..Default::default()
        };
        let results = r.lookup(&q);
        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // lookup — combined filters
    // -----------------------------------------------------------------------

    #[test]
    fn test_lookup_combined_name_and_precision() {
        let mut r = TensorKernelRegistry::new();
        reg_simple(&mut r, "matmul", 1, KernelPrecision::F32, KernelTarget::Cpu);
        reg_simple(&mut r, "matmul", 1, KernelPrecision::F64, KernelTarget::Cpu);
        reg_simple(&mut r, "relu", 1, KernelPrecision::F32, KernelTarget::Cpu);
        let q = KernelQuery {
            name: Some("matmul".to_string()),
            precision: Some(KernelPrecision::F32),
            ..Default::default()
        };
        let results = r.lookup(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "matmul");
        assert_eq!(results[0].precision, KernelPrecision::F32);
    }

    #[test]
    fn test_lookup_combined_precision_and_target() {
        let mut r = TensorKernelRegistry::new();
        reg_simple(&mut r, "conv", 1, KernelPrecision::F32, KernelTarget::Simd);
        reg_simple(&mut r, "conv", 1, KernelPrecision::F32, KernelTarget::Gpu);
        reg_simple(&mut r, "conv", 1, KernelPrecision::F64, KernelTarget::Simd);
        let q = KernelQuery {
            precision: Some(KernelPrecision::F32),
            target: Some(KernelTarget::Simd),
            ..Default::default()
        };
        let results = r.lookup(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].target, KernelTarget::Simd);
        assert_eq!(results[0].precision, KernelPrecision::F32);
    }

    #[test]
    fn test_lookup_all_filters() {
        let mut r = TensorKernelRegistry::new();
        reg_tagged(
            &mut r,
            "matmul",
            2,
            KernelPrecision::F32,
            KernelTarget::Simd,
            vec!["fast"],
        );
        reg_tagged(
            &mut r,
            "matmul",
            2,
            KernelPrecision::F32,
            KernelTarget::Cpu,
            vec!["fast"],
        );
        reg_tagged(
            &mut r,
            "gemm",
            2,
            KernelPrecision::F32,
            KernelTarget::Simd,
            vec!["fast"],
        );
        let q = KernelQuery {
            name: Some("matmul".to_string()),
            precision: Some(KernelPrecision::F32),
            target: Some(KernelTarget::Simd),
            tag: Some("fast".to_string()),
        };
        let results = r.lookup(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "matmul");
        assert_eq!(results[0].target, KernelTarget::Simd);
    }

    // -----------------------------------------------------------------------
    // lookup — sort order
    // -----------------------------------------------------------------------

    #[test]
    fn test_lookup_sorted_name_asc_version_desc() {
        let mut r = TensorKernelRegistry::new();
        reg_simple(&mut r, "relu", 1, KernelPrecision::F32, KernelTarget::Cpu);
        reg_simple(&mut r, "matmul", 2, KernelPrecision::F32, KernelTarget::Cpu);
        reg_simple(&mut r, "matmul", 1, KernelPrecision::F32, KernelTarget::Cpu);
        reg_simple(&mut r, "relu", 3, KernelPrecision::F32, KernelTarget::Cpu);
        let q = KernelQuery::default();
        let results = r.lookup(&q);
        assert_eq!(results.len(), 4);
        // name asc: matmul comes before relu
        assert_eq!(results[0].name, "matmul");
        assert_eq!(results[1].name, "matmul");
        // within matmul, version desc: 2 before 1
        assert_eq!(results[0].version, 2);
        assert_eq!(results[1].version, 1);
        // relu entries
        assert_eq!(results[2].name, "relu");
        assert_eq!(results[3].name, "relu");
        // within relu, version desc: 3 before 1
        assert_eq!(results[2].version, 3);
        assert_eq!(results[3].version, 1);
    }

    #[test]
    fn test_lookup_empty_registry_returns_nothing() {
        let mut r = TensorKernelRegistry::new();
        let results = r.lookup(&KernelQuery::default());
        assert!(results.is_empty());
    }

    #[test]
    fn test_lookup_increments_total_lookups() {
        let mut r = TensorKernelRegistry::new();
        reg_simple(&mut r, "relu", 1, KernelPrecision::F32, KernelTarget::Cpu);
        assert_eq!(r.stats().total_lookups, 0);
        r.lookup(&KernelQuery::default());
        assert_eq!(r.stats().total_lookups, 1);
        r.lookup(&KernelQuery::default());
        assert_eq!(r.stats().total_lookups, 2);
    }

    // -----------------------------------------------------------------------
    // best_for
    // -----------------------------------------------------------------------

    #[test]
    fn test_best_for_prefers_simd_over_cpu() {
        let mut r = TensorKernelRegistry::new();
        reg_simple(&mut r, "relu", 1, KernelPrecision::F32, KernelTarget::Cpu);
        reg_simple(&mut r, "relu", 1, KernelPrecision::F32, KernelTarget::Simd);
        let best = r.best_for("relu", KernelPrecision::F32).expect("must find");
        assert_eq!(best.target, KernelTarget::Simd);
    }

    #[test]
    fn test_best_for_prefers_cpu_over_gpu() {
        let mut r = TensorKernelRegistry::new();
        reg_simple(&mut r, "relu", 1, KernelPrecision::F32, KernelTarget::Gpu);
        reg_simple(&mut r, "relu", 1, KernelPrecision::F32, KernelTarget::Cpu);
        let best = r.best_for("relu", KernelPrecision::F32).expect("must find");
        assert_eq!(best.target, KernelTarget::Cpu);
    }

    #[test]
    fn test_best_for_prefers_gpu_over_generic() {
        let mut r = TensorKernelRegistry::new();
        reg_simple(
            &mut r,
            "conv",
            1,
            KernelPrecision::F32,
            KernelTarget::Generic,
        );
        reg_simple(&mut r, "conv", 1, KernelPrecision::F32, KernelTarget::Gpu);
        let best = r.best_for("conv", KernelPrecision::F32).expect("must find");
        assert_eq!(best.target, KernelTarget::Gpu);
    }

    #[test]
    fn test_best_for_prefers_simd_highest_priority() {
        let mut r = TensorKernelRegistry::new();
        reg_simple(&mut r, "op", 1, KernelPrecision::F64, KernelTarget::Generic);
        reg_simple(&mut r, "op", 1, KernelPrecision::F64, KernelTarget::Gpu);
        reg_simple(&mut r, "op", 1, KernelPrecision::F64, KernelTarget::Cpu);
        reg_simple(&mut r, "op", 1, KernelPrecision::F64, KernelTarget::Simd);
        let best = r.best_for("op", KernelPrecision::F64).expect("must find");
        assert_eq!(best.target, KernelTarget::Simd);
    }

    #[test]
    fn test_best_for_prefers_higher_version_same_target() {
        let mut r = TensorKernelRegistry::new();
        reg_simple(&mut r, "matmul", 1, KernelPrecision::F32, KernelTarget::Cpu);
        reg_simple(&mut r, "matmul", 5, KernelPrecision::F32, KernelTarget::Cpu);
        reg_simple(&mut r, "matmul", 3, KernelPrecision::F32, KernelTarget::Cpu);
        let best = r
            .best_for("matmul", KernelPrecision::F32)
            .expect("must find");
        assert_eq!(best.version, 5);
    }

    #[test]
    fn test_best_for_simd_higher_version_wins_over_simd_lower() {
        let mut r = TensorKernelRegistry::new();
        reg_simple(
            &mut r,
            "sigmoid",
            2,
            KernelPrecision::F32,
            KernelTarget::Simd,
        );
        reg_simple(
            &mut r,
            "sigmoid",
            7,
            KernelPrecision::F32,
            KernelTarget::Simd,
        );
        let best = r
            .best_for("sigmoid", KernelPrecision::F32)
            .expect("must find");
        assert_eq!(best.version, 7);
    }

    #[test]
    fn test_best_for_returns_none_when_no_name_match() {
        let mut r = TensorKernelRegistry::new();
        reg_simple(&mut r, "relu", 1, KernelPrecision::F32, KernelTarget::Cpu);
        assert!(r.best_for("matmul", KernelPrecision::F32).is_none());
    }

    #[test]
    fn test_best_for_returns_none_when_no_precision_match() {
        let mut r = TensorKernelRegistry::new();
        reg_simple(&mut r, "relu", 1, KernelPrecision::F32, KernelTarget::Cpu);
        assert!(r.best_for("relu", KernelPrecision::F64).is_none());
    }

    #[test]
    fn test_best_for_empty_registry_returns_none() {
        let mut r = TensorKernelRegistry::new();
        assert!(r.best_for("anything", KernelPrecision::F32).is_none());
    }

    #[test]
    fn test_best_for_increments_total_lookups() {
        let mut r = TensorKernelRegistry::new();
        reg_simple(&mut r, "relu", 1, KernelPrecision::F32, KernelTarget::Cpu);
        assert_eq!(r.stats().total_lookups, 0);
        let _ = r.best_for("relu", KernelPrecision::F32);
        assert_eq!(r.stats().total_lookups, 1);
    }

    // -----------------------------------------------------------------------
    // remove
    // -----------------------------------------------------------------------

    #[test]
    fn test_remove_returns_true_when_exists() {
        let mut r = TensorKernelRegistry::new();
        let id = reg_simple(&mut r, "relu", 1, KernelPrecision::F32, KernelTarget::Cpu);
        assert!(r.remove(id));
    }

    #[test]
    fn test_remove_returns_false_when_not_exists() {
        let mut r = TensorKernelRegistry::new();
        assert!(!r.remove(9999));
    }

    #[test]
    fn test_remove_decrements_total_kernels() {
        let mut r = TensorKernelRegistry::new();
        let id = reg_simple(&mut r, "relu", 1, KernelPrecision::F32, KernelTarget::Cpu);
        assert_eq!(r.stats().total_kernels, 1);
        r.remove(id);
        assert_eq!(r.stats().total_kernels, 0);
    }

    #[test]
    fn test_remove_makes_kernel_unreachable() {
        let mut r = TensorKernelRegistry::new();
        let id = reg_simple(&mut r, "relu", 1, KernelPrecision::F32, KernelTarget::Cpu);
        r.remove(id);
        assert!(r.get(id).is_none());
    }

    #[test]
    fn test_remove_idempotent_second_call() {
        let mut r = TensorKernelRegistry::new();
        let id = reg_simple(&mut r, "relu", 1, KernelPrecision::F32, KernelTarget::Cpu);
        assert!(r.remove(id));
        assert!(!r.remove(id));
    }

    // -----------------------------------------------------------------------
    // get
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_returns_some_for_existing() {
        let mut r = TensorKernelRegistry::new();
        let id = reg_simple(
            &mut r,
            "conv",
            1,
            KernelPrecision::I8,
            KernelTarget::Generic,
        );
        let desc = r.get(id);
        assert!(desc.is_some());
        assert_eq!(desc.expect("test: should succeed").name, "conv");
    }

    #[test]
    fn test_get_returns_none_for_missing() {
        let r = TensorKernelRegistry::new();
        assert!(r.get(42).is_none());
    }

    // -----------------------------------------------------------------------
    // stats — histograms
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_by_target() {
        let mut r = TensorKernelRegistry::new();
        reg_simple(&mut r, "a", 1, KernelPrecision::F32, KernelTarget::Cpu);
        reg_simple(&mut r, "b", 1, KernelPrecision::F32, KernelTarget::Cpu);
        reg_simple(&mut r, "c", 1, KernelPrecision::F32, KernelTarget::Gpu);
        let stats = r.stats();
        assert_eq!(stats.by_target[&KernelTarget::Cpu], 2);
        assert_eq!(stats.by_target[&KernelTarget::Gpu], 1);
    }

    #[test]
    fn test_stats_by_precision() {
        let mut r = TensorKernelRegistry::new();
        reg_simple(&mut r, "a", 1, KernelPrecision::F32, KernelTarget::Cpu);
        reg_simple(&mut r, "b", 1, KernelPrecision::F16, KernelTarget::Cpu);
        reg_simple(&mut r, "c", 1, KernelPrecision::I32, KernelTarget::Cpu);
        reg_simple(&mut r, "d", 1, KernelPrecision::I32, KernelTarget::Cpu);
        let stats = r.stats();
        assert_eq!(stats.by_precision[&KernelPrecision::F32], 1);
        assert_eq!(stats.by_precision[&KernelPrecision::F16], 1);
        assert_eq!(stats.by_precision[&KernelPrecision::I32], 2);
    }

    #[test]
    fn test_stats_histograms_updated_on_remove() {
        let mut r = TensorKernelRegistry::new();
        let id = reg_simple(&mut r, "a", 1, KernelPrecision::F32, KernelTarget::Simd);
        assert_eq!(r.stats().by_target[&KernelTarget::Simd], 1);
        r.remove(id);
        assert_eq!(
            r.stats()
                .by_target
                .get(&KernelTarget::Simd)
                .copied()
                .unwrap_or(0),
            0
        );
    }

    #[test]
    fn test_stats_total_lookups_combined() {
        let mut r = TensorKernelRegistry::new();
        reg_simple(&mut r, "relu", 1, KernelPrecision::F32, KernelTarget::Cpu);
        r.lookup(&KernelQuery::default());
        let _ = r.best_for("relu", KernelPrecision::F32);
        r.lookup(&KernelQuery::default());
        assert_eq!(r.stats().total_lookups, 3);
    }

    #[test]
    fn test_lookup_i8_precision() {
        let mut r = TensorKernelRegistry::new();
        reg_simple(
            &mut r,
            "quantized_matmul",
            1,
            KernelPrecision::I8,
            KernelTarget::Cpu,
        );
        reg_simple(&mut r, "relu", 1, KernelPrecision::F32, KernelTarget::Cpu);
        let q = KernelQuery {
            precision: Some(KernelPrecision::I8),
            ..Default::default()
        };
        let results = r.lookup(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "quantized_matmul");
    }
}
