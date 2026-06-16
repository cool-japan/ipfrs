//! TensorOpDispatcher — routes tensor operations to registered backends
//! (CPU / GPU / Remote / Simulated) based on op type and backend availability,
//! with priority-ordered fallback chains and per-backend statistics.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// BackendKind
// ---------------------------------------------------------------------------

/// Identifies the kind (class) of a compute backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// Host CPU backend.
    Cpu,
    /// GPU backend (CUDA / Vulkan / OpenCL / Metal).
    Gpu,
    /// Remote / distributed backend reached over a network.
    Remote,
    /// Simulated backend used in unit tests.
    Simulated,
}

// ---------------------------------------------------------------------------
// DispatchOp
// ---------------------------------------------------------------------------

/// A tensor operation that must be dispatched to a backend.
#[derive(Clone, Debug, PartialEq)]
pub enum DispatchOp {
    /// General matrix multiplication: (m × k) × (k × n).
    MatMul { m: usize, n: usize, k: usize },
    /// Element-wise operation applied to a flat tensor.
    ElementWise {
        op_name: String,
        element_count: usize,
    },
    /// Reduction over one or more dimensions.
    Reduction { op_name: String, dims: Vec<usize> },
    /// Convolution with a given kernel size and channel count.
    Convolution { kernel_size: usize, channels: usize },
}

// ---------------------------------------------------------------------------
// BackendRegistration
// ---------------------------------------------------------------------------

/// Describes a single registered backend and the operations it supports.
#[derive(Clone, Debug)]
pub struct BackendRegistration {
    /// What kind of backend this is.
    pub kind: BackendKind,
    /// Higher priority values are preferred over lower ones.
    pub priority: u32,
    /// Canonical op names this backend can handle (used for ElementWise,
    /// Reduction, and the "matmul" / "conv" capability checks for Gpu/Remote).
    pub supported_ops: Vec<String>,
    /// Whether the backend is currently reachable / healthy.
    pub is_available: bool,
}

impl BackendRegistration {
    /// Returns `true` when this backend is capable of executing `op`.
    ///
    /// Matching rules:
    /// - [`DispatchOp::MatMul`]: always supported by `Cpu` and `Simulated`;
    ///   `Gpu` and `Remote` must list `"matmul"` in `supported_ops`.
    /// - [`DispatchOp::ElementWise`]: `supported_ops` must contain `op_name`.
    /// - [`DispatchOp::Reduction`]: `supported_ops` must contain `op_name`.
    /// - [`DispatchOp::Convolution`]: `supported_ops` must contain `"conv"`.
    pub fn supports_op(&self, op: &DispatchOp) -> bool {
        match op {
            DispatchOp::MatMul { .. } => match self.kind {
                BackendKind::Cpu | BackendKind::Simulated => true,
                BackendKind::Gpu | BackendKind::Remote => {
                    self.supported_ops.iter().any(|s| s == "matmul")
                }
            },
            DispatchOp::ElementWise { op_name, .. } => {
                self.supported_ops.iter().any(|s| s == op_name)
            }
            DispatchOp::Reduction { op_name, .. } => {
                self.supported_ops.iter().any(|s| s == op_name)
            }
            DispatchOp::Convolution { .. } => self.supported_ops.iter().any(|s| s == "conv"),
        }
    }
}

// ---------------------------------------------------------------------------
// DispatchResult
// ---------------------------------------------------------------------------

/// The outcome of a successful dispatch attempt.
#[derive(Clone, Debug, PartialEq)]
pub struct DispatchResult {
    /// The operation that was dispatched.
    pub op: DispatchOp,
    /// The backend that was ultimately selected to handle the operation.
    pub selected_backend: BackendKind,
    /// `true` when the highest-priority capable backend was unavailable and a
    /// lower-priority backend was chosen instead.
    pub fallback_used: bool,
}

// ---------------------------------------------------------------------------
// BackendStats
// ---------------------------------------------------------------------------

/// Per-backend accounting data.
#[derive(Clone, Debug)]
pub struct BackendStats {
    /// Which backend these statistics belong to.
    pub backend: BackendKind,
    /// Total number of operations dispatched *to* this backend.
    pub total_dispatched: u64,
    /// Number of times this backend was selected as a *fallback* (i.e., at
    /// least one higher-priority capable backend was unavailable first).
    pub total_fallback_selected: u64,
}

impl BackendStats {
    fn new(backend: BackendKind) -> Self {
        Self {
            backend,
            total_dispatched: 0,
            total_fallback_selected: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// DispatcherStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for the [`TensorOpDispatcher`].
#[derive(Clone, Debug, Default)]
pub struct DispatcherStats {
    /// Total operations successfully dispatched.
    pub total_dispatched: u64,
    /// Total operations dispatched via a fallback backend.
    pub total_fallbacks: u64,
    /// Total operations where *no* capable backend was found.
    pub total_failed: u64,
}

// ---------------------------------------------------------------------------
// TensorOpDispatcher
// ---------------------------------------------------------------------------

/// Routes [`DispatchOp`] instances to the best available registered backend,
/// maintaining per-backend and aggregate statistics.
///
/// # Priority
///
/// Backends are stored in descending priority order.  `dispatch` always picks
/// the first *available* backend that `supports_op` returns `true` for.
///
/// # Fallback
///
/// `fallback_used` is set to `true` in the [`DispatchResult`] when at least
/// one backend that *could* handle the op (i.e., `supports_op` → `true`) was
/// skipped because `is_available` was `false`.
pub struct TensorOpDispatcher {
    /// Backends ordered by priority (highest first).
    pub backends: Vec<BackendRegistration>,
    /// Per-backend statistics keyed by [`BackendKind`].
    pub backend_stats: HashMap<BackendKind, BackendStats>,
    /// Aggregate dispatcher statistics.
    pub dispatcher_stats: DispatcherStats,
}

impl TensorOpDispatcher {
    /// Creates an empty dispatcher with no registered backends.
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
            backend_stats: HashMap::new(),
            dispatcher_stats: DispatcherStats::default(),
        }
    }

    /// Registers a backend and re-sorts the list so highest-priority backends
    /// come first.  Initialises a [`BackendStats`] entry if one does not already
    /// exist.
    pub fn register_backend(&mut self, registration: BackendRegistration) {
        self.backend_stats
            .entry(registration.kind)
            .or_insert_with(|| BackendStats::new(registration.kind));
        self.backends.push(registration);
        // Maintain descending priority order.
        self.backends.sort_by_key(|b| std::cmp::Reverse(b.priority));
    }

    /// Dispatches `op` to the highest-priority available backend that supports
    /// it.
    ///
    /// Returns `None` and increments `total_failed` when no capable backend is
    /// available.
    pub fn dispatch(&mut self, op: DispatchOp) -> Option<DispatchResult> {
        // Determine whether any capable-but-unavailable backend precedes the
        // selected one (fallback detection) in a single linear pass.
        let mut found_unavailable_capable = false;
        let mut selected: Option<BackendKind> = None;

        for backend in &self.backends {
            if !backend.supports_op(&op) {
                continue;
            }
            if !backend.is_available {
                // This backend could handle the op but is currently down.
                found_unavailable_capable = true;
                continue;
            }
            // First available capable backend.
            selected = Some(backend.kind);
            break;
        }

        match selected {
            None => {
                self.dispatcher_stats.total_failed += 1;
                None
            }
            Some(kind) => {
                let fallback_used = found_unavailable_capable;

                // Update dispatcher-level stats.
                self.dispatcher_stats.total_dispatched += 1;
                if fallback_used {
                    self.dispatcher_stats.total_fallbacks += 1;
                }

                // Update per-backend stats.
                if let Some(stats) = self.backend_stats.get_mut(&kind) {
                    stats.total_dispatched += 1;
                    if fallback_used {
                        stats.total_fallback_selected += 1;
                    }
                }

                Some(DispatchResult {
                    op,
                    selected_backend: kind,
                    fallback_used,
                })
            }
        }
    }

    /// Toggles the availability flag of a backend identified by `kind`.
    /// Has no effect if `kind` is not registered.
    pub fn set_backend_available(&mut self, kind: BackendKind, available: bool) {
        for backend in &mut self.backends {
            if backend.kind == kind {
                backend.is_available = available;
            }
        }
    }

    /// Returns a reference to the aggregate [`DispatcherStats`].
    pub fn stats(&self) -> &DispatcherStats {
        &self.dispatcher_stats
    }

    /// Returns per-backend statistics for `kind`, or `None` if that backend
    /// has never been registered.
    pub fn backend_stats(&self, kind: BackendKind) -> Option<&BackendStats> {
        self.backend_stats.get(&kind)
    }
}

impl Default for TensorOpDispatcher {
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

    fn cpu_backend(priority: u32) -> BackendRegistration {
        BackendRegistration {
            kind: BackendKind::Cpu,
            priority,
            supported_ops: vec![
                "add".to_string(),
                "mul".to_string(),
                "sum".to_string(),
                "mean".to_string(),
                "conv".to_string(),
                "relu".to_string(),
            ],
            is_available: true,
        }
    }

    fn gpu_backend(priority: u32) -> BackendRegistration {
        BackendRegistration {
            kind: BackendKind::Gpu,
            priority,
            supported_ops: vec![
                "matmul".to_string(),
                "add".to_string(),
                "mul".to_string(),
                "sum".to_string(),
                "conv".to_string(),
                "relu".to_string(),
            ],
            is_available: true,
        }
    }

    fn remote_backend(priority: u32) -> BackendRegistration {
        BackendRegistration {
            kind: BackendKind::Remote,
            priority,
            supported_ops: vec!["matmul".to_string(), "sum".to_string()],
            is_available: true,
        }
    }

    fn sim_backend(priority: u32) -> BackendRegistration {
        BackendRegistration {
            kind: BackendKind::Simulated,
            priority,
            supported_ops: vec![
                "add".to_string(),
                "mul".to_string(),
                "sum".to_string(),
                "conv".to_string(),
            ],
            is_available: true,
        }
    }

    fn matmul_op() -> DispatchOp {
        DispatchOp::MatMul { m: 4, n: 4, k: 4 }
    }

    fn ew_op(name: &str) -> DispatchOp {
        DispatchOp::ElementWise {
            op_name: name.to_string(),
            element_count: 1024,
        }
    }

    fn red_op(name: &str) -> DispatchOp {
        DispatchOp::Reduction {
            op_name: name.to_string(),
            dims: vec![0, 1],
        }
    }

    fn conv_op() -> DispatchOp {
        DispatchOp::Convolution {
            kernel_size: 3,
            channels: 64,
        }
    }

    // ------------------------------------------------------------------
    // 1. register_backend — priority order is maintained
    // ------------------------------------------------------------------

    #[test]
    fn test_register_maintains_priority_order_ascending_insert() {
        let mut d = TensorOpDispatcher::new();
        d.register_backend(cpu_backend(10));
        d.register_backend(gpu_backend(50));
        d.register_backend(remote_backend(30));

        assert_eq!(d.backends[0].priority, 50);
        assert_eq!(d.backends[1].priority, 30);
        assert_eq!(d.backends[2].priority, 10);
    }

    #[test]
    fn test_register_maintains_priority_order_descending_insert() {
        let mut d = TensorOpDispatcher::new();
        d.register_backend(gpu_backend(100));
        d.register_backend(cpu_backend(10));

        assert_eq!(d.backends[0].kind, BackendKind::Gpu);
        assert_eq!(d.backends[1].kind, BackendKind::Cpu);
    }

    #[test]
    fn test_register_creates_stats_entry() {
        let mut d = TensorOpDispatcher::new();
        d.register_backend(cpu_backend(10));
        assert!(d.backend_stats(BackendKind::Cpu).is_some());
    }

    #[test]
    fn test_register_same_kind_twice_does_not_duplicate_stats_entry() {
        let mut d = TensorOpDispatcher::new();
        d.register_backend(cpu_backend(10));
        d.register_backend(cpu_backend(5)); // second CPU at lower priority
                                            // Two registrations but stats initialised once
        assert_eq!(
            d.backend_stats(BackendKind::Cpu)
                .expect("test: should succeed")
                .total_dispatched,
            0
        );
        // Two entries in backends list (different priority slots)
        assert_eq!(d.backends.len(), 2);
    }

    // ------------------------------------------------------------------
    // 2. dispatch — selects highest priority available backend
    // ------------------------------------------------------------------

    #[test]
    fn test_dispatch_selects_highest_priority() {
        let mut d = TensorOpDispatcher::new();
        d.register_backend(cpu_backend(10));
        d.register_backend(gpu_backend(50)); // GPU wins

        let result = d.dispatch(matmul_op()).expect("should dispatch");
        assert_eq!(result.selected_backend, BackendKind::Gpu);
        assert!(!result.fallback_used);
    }

    #[test]
    fn test_dispatch_selects_only_capable_backend() {
        let mut d = TensorOpDispatcher::new();
        // GPU does NOT list "add" in its supported ops
        d.register_backend(BackendRegistration {
            kind: BackendKind::Gpu,
            priority: 100,
            supported_ops: vec!["matmul".to_string()],
            is_available: true,
        });
        d.register_backend(cpu_backend(10));

        let result = d.dispatch(ew_op("add")).expect("should dispatch");
        assert_eq!(result.selected_backend, BackendKind::Cpu);
        // GPU does not support "add" so there is no capable-but-unavailable
        // backend ahead of CPU — fallback_used must be false.
        assert!(!result.fallback_used);
    }

    // ------------------------------------------------------------------
    // 3. fallback when primary unavailable
    // ------------------------------------------------------------------

    #[test]
    fn test_dispatch_fallback_when_primary_unavailable() {
        let mut d = TensorOpDispatcher::new();
        let mut gpu = gpu_backend(100);
        gpu.is_available = false;
        d.register_backend(gpu);
        d.register_backend(cpu_backend(10));

        let result = d.dispatch(matmul_op()).expect("should dispatch");
        assert_eq!(result.selected_backend, BackendKind::Cpu);
        assert!(result.fallback_used);
    }

    // ------------------------------------------------------------------
    // 4. fallback_used true / false
    // ------------------------------------------------------------------

    #[test]
    fn test_fallback_used_false_when_primary_available() {
        let mut d = TensorOpDispatcher::new();
        d.register_backend(gpu_backend(100));
        d.register_backend(cpu_backend(10));

        let result = d.dispatch(matmul_op()).expect("should dispatch");
        assert!(!result.fallback_used);
    }

    #[test]
    fn test_fallback_used_true_when_skipping_unavailable_capable() {
        let mut d = TensorOpDispatcher::new();
        let mut gpu = gpu_backend(100);
        gpu.is_available = false;
        d.register_backend(gpu);
        d.register_backend(cpu_backend(10));

        let result = d.dispatch(matmul_op()).expect("should dispatch");
        assert!(result.fallback_used);
    }

    #[test]
    fn test_fallback_used_false_when_incapable_backend_skipped() {
        // The GPU does not support "add" — skipping it should NOT count as a
        // fallback.
        let mut d = TensorOpDispatcher::new();
        d.register_backend(BackendRegistration {
            kind: BackendKind::Gpu,
            priority: 200,
            supported_ops: vec!["matmul".to_string()],
            is_available: true,
        });
        d.register_backend(cpu_backend(10));

        let result = d.dispatch(ew_op("add")).expect("should dispatch");
        assert_eq!(result.selected_backend, BackendKind::Cpu);
        assert!(!result.fallback_used);
    }

    // ------------------------------------------------------------------
    // 5. dispatch returns None when no backend found
    // ------------------------------------------------------------------

    #[test]
    fn test_dispatch_none_when_no_backends() {
        let mut d = TensorOpDispatcher::new();
        assert!(d.dispatch(matmul_op()).is_none());
    }

    #[test]
    fn test_dispatch_none_when_all_unavailable() {
        let mut d = TensorOpDispatcher::new();
        let mut gpu = gpu_backend(100);
        gpu.is_available = false;
        let mut cpu = cpu_backend(10);
        cpu.is_available = false;
        d.register_backend(gpu);
        d.register_backend(cpu);

        assert!(d.dispatch(matmul_op()).is_none());
    }

    #[test]
    fn test_dispatch_none_when_op_unsupported_by_all() {
        let mut d = TensorOpDispatcher::new();
        d.register_backend(BackendRegistration {
            kind: BackendKind::Gpu,
            priority: 50,
            supported_ops: vec!["matmul".to_string()],
            is_available: true,
        });
        // "sigmoid" not listed in any backend
        assert!(d.dispatch(ew_op("sigmoid")).is_none());
    }

    // ------------------------------------------------------------------
    // 6. set_backend_available toggles availability
    // ------------------------------------------------------------------

    #[test]
    fn test_set_backend_available_disables() {
        let mut d = TensorOpDispatcher::new();
        d.register_backend(gpu_backend(100));
        d.register_backend(cpu_backend(10));

        d.set_backend_available(BackendKind::Gpu, false);
        let result = d.dispatch(matmul_op()).expect("CPU should be fallback");
        assert_eq!(result.selected_backend, BackendKind::Cpu);
        assert!(result.fallback_used);
    }

    #[test]
    fn test_set_backend_available_re_enables() {
        let mut d = TensorOpDispatcher::new();
        let mut gpu = gpu_backend(100);
        gpu.is_available = false;
        d.register_backend(gpu);
        d.register_backend(cpu_backend(10));

        // GPU is down — CPU selected
        let r1 = d.dispatch(matmul_op()).expect("should dispatch");
        assert_eq!(r1.selected_backend, BackendKind::Cpu);

        // Re-enable GPU
        d.set_backend_available(BackendKind::Gpu, true);
        let r2 = d.dispatch(matmul_op()).expect("should dispatch");
        assert_eq!(r2.selected_backend, BackendKind::Gpu);
        assert!(!r2.fallback_used);
    }

    // ------------------------------------------------------------------
    // 7. MatMul always works on Cpu
    // ------------------------------------------------------------------

    #[test]
    fn test_matmul_always_supported_on_cpu() {
        let reg = cpu_backend(10);
        // Even with an empty supported_ops Cpu still supports MatMul
        let minimal_cpu = BackendRegistration {
            kind: BackendKind::Cpu,
            priority: 10,
            supported_ops: vec![],
            is_available: true,
        };
        assert!(minimal_cpu.supports_op(&matmul_op()));
        assert!(reg.supports_op(&matmul_op()));
    }

    #[test]
    fn test_matmul_always_supported_on_simulated() {
        let minimal_sim = BackendRegistration {
            kind: BackendKind::Simulated,
            priority: 1,
            supported_ops: vec![],
            is_available: true,
        };
        assert!(minimal_sim.supports_op(&matmul_op()));
    }

    #[test]
    fn test_matmul_requires_capability_on_gpu() {
        let gpu_no_mm = BackendRegistration {
            kind: BackendKind::Gpu,
            priority: 10,
            supported_ops: vec!["add".to_string()],
            is_available: true,
        };
        assert!(!gpu_no_mm.supports_op(&matmul_op()));

        let gpu_with_mm = BackendRegistration {
            kind: BackendKind::Gpu,
            priority: 10,
            supported_ops: vec!["matmul".to_string()],
            is_available: true,
        };
        assert!(gpu_with_mm.supports_op(&matmul_op()));
    }

    #[test]
    fn test_matmul_requires_capability_on_remote() {
        let remote_no_mm = BackendRegistration {
            kind: BackendKind::Remote,
            priority: 10,
            supported_ops: vec!["sum".to_string()],
            is_available: true,
        };
        assert!(!remote_no_mm.supports_op(&matmul_op()));
    }

    // ------------------------------------------------------------------
    // 8. ElementWise matches op_name
    // ------------------------------------------------------------------

    #[test]
    fn test_elementwise_matches_op_name() {
        let cpu = cpu_backend(10); // supports "add", "mul", "relu"
        assert!(cpu.supports_op(&ew_op("add")));
        assert!(cpu.supports_op(&ew_op("mul")));
        assert!(cpu.supports_op(&ew_op("relu")));
        assert!(!cpu.supports_op(&ew_op("sigmoid")));
    }

    #[test]
    fn test_dispatch_elementwise_correct_backend() {
        let mut d = TensorOpDispatcher::new();
        // GPU supports "mul" but not "relu"
        d.register_backend(BackendRegistration {
            kind: BackendKind::Gpu,
            priority: 100,
            supported_ops: vec!["matmul".to_string(), "mul".to_string()],
            is_available: true,
        });
        d.register_backend(cpu_backend(10)); // supports "relu"

        let r = d.dispatch(ew_op("relu")).expect("should dispatch");
        assert_eq!(r.selected_backend, BackendKind::Cpu);
        // GPU did not support "relu" — not a capable-but-unavailable skip.
        assert!(!r.fallback_used);
    }

    // ------------------------------------------------------------------
    // 9. Convolution matches "conv"
    // ------------------------------------------------------------------

    #[test]
    fn test_convolution_matches_conv_keyword() {
        let backend = BackendRegistration {
            kind: BackendKind::Gpu,
            priority: 10,
            supported_ops: vec!["conv".to_string(), "matmul".to_string()],
            is_available: true,
        };
        assert!(backend.supports_op(&conv_op()));
    }

    #[test]
    fn test_convolution_fails_without_conv_keyword() {
        let backend = BackendRegistration {
            kind: BackendKind::Gpu,
            priority: 10,
            supported_ops: vec!["matmul".to_string()],
            is_available: true,
        };
        assert!(!backend.supports_op(&conv_op()));
    }

    #[test]
    fn test_dispatch_convolution_selects_correct_backend() {
        let mut d = TensorOpDispatcher::new();
        d.register_backend(BackendRegistration {
            kind: BackendKind::Remote,
            priority: 200,
            supported_ops: vec!["matmul".to_string()], // no conv
            is_available: true,
        });
        d.register_backend(gpu_backend(100)); // has conv

        let r = d.dispatch(conv_op()).expect("GPU should handle conv");
        assert_eq!(r.selected_backend, BackendKind::Gpu);
        assert!(!r.fallback_used);
    }

    // ------------------------------------------------------------------
    // 10. Reduction matches op_name
    // ------------------------------------------------------------------

    #[test]
    fn test_reduction_matches_op_name() {
        let cpu = cpu_backend(10);
        assert!(cpu.supports_op(&red_op("sum")));
        assert!(cpu.supports_op(&red_op("mean")));
        assert!(!cpu.supports_op(&red_op("prod")));
    }

    // ------------------------------------------------------------------
    // 11. dispatcher_stats counts
    // ------------------------------------------------------------------

    #[test]
    fn test_dispatcher_stats_total_dispatched() {
        let mut d = TensorOpDispatcher::new();
        d.register_backend(cpu_backend(10));
        d.dispatch(matmul_op());
        d.dispatch(matmul_op());
        assert_eq!(d.stats().total_dispatched, 2);
    }

    #[test]
    fn test_dispatcher_stats_total_failed() {
        let mut d = TensorOpDispatcher::new();
        d.dispatch(matmul_op()); // no backends
        d.dispatch(ew_op("nonexistent"));
        assert_eq!(d.stats().total_failed, 2);
    }

    #[test]
    fn test_dispatcher_stats_total_fallbacks() {
        let mut d = TensorOpDispatcher::new();
        let mut gpu = gpu_backend(100);
        gpu.is_available = false;
        d.register_backend(gpu);
        d.register_backend(cpu_backend(10));

        d.dispatch(matmul_op()); // fallback
        d.dispatch(matmul_op()); // fallback again
        assert_eq!(d.stats().total_fallbacks, 2);
    }

    #[test]
    fn test_dispatcher_stats_mixed_outcomes() {
        let mut d = TensorOpDispatcher::new();
        d.register_backend(gpu_backend(100));
        d.register_backend(cpu_backend(10));

        d.dispatch(matmul_op()); // normal
        d.set_backend_available(BackendKind::Gpu, false);
        d.dispatch(matmul_op()); // fallback
        d.dispatch(ew_op("sigmoid")); // failed (unsupported)

        assert_eq!(d.stats().total_dispatched, 2);
        assert_eq!(d.stats().total_fallbacks, 1);
        assert_eq!(d.stats().total_failed, 1);
    }

    // ------------------------------------------------------------------
    // 12. backend_stats per backend
    // ------------------------------------------------------------------

    #[test]
    fn test_backend_stats_dispatched_count() {
        let mut d = TensorOpDispatcher::new();
        d.register_backend(cpu_backend(10));

        d.dispatch(matmul_op());
        d.dispatch(ew_op("add"));
        let stats = d.backend_stats(BackendKind::Cpu).expect("stats exist");
        assert_eq!(stats.total_dispatched, 2);
    }

    #[test]
    fn test_backend_stats_fallback_count() {
        let mut d = TensorOpDispatcher::new();
        let mut gpu = gpu_backend(100);
        gpu.is_available = false;
        d.register_backend(gpu);
        d.register_backend(cpu_backend(10));

        d.dispatch(matmul_op()); // CPU as fallback
        d.dispatch(matmul_op()); // CPU as fallback again

        let cpu_stats = d.backend_stats(BackendKind::Cpu).expect("CPU stats");
        assert_eq!(cpu_stats.total_dispatched, 2);
        assert_eq!(cpu_stats.total_fallback_selected, 2);

        // GPU was never selected
        let gpu_stats = d.backend_stats(BackendKind::Gpu).expect("GPU stats");
        assert_eq!(gpu_stats.total_dispatched, 0);
    }

    #[test]
    fn test_backend_stats_none_for_unregistered() {
        let d = TensorOpDispatcher::new();
        assert!(d.backend_stats(BackendKind::Remote).is_none());
    }

    #[test]
    fn test_backend_stats_independent_per_backend() {
        let mut d = TensorOpDispatcher::new();
        d.register_backend(gpu_backend(100));
        d.register_backend(cpu_backend(10));

        // MatMul → GPU (priority wins)
        d.dispatch(matmul_op());
        // Disable GPU → CPU fallback
        d.set_backend_available(BackendKind::Gpu, false);
        d.dispatch(matmul_op());

        let gpu_stats = d
            .backend_stats(BackendKind::Gpu)
            .expect("test: should succeed");
        assert_eq!(gpu_stats.total_dispatched, 1);
        assert_eq!(gpu_stats.total_fallback_selected, 0);

        let cpu_stats = d
            .backend_stats(BackendKind::Cpu)
            .expect("test: should succeed");
        assert_eq!(cpu_stats.total_dispatched, 1);
        assert_eq!(cpu_stats.total_fallback_selected, 1);
    }

    // ------------------------------------------------------------------
    // 13. Simulated backend
    // ------------------------------------------------------------------

    #[test]
    fn test_simulated_backend_matmul_always_supported() {
        let mut d = TensorOpDispatcher::new();
        d.register_backend(sim_backend(5));

        let r = d
            .dispatch(matmul_op())
            .expect("simulated should handle matmul");
        assert_eq!(r.selected_backend, BackendKind::Simulated);
        assert!(!r.fallback_used);
    }

    #[test]
    fn test_simulated_backend_elementwise_by_op_name() {
        let mut d = TensorOpDispatcher::new();
        d.register_backend(sim_backend(5)); // supports "add", "mul", "sum", "conv"

        let r = d
            .dispatch(ew_op("add"))
            .expect("simulated should handle add");
        assert_eq!(r.selected_backend, BackendKind::Simulated);

        assert!(d.dispatch(ew_op("relu")).is_none()); // not in sim supported_ops
    }

    // ------------------------------------------------------------------
    // 14. Dispatch result carries the original op
    // ------------------------------------------------------------------

    #[test]
    fn test_dispatch_result_carries_op() {
        let mut d = TensorOpDispatcher::new();
        d.register_backend(cpu_backend(10));

        let op = DispatchOp::MatMul { m: 8, n: 16, k: 32 };
        let r = d.dispatch(op.clone()).expect("should dispatch");
        assert_eq!(r.op, op);
    }

    // ------------------------------------------------------------------
    // 15. Multiple backends of same kind (e.g., two CPU registrations)
    // ------------------------------------------------------------------

    #[test]
    fn test_multiple_registrations_same_kind_priority_wins() {
        let mut d = TensorOpDispatcher::new();
        d.register_backend(cpu_backend(20)); // higher priority CPU
        d.register_backend(cpu_backend(10)); // lower priority CPU

        // First registration (priority 20) should be selected.
        let r = d.dispatch(matmul_op()).expect("should dispatch");
        assert_eq!(r.selected_backend, BackendKind::Cpu);
        assert!(!r.fallback_used);
        // Exactly one op dispatched to CPU
        assert_eq!(
            d.backend_stats(BackendKind::Cpu)
                .expect("test: should succeed")
                .total_dispatched,
            1
        );
    }
}
