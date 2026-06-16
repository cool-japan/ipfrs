//! Garbage collector for unreachable tensor allocations.
//!
//! Identifies and collects unreachable tensor allocations using reference
//! counting and reachability analysis from named roots. The collector
//! implements a classic mark-and-sweep algorithm operating in three phases:
//!
//! 1. **MarkRoots** — seed the reachable set with explicit roots and pinned tensors.
//! 2. **Trace** — BFS-expand the reachable set through dependency edges.
//! 3. **Sweep** — remove every tensor that is neither reachable nor ref-counted.
//!
//! # Example
//!
//! ```
//! use ipfrs_tensorlogic::tensor_gc::{TensorGarbageCollector, TensorRef};
//!
//! let mut gc = TensorGarbageCollector::new();
//!
//! // Register two tensors: A depends on B.
//! gc.register(TensorRef {
//!     tensor_id: 1,
//!     name: Some("A".to_string()),
//!     size_bytes: 1024,
//!     ref_count: 0,
//!     dependencies: vec![2],
//!     pinned: false,
//! });
//! gc.register(TensorRef {
//!     tensor_id: 2,
//!     name: Some("B".to_string()),
//!     size_bytes: 512,
//!     ref_count: 0,
//!     dependencies: vec![],
//!     pinned: false,
//! });
//!
//! // Make A a root; B is reachable through A's dependency edge.
//! gc.add_root(1);
//! let stats = gc.collect();
//! assert_eq!(stats.collected, 0);
//! assert_eq!(stats.reachable, 2);
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// TensorRef
// ---------------------------------------------------------------------------

/// Descriptor for a single tensor allocation tracked by the GC.
#[derive(Debug, Clone)]
pub struct TensorRef {
    /// Unique identifier for this tensor.
    pub tensor_id: u64,
    /// Optional human-readable name.
    pub name: Option<String>,
    /// Size of the tensor allocation in bytes.
    pub size_bytes: u64,
    /// External reference count.  A tensor with `ref_count > 0` is never
    /// collected even if it is unreachable from the root set.
    pub ref_count: u32,
    /// IDs of tensors that this tensor depends on (outgoing dependency edges).
    pub dependencies: Vec<u64>,
    /// If `true` the tensor is never collected regardless of reachability or
    /// reference count.
    pub pinned: bool,
}

// ---------------------------------------------------------------------------
// GcPhase
// ---------------------------------------------------------------------------

/// The current phase of a mark-and-sweep garbage collection cycle.
#[derive(Clone, Debug, PartialEq)]
pub enum GcPhase {
    /// Phase 1 — seed the reachable set from roots and pinned tensors.
    MarkRoots,
    /// Phase 2 — BFS/DFS expansion of the reachable set through dependency edges.
    Trace,
    /// Phase 3 — remove tensors absent from the reachable set that have no
    /// external references.
    Sweep,
}

// ---------------------------------------------------------------------------
// GcStats
// ---------------------------------------------------------------------------

/// Statistics produced at the end of a collection cycle.
#[derive(Debug, Clone, Default)]
pub struct GcStats {
    /// Total number of tensors registered at the time of collection.
    pub total_tensors: usize,
    /// Number of tensors that were reachable from roots or pinned.
    pub reachable: usize,
    /// Number of tensors reclaimed during the sweep phase.
    pub collected: usize,
    /// Total bytes freed during the sweep phase.
    pub freed_bytes: u64,
    /// Number of tensors with `pinned == true`.
    pub pinned_tensors: usize,
}

impl GcStats {
    /// Fraction of registered tensors that were collected.
    ///
    /// Returns a value in `[0.0, 1.0]`.  Returns `0.0` when there are no
    /// registered tensors.
    pub fn collection_rate(&self) -> f64 {
        self.collected as f64 / self.total_tensors.max(1) as f64
    }
}

// ---------------------------------------------------------------------------
// TensorGarbageCollector
// ---------------------------------------------------------------------------

/// Mark-and-sweep garbage collector for tensor allocations.
///
/// Tracks a set of [`TensorRef`] descriptors connected by dependency edges,
/// and periodically reclaims those that are unreachable from the named root
/// set and carry no external references.
pub struct TensorGarbageCollector {
    /// All registered tensor descriptors keyed by `tensor_id`.
    pub tensors: HashMap<u64, TensorRef>,
    /// Tensor IDs that are always considered reachable (GC roots).
    pub roots: Vec<u64>,
}

impl TensorGarbageCollector {
    /// Create a new, empty garbage collector.
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            roots: Vec::new(),
        }
    }

    /// Register a tensor with the collector.
    ///
    /// If a tensor with the same `tensor_id` already exists it is replaced.
    pub fn register(&mut self, tensor: TensorRef) {
        self.tensors.insert(tensor.tensor_id, tensor);
    }

    /// Add `tensor_id` to the GC root set.
    ///
    /// Roots are always considered reachable regardless of incoming edges.
    /// Duplicates are silently ignored.
    pub fn add_root(&mut self, tensor_id: u64) {
        if !self.roots.contains(&tensor_id) {
            self.roots.push(tensor_id);
        }
    }

    /// Remove `tensor_id` from the GC root set.
    ///
    /// Does nothing if the ID is not present.
    pub fn remove_root(&mut self, tensor_id: u64) {
        self.roots.retain(|&id| id != tensor_id);
    }

    /// Mark a registered tensor as pinned so it is never collected.
    ///
    /// Does nothing if `tensor_id` is not registered.
    pub fn pin(&mut self, tensor_id: u64) {
        if let Some(t) = self.tensors.get_mut(&tensor_id) {
            t.pinned = true;
        }
    }

    /// Increment the external reference count for `tensor_id`.
    ///
    /// Does nothing if `tensor_id` is not registered.
    pub fn add_ref(&mut self, tensor_id: u64) {
        if let Some(t) = self.tensors.get_mut(&tensor_id) {
            t.ref_count = t.ref_count.saturating_add(1);
        }
    }

    /// Decrement the external reference count for `tensor_id` (saturating at 0).
    ///
    /// Does nothing if `tensor_id` is not registered.
    pub fn remove_ref(&mut self, tensor_id: u64) {
        if let Some(t) = self.tensors.get_mut(&tensor_id) {
            t.ref_count = t.ref_count.saturating_sub(1);
        }
    }

    /// Compute the set of tensor IDs reachable from roots and pinned tensors
    /// without mutating collector state.
    ///
    /// Reachability is determined by BFS expansion through
    /// [`TensorRef::dependencies`] edges.
    pub fn reachable_set(&self) -> Vec<u64> {
        let mut visited: HashSet<u64> = HashSet::new();
        let mut queue: VecDeque<u64> = VecDeque::new();

        // Phase 1 — seed from roots.
        for &id in &self.roots {
            if self.tensors.contains_key(&id) && visited.insert(id) {
                queue.push_back(id);
            }
        }

        // Seed from pinned tensors.
        for (id, tensor) in &self.tensors {
            if tensor.pinned && visited.insert(*id) {
                queue.push_back(*id);
            }
        }

        // Phase 2 — BFS trace through dependency edges.
        while let Some(current) = queue.pop_front() {
            if let Some(tensor) = self.tensors.get(&current) {
                for &dep in &tensor.dependencies {
                    if self.tensors.contains_key(&dep) && visited.insert(dep) {
                        queue.push_back(dep);
                    }
                }
            }
        }

        visited.into_iter().collect()
    }

    /// Run a complete mark-and-sweep garbage collection cycle.
    ///
    /// The collection proceeds through three [`GcPhase`]s:
    ///
    /// * [`GcPhase::MarkRoots`] — seed the reachable set from roots and pinned tensors.
    /// * [`GcPhase::Trace`] — BFS-expand through dependency edges.
    /// * [`GcPhase::Sweep`] — remove unreachable tensors with `ref_count == 0`.
    ///
    /// Returns [`GcStats`] describing the result.
    pub fn collect(&mut self) -> GcStats {
        let total_tensors = self.tensors.len();
        let pinned_tensors = self.tensors.values().filter(|t| t.pinned).count();

        // ----- Phase 1: MarkRoots -----
        let _phase = GcPhase::MarkRoots;
        let mut visited: HashSet<u64> = HashSet::new();
        let mut queue: VecDeque<u64> = VecDeque::new();

        for &id in &self.roots {
            if self.tensors.contains_key(&id) && visited.insert(id) {
                queue.push_back(id);
            }
        }

        for (id, tensor) in &self.tensors {
            if tensor.pinned && visited.insert(*id) {
                queue.push_back(*id);
            }
        }

        // ----- Phase 2: Trace -----
        let _phase = GcPhase::Trace;
        while let Some(current) = queue.pop_front() {
            if let Some(tensor) = self.tensors.get(&current) {
                for &dep in &tensor.dependencies {
                    if self.tensors.contains_key(&dep) && visited.insert(dep) {
                        queue.push_back(dep);
                    }
                }
            }
        }

        let reachable = visited.len();

        // ----- Phase 3: Sweep -----
        let _phase = GcPhase::Sweep;
        let mut collected = 0usize;
        let mut freed_bytes = 0u64;

        // Collect the IDs to remove first to avoid borrow conflicts.
        let to_remove: Vec<u64> = self
            .tensors
            .iter()
            .filter_map(|(id, tensor)| {
                if !visited.contains(id) && tensor.ref_count == 0 {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect();

        for id in to_remove {
            if let Some(tensor) = self.tensors.remove(&id) {
                freed_bytes += tensor.size_bytes;
                collected += 1;
            }
        }

        GcStats {
            total_tensors,
            reachable,
            collected,
            freed_bytes,
            pinned_tensors,
        }
    }
}

impl Default for TensorGarbageCollector {
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

    /// Helper to build a simple TensorRef with no dependencies.
    fn make_tensor(id: u64, size: u64) -> TensorRef {
        TensorRef {
            tensor_id: id,
            name: None,
            size_bytes: size,
            ref_count: 0,
            dependencies: vec![],
            pinned: false,
        }
    }

    // ------------------------------------------------------------------
    // 1. register
    // ------------------------------------------------------------------

    #[test]
    fn test_register_inserts_tensor() {
        let mut gc = TensorGarbageCollector::new();
        gc.register(make_tensor(1, 128));
        assert!(gc.tensors.contains_key(&1));
    }

    #[test]
    fn test_register_overwrites_existing() {
        let mut gc = TensorGarbageCollector::new();
        gc.register(make_tensor(1, 128));
        gc.register(TensorRef {
            tensor_id: 1,
            name: Some("updated".to_string()),
            size_bytes: 256,
            ref_count: 0,
            dependencies: vec![],
            pinned: false,
        });
        assert_eq!(gc.tensors[&1].size_bytes, 256);
        assert_eq!(gc.tensors[&1].name.as_deref(), Some("updated"));
    }

    // ------------------------------------------------------------------
    // 2. add_root / remove_root
    // ------------------------------------------------------------------

    #[test]
    fn test_add_root_appends() {
        let mut gc = TensorGarbageCollector::new();
        gc.add_root(42);
        assert!(gc.roots.contains(&42));
    }

    #[test]
    fn test_add_root_no_duplicates() {
        let mut gc = TensorGarbageCollector::new();
        gc.add_root(7);
        gc.add_root(7);
        assert_eq!(gc.roots.iter().filter(|&&id| id == 7).count(), 1);
    }

    #[test]
    fn test_remove_root_removes_entry() {
        let mut gc = TensorGarbageCollector::new();
        gc.add_root(5);
        gc.add_root(6);
        gc.remove_root(5);
        assert!(!gc.roots.contains(&5));
        assert!(gc.roots.contains(&6));
    }

    #[test]
    fn test_remove_root_nonexistent_is_noop() {
        let mut gc = TensorGarbageCollector::new();
        gc.add_root(3);
        gc.remove_root(999); // should not panic
        assert_eq!(gc.roots.len(), 1);
    }

    // ------------------------------------------------------------------
    // 3. collect: unreachable tensors are removed
    // ------------------------------------------------------------------

    #[test]
    fn test_collect_removes_unreachable() {
        let mut gc = TensorGarbageCollector::new();
        gc.register(make_tensor(1, 100)); // root
        gc.register(make_tensor(2, 200)); // unreachable
        gc.add_root(1);

        let stats = gc.collect();

        assert_eq!(stats.collected, 1);
        assert_eq!(stats.freed_bytes, 200);
        assert!(gc.tensors.contains_key(&1));
        assert!(!gc.tensors.contains_key(&2));
    }

    #[test]
    fn test_collect_keeps_reachable_tensors() {
        let mut gc = TensorGarbageCollector::new();
        gc.register(TensorRef {
            tensor_id: 1,
            name: None,
            size_bytes: 50,
            ref_count: 0,
            dependencies: vec![2],
            pinned: false,
        });
        gc.register(make_tensor(2, 75));
        gc.add_root(1);

        let stats = gc.collect();
        assert_eq!(stats.collected, 0);
        assert!(gc.tensors.contains_key(&2));
    }

    // ------------------------------------------------------------------
    // 4. pinned tensors are never collected
    // ------------------------------------------------------------------

    #[test]
    fn test_pinned_never_collected() {
        let mut gc = TensorGarbageCollector::new();
        gc.register(TensorRef {
            tensor_id: 10,
            name: None,
            size_bytes: 512,
            ref_count: 0,
            dependencies: vec![],
            pinned: true,
        });
        // No roots — tensor 10 is only reachable because it is pinned.
        let stats = gc.collect();
        assert_eq!(stats.collected, 0);
        assert!(gc.tensors.contains_key(&10));
    }

    #[test]
    fn test_pin_method_marks_tensor() {
        let mut gc = TensorGarbageCollector::new();
        gc.register(make_tensor(99, 1024));
        assert!(!gc.tensors[&99].pinned);
        gc.pin(99);
        assert!(gc.tensors[&99].pinned);
    }

    #[test]
    fn test_pin_nonexistent_is_noop() {
        let mut gc = TensorGarbageCollector::new();
        gc.pin(1234); // must not panic
    }

    // ------------------------------------------------------------------
    // 5. ref_count > 0 prevents collection
    // ------------------------------------------------------------------

    #[test]
    fn test_ref_count_prevents_collection() {
        let mut gc = TensorGarbageCollector::new();
        gc.register(TensorRef {
            tensor_id: 5,
            name: None,
            size_bytes: 300,
            ref_count: 1, // externally referenced
            dependencies: vec![],
            pinned: false,
        });
        // No roots — tensor 5 would be unreachable but ref_count keeps it alive.
        let stats = gc.collect();
        assert_eq!(stats.collected, 0);
        assert!(gc.tensors.contains_key(&5));
    }

    // ------------------------------------------------------------------
    // 6. BFS traces multi-level dependency chains
    // ------------------------------------------------------------------

    #[test]
    fn test_bfs_traces_multi_level_chain() {
        // Chain: root(1) → 2 → 3 → 4  plus isolated 5
        let mut gc = TensorGarbageCollector::new();
        for (id, dep) in [(1u64, Some(2u64)), (2, Some(3)), (3, Some(4)), (4, None)] {
            gc.register(TensorRef {
                tensor_id: id,
                name: None,
                size_bytes: id * 10,
                ref_count: 0,
                dependencies: dep.into_iter().collect(),
                pinned: false,
            });
        }
        gc.register(make_tensor(5, 999)); // isolated
        gc.add_root(1);

        let stats = gc.collect();
        assert_eq!(stats.reachable, 4);
        assert_eq!(stats.collected, 1);
        assert_eq!(stats.freed_bytes, 999);
        assert!(!gc.tensors.contains_key(&5));
    }

    // ------------------------------------------------------------------
    // 7. reachable_set is correct and non-mutating
    // ------------------------------------------------------------------

    #[test]
    fn test_reachable_set_correct() {
        let mut gc = TensorGarbageCollector::new();
        gc.register(TensorRef {
            tensor_id: 1,
            name: None,
            size_bytes: 10,
            ref_count: 0,
            dependencies: vec![2],
            pinned: false,
        });
        gc.register(make_tensor(2, 20));
        gc.register(make_tensor(3, 30)); // unreachable
        gc.add_root(1);

        let mut reachable = gc.reachable_set();
        reachable.sort();
        assert_eq!(reachable, vec![1, 2]);
        // State unchanged (tensors still present)
        assert_eq!(gc.tensors.len(), 3);
    }

    #[test]
    fn test_reachable_set_includes_pinned() {
        let mut gc = TensorGarbageCollector::new();
        gc.register(make_tensor(1, 10));
        gc.register(TensorRef {
            tensor_id: 2,
            name: None,
            size_bytes: 20,
            ref_count: 0,
            dependencies: vec![],
            pinned: true,
        });
        // No roots; tensor 2 is pinned.
        let reachable = gc.reachable_set();
        assert!(reachable.contains(&2));
        assert!(!reachable.contains(&1));
    }

    // ------------------------------------------------------------------
    // 8. GcStats fields and collection_rate
    // ------------------------------------------------------------------

    #[test]
    fn test_gc_stats_fields() {
        let mut gc = TensorGarbageCollector::new();
        gc.register(make_tensor(1, 100));
        gc.register(make_tensor(2, 200));
        gc.register(TensorRef {
            tensor_id: 3,
            name: None,
            size_bytes: 50,
            ref_count: 0,
            dependencies: vec![],
            pinned: true,
        });
        gc.add_root(1);

        let stats = gc.collect();
        assert_eq!(stats.total_tensors, 3);
        assert_eq!(stats.reachable, 2); // 1 (root) + 3 (pinned)
        assert_eq!(stats.collected, 1); // tensor 2
        assert_eq!(stats.freed_bytes, 200);
        assert_eq!(stats.pinned_tensors, 1);
    }

    #[test]
    fn test_collection_rate_correct() {
        let stats = GcStats {
            total_tensors: 10,
            reachable: 6,
            collected: 4,
            freed_bytes: 0,
            pinned_tensors: 0,
        };
        let rate = stats.collection_rate();
        assert!((rate - 0.4).abs() < f64::EPSILON * 10.0);
    }

    #[test]
    fn test_collection_rate_zero_total() {
        let stats = GcStats::default();
        // Must not divide by zero.
        assert_eq!(stats.collection_rate(), 0.0);
    }

    // ------------------------------------------------------------------
    // 9. add_ref / remove_ref
    // ------------------------------------------------------------------

    #[test]
    fn test_add_ref_increments() {
        let mut gc = TensorGarbageCollector::new();
        gc.register(make_tensor(1, 64));
        assert_eq!(gc.tensors[&1].ref_count, 0);
        gc.add_ref(1);
        assert_eq!(gc.tensors[&1].ref_count, 1);
        gc.add_ref(1);
        assert_eq!(gc.tensors[&1].ref_count, 2);
    }

    #[test]
    fn test_remove_ref_decrements_saturating() {
        let mut gc = TensorGarbageCollector::new();
        gc.register(make_tensor(1, 64));
        gc.add_ref(1);
        gc.remove_ref(1);
        assert_eq!(gc.tensors[&1].ref_count, 0);
        // Saturating — should not underflow.
        gc.remove_ref(1);
        assert_eq!(gc.tensors[&1].ref_count, 0);
    }

    #[test]
    fn test_add_ref_remove_ref_nonexistent_is_noop() {
        let mut gc = TensorGarbageCollector::new();
        gc.add_ref(9999); // must not panic
        gc.remove_ref(9999); // must not panic
    }

    // ------------------------------------------------------------------
    // 10. Diamond dependency graph
    // ------------------------------------------------------------------

    #[test]
    fn test_diamond_dependency_all_reachable() {
        // Diamond: root(1) → {2, 3} → 4
        let mut gc = TensorGarbageCollector::new();
        gc.register(TensorRef {
            tensor_id: 1,
            name: None,
            size_bytes: 10,
            ref_count: 0,
            dependencies: vec![2, 3],
            pinned: false,
        });
        gc.register(TensorRef {
            tensor_id: 2,
            name: None,
            size_bytes: 20,
            ref_count: 0,
            dependencies: vec![4],
            pinned: false,
        });
        gc.register(TensorRef {
            tensor_id: 3,
            name: None,
            size_bytes: 30,
            ref_count: 0,
            dependencies: vec![4],
            pinned: false,
        });
        gc.register(make_tensor(4, 40));
        gc.add_root(1);

        let stats = gc.collect();
        assert_eq!(stats.reachable, 4);
        assert_eq!(stats.collected, 0);
    }

    // ------------------------------------------------------------------
    // 11. Multiple roots
    // ------------------------------------------------------------------

    #[test]
    fn test_multiple_roots() {
        let mut gc = TensorGarbageCollector::new();
        gc.register(make_tensor(1, 10));
        gc.register(make_tensor(2, 20));
        gc.register(make_tensor(3, 30)); // unreachable
        gc.add_root(1);
        gc.add_root(2);

        let stats = gc.collect();
        assert_eq!(stats.reachable, 2);
        assert_eq!(stats.collected, 1);
    }

    // ------------------------------------------------------------------
    // 12. Remove root makes previously reachable tensor collectable
    // ------------------------------------------------------------------

    #[test]
    fn test_remove_root_enables_collection() {
        let mut gc = TensorGarbageCollector::new();
        gc.register(make_tensor(1, 100));
        gc.register(make_tensor(2, 200));
        gc.add_root(1);
        gc.add_root(2);

        // First collect — both survive.
        let s1 = gc.collect();
        assert_eq!(s1.collected, 0);

        // Remove root 2 — tensor 2 is now unreachable.
        gc.remove_root(2);
        let s2 = gc.collect();
        assert_eq!(s2.collected, 1);
        assert!(!gc.tensors.contains_key(&2));
    }

    // ------------------------------------------------------------------
    // 13. Named tensors
    // ------------------------------------------------------------------

    #[test]
    fn test_named_tensor_survives_as_root() {
        let mut gc = TensorGarbageCollector::new();
        gc.register(TensorRef {
            tensor_id: 7,
            name: Some("embedding_weights".to_string()),
            size_bytes: 4096,
            ref_count: 0,
            dependencies: vec![],
            pinned: false,
        });
        gc.add_root(7);
        let stats = gc.collect();
        assert_eq!(stats.collected, 0);
        assert_eq!(gc.tensors[&7].name.as_deref(), Some("embedding_weights"));
    }

    // ------------------------------------------------------------------
    // 14. Empty GC collect
    // ------------------------------------------------------------------

    #[test]
    fn test_empty_gc_collect_returns_zero_stats() {
        let mut gc = TensorGarbageCollector::new();
        let stats = gc.collect();
        assert_eq!(stats.total_tensors, 0);
        assert_eq!(stats.reachable, 0);
        assert_eq!(stats.collected, 0);
        assert_eq!(stats.freed_bytes, 0);
        assert_eq!(stats.pinned_tensors, 0);
    }

    // ------------------------------------------------------------------
    // 15. GcPhase derives
    // ------------------------------------------------------------------

    #[test]
    fn test_gc_phase_clone_debug_partialeq() {
        let p = GcPhase::Trace;
        let q = p.clone();
        assert_eq!(p, q);
        let r = GcPhase::Sweep;
        assert_ne!(p, r);
        let debug_str = format!("{:?}", GcPhase::MarkRoots);
        assert!(debug_str.contains("MarkRoots"));
    }
}
