//! Cache eviction policy simulator.
//!
//! Simulates LRU, LFU, and ARC-approximation eviction policies against an
//! access trace and compares their hit rates.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Eviction policy to simulate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least-recently used.
    Lru,
    /// Least-frequently used (tie-break: LRU among equal freq).
    Lfu,
    /// Simplified ARC: T1 (recency) + T2 (frequency).
    ArcApprox,
}

/// A single cache access event in the trace.
#[derive(Clone, Debug)]
pub struct AccessEvent {
    /// Content identifier (opaque string key).
    pub cid: String,
    /// Logical clock tick for ordering.
    pub timestamp_tick: u64,
}

/// Result of simulating one eviction policy against a trace.
#[derive(Clone, Debug)]
pub struct SimulationResult {
    pub policy: EvictionPolicy,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl SimulationResult {
    /// Fraction of accesses that were cache hits (0.0 – 1.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Fraction of accesses that were cache misses (0.0 – 1.0).
    pub fn miss_rate(&self) -> f64 {
        1.0 - self.hit_rate()
    }
}

// ---------------------------------------------------------------------------
// Internal simulation state
// ---------------------------------------------------------------------------

/// Internal state threaded through each step of the simulation.
struct CacheState {
    capacity: usize,
    /// Ordered eviction queue for LRU/LFU: index 0 is evicted first.
    items: Vec<String>,
    /// Per-item access frequency (used by LFU).
    freq: HashMap<String, u64>,
    /// ARC T1 list (recency queue).
    t1: Vec<String>,
    /// ARC T2 list (frequency queue).
    t2: Vec<String>,
}

impl CacheState {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            items: Vec::new(),
            freq: HashMap::new(),
            t1: Vec::new(),
            t2: Vec::new(),
        }
    }

    // -----------------------------------------------------------------------
    // LRU helpers
    // -----------------------------------------------------------------------

    #[cfg(test)]
    #[allow(dead_code)]
    fn lru_contains(&self, cid: &str) -> bool {
        self.items.iter().any(|x| x == cid)
    }

    /// Record an LRU access.  Returns `true` when it was a hit.
    fn lru_access(&mut self, cid: &str) -> (bool, u64) {
        let mut evictions = 0u64;
        if let Some(pos) = self.items.iter().position(|x| x == cid) {
            // Hit — move to back (most-recently used).
            self.items.remove(pos);
            self.items.push(cid.to_owned());
            (true, evictions)
        } else {
            // Miss — evict if full.
            if self.items.len() >= self.capacity {
                self.items.remove(0);
                evictions += 1;
            }
            self.items.push(cid.to_owned());
            (false, evictions)
        }
    }

    // -----------------------------------------------------------------------
    // LFU helpers
    // -----------------------------------------------------------------------

    fn lfu_contains(&self, cid: &str) -> bool {
        self.items.iter().any(|x| x == cid)
    }

    /// Record an LFU access.  Returns `(hit, evictions)`.
    fn lfu_access(&mut self, cid: &str) -> (bool, u64) {
        let mut evictions = 0u64;
        if self.lfu_contains(cid) {
            // Hit — increment frequency, keep position (tie-break by insertion order).
            *self.freq.entry(cid.to_owned()).or_insert(0) += 1;
            (true, evictions)
        } else {
            // Miss.
            if self.items.len() >= self.capacity {
                // Evict item with minimum frequency; among ties use earliest
                // position in `items` (index 0 wins).
                let evict_pos = self.lfu_evict_pos();
                let evicted = self.items.remove(evict_pos);
                self.freq.remove(&evicted);
                evictions += 1;
            }
            self.freq.insert(cid.to_owned(), 1);
            self.items.push(cid.to_owned());
            (false, evictions)
        }
    }

    /// Index of the item to evict under LFU policy.
    fn lfu_evict_pos(&self) -> usize {
        let mut best_pos = 0usize;
        let mut best_freq = u64::MAX;
        for (pos, item) in self.items.iter().enumerate() {
            let f = *self.freq.get(item).unwrap_or(&0);
            if f < best_freq {
                best_freq = f;
                best_pos = pos;
            }
            // Equal-frequency ties are broken by earliest position, which
            // is already captured by iterating front-to-back and using
            // strict `<` for replacement.
        }
        best_pos
    }

    // -----------------------------------------------------------------------
    // ARC-approximation helpers
    // -----------------------------------------------------------------------

    #[cfg(test)]
    #[allow(dead_code)]
    fn arc_contains(&self, cid: &str) -> bool {
        self.t1.iter().any(|x| x == cid) || self.t2.iter().any(|x| x == cid)
    }

    /// Record an ARC-approx access.  Returns `(hit, evictions)`.
    fn arc_access(&mut self, cid: &str) -> (bool, u64) {
        let mut evictions = 0u64;

        // Check T2 first (promoted / frequent items).
        if let Some(pos) = self.t2.iter().position(|x| x == cid) {
            // Hit in T2 — refresh to back of T2.
            self.t2.remove(pos);
            self.t2.push(cid.to_owned());
            return (true, evictions);
        }

        // Check T1 (first-time / recent items).
        if let Some(pos) = self.t1.iter().position(|x| x == cid) {
            // Hit in T1 — promote to T2.
            self.t1.remove(pos);
            // Make room in T2 if total capacity is exceeded.
            let total = self.t1.len() + self.t2.len();
            if total >= self.capacity {
                evictions += self.arc_evict();
            }
            self.t2.push(cid.to_owned());
            return (true, evictions);
        }

        // Miss — insert into T1.
        let total = self.t1.len() + self.t2.len();
        if total >= self.capacity {
            evictions += self.arc_evict();
        }
        self.t1.push(cid.to_owned());
        (false, evictions)
    }

    /// Evict one item following the simplified ARC rule.
    /// Evict from T1 if `T1.len() > capacity / 2`, else from T2.
    fn arc_evict(&mut self) -> u64 {
        let half = self.capacity / 2;
        if self.t1.len() > half && !self.t1.is_empty() {
            self.t1.remove(0);
            return 1;
        }
        if !self.t2.is_empty() {
            self.t2.remove(0);
            return 1;
        }
        // Fallback: evict from T1 even if not exceeding half.
        if !self.t1.is_empty() {
            self.t1.remove(0);
            return 1;
        }
        0
    }
}

// ---------------------------------------------------------------------------
// Simulator
// ---------------------------------------------------------------------------

/// Simulates cache eviction policies against access traces.
pub struct CacheEvictionSimulator {
    pub capacity: usize,
}

impl CacheEvictionSimulator {
    /// Create a new simulator with the given cache capacity.
    pub fn new(capacity: usize) -> Self {
        Self { capacity }
    }

    /// Replay `trace` against a fresh `CacheState` using `policy`.
    pub fn simulate(&self, policy: EvictionPolicy, trace: &[AccessEvent]) -> SimulationResult {
        let mut state = CacheState::new(self.capacity);
        let mut hits = 0u64;
        let mut misses = 0u64;
        let mut evictions = 0u64;

        for event in trace {
            let (hit, ev) = match policy {
                EvictionPolicy::Lru => state.lru_access(&event.cid),
                EvictionPolicy::Lfu => state.lfu_access(&event.cid),
                EvictionPolicy::ArcApprox => state.arc_access(&event.cid),
            };
            if hit {
                hits += 1;
            } else {
                misses += 1;
            }
            evictions += ev;
        }

        SimulationResult {
            policy,
            hits,
            misses,
            evictions,
        }
    }

    /// Simulate all three policies and return results sorted by hit rate (descending).
    pub fn compare_policies(&self, trace: &[AccessEvent]) -> Vec<SimulationResult> {
        let mut results = vec![
            self.simulate(EvictionPolicy::Lru, trace),
            self.simulate(EvictionPolicy::Lfu, trace),
            self.simulate(EvictionPolicy::ArcApprox, trace),
        ];
        results.sort_by(|a, b| {
            b.hit_rate()
                .partial_cmp(&a.hit_rate())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a trace from a slice of (&str, u64) pairs.
    fn make_trace(pairs: &[(&str, u64)]) -> Vec<AccessEvent> {
        pairs
            .iter()
            .map(|(cid, tick)| AccessEvent {
                cid: cid.to_string(),
                timestamp_tick: *tick,
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // LRU tests
    // -----------------------------------------------------------------------

    #[test]
    fn lru_hit_basic() {
        let sim = CacheEvictionSimulator::new(3);
        // a, b, c load; then a is a hit.
        let trace = make_trace(&[("a", 1), ("b", 2), ("c", 3), ("a", 4)]);
        let res = sim.simulate(EvictionPolicy::Lru, &trace);
        assert_eq!(res.hits, 1);
        assert_eq!(res.misses, 3);
    }

    #[test]
    fn lru_miss_and_eviction() {
        let sim = CacheEvictionSimulator::new(2);
        // Load a, b (cap full). Access c → evicts a (LRU). Then access a → miss again.
        let trace = make_trace(&[("a", 1), ("b", 2), ("c", 3), ("a", 4)]);
        let res = sim.simulate(EvictionPolicy::Lru, &trace);
        assert_eq!(res.hits, 0);
        assert_eq!(res.evictions, 2); // evicts a then b
        assert_eq!(res.misses, 4);
    }

    #[test]
    fn lru_order_after_access() {
        // Capacity 2: load a, b. Access a (hit → a is now MRU). Load c → evicts b.
        let sim = CacheEvictionSimulator::new(2);
        let trace = make_trace(&[("a", 1), ("b", 2), ("a", 3), ("c", 4), ("b", 5)]);
        let res = sim.simulate(EvictionPolicy::Lru, &trace);
        // hits: a@3, (b should be evicted at c@4), so b@5 is miss
        assert_eq!(res.hits, 1); // only a@3
        assert_eq!(res.evictions, 2); // b evicted when c loaded; a evicted when b reloaded
    }

    #[test]
    fn lru_hit_rate_and_miss_rate() {
        let sim = CacheEvictionSimulator::new(3);
        let trace = make_trace(&[("a", 1), ("b", 2), ("c", 3), ("a", 4), ("b", 5)]);
        let res = sim.simulate(EvictionPolicy::Lru, &trace);
        assert_eq!(res.hits, 2);
        assert_eq!(res.misses, 3);
        let hr = res.hit_rate();
        let mr = res.miss_rate();
        assert!((hr - 0.4).abs() < 1e-9);
        assert!((mr - 0.6).abs() < 1e-9);
        assert!((hr + mr - 1.0).abs() < 1e-9);
    }

    #[test]
    fn lru_evictions_count() {
        let sim = CacheEvictionSimulator::new(2);
        // Each new item after capacity is full causes an eviction.
        let trace = make_trace(&[("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)]);
        let res = sim.simulate(EvictionPolicy::Lru, &trace);
        assert_eq!(res.evictions, 3); // c, d, e each evict one item
    }

    #[test]
    fn lru_capacity_one() {
        let sim = CacheEvictionSimulator::new(1);
        let trace = make_trace(&[("a", 1), ("a", 2), ("b", 3), ("b", 4)]);
        let res = sim.simulate(EvictionPolicy::Lru, &trace);
        // a@1 miss, a@2 hit, b@3 miss+evict-a, b@4 hit
        assert_eq!(res.hits, 2);
        assert_eq!(res.misses, 2);
        assert_eq!(res.evictions, 1);
    }

    // -----------------------------------------------------------------------
    // LFU tests
    // -----------------------------------------------------------------------

    #[test]
    fn lfu_selects_min_freq() {
        let sim = CacheEvictionSimulator::new(2);
        // a@1 miss(freq=1), b@2 miss(freq=1). a@3 hit(freq=2), a@4 hit(freq=3).
        // c@5 miss: evicts lowest-freq item = b(freq=1). cache=[a(3),c(1)].
        // b@6 miss: evicts lowest-freq item = c(freq=1). cache=[a(3),b(1)].
        let trace = make_trace(&[("a", 1), ("b", 2), ("a", 3), ("a", 4), ("c", 5), ("b", 6)]);
        let res = sim.simulate(EvictionPolicy::Lfu, &trace);
        assert_eq!(res.hits, 2); // a@3, a@4
        assert_eq!(res.misses, 4); // a@1, b@2, c@5, b@6
        assert_eq!(res.evictions, 2); // b evicted at c@5; c evicted at b@6
    }

    #[test]
    fn lfu_tie_break_lru_order() {
        // Capacity 2: a and b both freq=1. Load c → must evict one.
        // Tie broken by earliest insertion: a was inserted first → evict a.
        let sim = CacheEvictionSimulator::new(2);
        let trace = make_trace(&[("a", 1), ("b", 2), ("c", 3), ("b", 4)]);
        let res = sim.simulate(EvictionPolicy::Lfu, &trace);
        // a@1 miss, b@2 miss, c@3 miss (evicts a, freq tie), b@4 hit
        assert_eq!(res.hits, 1); // b@4
        assert_eq!(res.evictions, 1);
    }

    #[test]
    fn lfu_hit_rate() {
        let sim = CacheEvictionSimulator::new(3);
        let trace = make_trace(&[("x", 1), ("y", 2), ("z", 3), ("x", 4), ("x", 5), ("y", 6)]);
        let res = sim.simulate(EvictionPolicy::Lfu, &trace);
        assert_eq!(res.hits, 3);
        assert_eq!(res.misses, 3);
        assert!((res.hit_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn lfu_capacity_one() {
        let sim = CacheEvictionSimulator::new(1);
        let trace = make_trace(&[("a", 1), ("a", 2), ("b", 3), ("b", 4)]);
        let res = sim.simulate(EvictionPolicy::Lfu, &trace);
        // a@1 miss; a@2 hit (freq=2); b@3 miss — a has freq 2, b freq 1; b evicted immediately? No:
        // At b@3: cache has [a freq=2]. Evict a (only item). Insert b freq=1.
        // b@4 hit.
        assert_eq!(res.hits, 2); // a@2 and b@4
        assert_eq!(res.evictions, 1);
    }

    // -----------------------------------------------------------------------
    // ARC-approx tests
    // -----------------------------------------------------------------------

    #[test]
    fn arc_t1_to_t2_promotion() {
        // First access → T1. Second access → T2.
        let sim = CacheEvictionSimulator::new(4);
        let trace = make_trace(&[("a", 1), ("b", 2), ("a", 3)]);
        let res = sim.simulate(EvictionPolicy::ArcApprox, &trace);
        // a@1 miss (T1), b@2 miss (T1), a@3 hit (promote to T2)
        assert_eq!(res.hits, 1);
        assert_eq!(res.misses, 2);
        assert_eq!(res.evictions, 0);
    }

    #[test]
    fn arc_evicts_t1_when_over_half() {
        // Capacity 3 → half = 1.
        // Promote one item from T1→T2 so T1.len remains 1 and T2.len=1, then fill T1 to 2 > half.
        // Trace: x@1 miss(T1=[x]), z@2 miss(T1=[x,z]), z@3 hit T1→promote T2(T1=[x],T2=[z]),
        //        y@4 miss total=2<3 no evict(T1=[x,y],T2=[z]),
        //        w@5 miss total=3>=3 → evict: T1.len=2>half=1 → evict x → T1=[y], push w → T1=[y,w].
        let sim = CacheEvictionSimulator::new(3);
        let trace = make_trace(&[("x", 1), ("z", 2), ("z", 3), ("y", 4), ("w", 5)]);
        let res = sim.simulate(EvictionPolicy::ArcApprox, &trace);
        assert_eq!(res.hits, 1); // z@3
        assert_eq!(res.misses, 4); // x@1, z@2, y@4, w@5
        assert_eq!(res.evictions, 1); // x evicted when w@5 is inserted
    }

    #[test]
    fn arc_t2_hit_refreshes() {
        let sim = CacheEvictionSimulator::new(4);
        // a promoted to T2, then hit again in T2.
        let trace = make_trace(&[("a", 1), ("a", 2), ("a", 3)]);
        let res = sim.simulate(EvictionPolicy::ArcApprox, &trace);
        assert_eq!(res.hits, 2); // a@2, a@3
        assert_eq!(res.misses, 1);
        assert_eq!(res.evictions, 0);
    }

    #[test]
    fn arc_capacity_one() {
        let sim = CacheEvictionSimulator::new(1);
        let trace = make_trace(&[("a", 1), ("a", 2), ("b", 3), ("b", 4)]);
        let res = sim.simulate(EvictionPolicy::ArcApprox, &trace);
        // a@1 miss, a@2 hit (T2), b@3 miss+evict, b@4 hit
        assert_eq!(res.hits, 2);
        assert_eq!(res.misses, 2);
        assert_eq!(res.evictions, 1);
    }

    // -----------------------------------------------------------------------
    // compare_policies tests
    // -----------------------------------------------------------------------

    #[test]
    fn compare_policies_returns_three_results() {
        let sim = CacheEvictionSimulator::new(3);
        let trace = make_trace(&[("a", 1), ("b", 2), ("a", 3)]);
        let results = sim.compare_policies(&trace);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn compare_policies_sorted_by_hit_rate_desc() {
        let sim = CacheEvictionSimulator::new(3);
        let trace = make_trace(&[("a", 1), ("b", 2), ("c", 3), ("a", 4), ("b", 5), ("c", 6)]);
        let results = sim.compare_policies(&trace);
        for pair in results.windows(2) {
            assert!(pair[0].hit_rate() >= pair[1].hit_rate());
        }
    }

    #[test]
    fn compare_policies_all_three_policies_present() {
        let sim = CacheEvictionSimulator::new(2);
        let trace = make_trace(&[("a", 1), ("b", 2), ("a", 3)]);
        let results = sim.compare_policies(&trace);
        let policies: Vec<EvictionPolicy> = results.iter().map(|r| r.policy).collect();
        assert!(policies.contains(&EvictionPolicy::Lru));
        assert!(policies.contains(&EvictionPolicy::Lfu));
        assert!(policies.contains(&EvictionPolicy::ArcApprox));
    }

    // -----------------------------------------------------------------------
    // Edge case tests
    // -----------------------------------------------------------------------

    #[test]
    fn empty_trace_all_policies() {
        let sim = CacheEvictionSimulator::new(4);
        for policy in [
            EvictionPolicy::Lru,
            EvictionPolicy::Lfu,
            EvictionPolicy::ArcApprox,
        ] {
            let res = sim.simulate(policy, &[]);
            assert_eq!(res.hits, 0);
            assert_eq!(res.misses, 0);
            assert_eq!(res.evictions, 0);
            assert!((res.hit_rate() - 0.0).abs() < 1e-9);
            assert!((res.miss_rate() - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn hit_rate_miss_rate_sum_to_one() {
        let sim = CacheEvictionSimulator::new(2);
        let trace = make_trace(&[("a", 1), ("b", 2), ("c", 3), ("a", 4)]);
        for policy in [
            EvictionPolicy::Lru,
            EvictionPolicy::Lfu,
            EvictionPolicy::ArcApprox,
        ] {
            let res = sim.simulate(policy, &trace);
            assert!((res.hit_rate() + res.miss_rate() - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn eviction_policy_derive_traits() {
        let p = EvictionPolicy::Lru;
        let q = p; // Copy
        let r = p; // Clone
        assert_eq!(p, q);
        assert_eq!(p, r);
        let _ = format!("{p:?}"); // Debug
    }

    #[test]
    fn large_trace_lru_no_panic() {
        let sim = CacheEvictionSimulator::new(10);
        let trace: Vec<AccessEvent> = (0u64..200)
            .map(|i| AccessEvent {
                cid: format!("item-{}", i % 15),
                timestamp_tick: i,
            })
            .collect();
        let res = sim.simulate(EvictionPolicy::Lru, &trace);
        assert!(res.hits + res.misses == 200);
    }

    #[test]
    fn lru_no_eviction_under_capacity() {
        let sim = CacheEvictionSimulator::new(10);
        let trace = make_trace(&[("a", 1), ("b", 2), ("c", 3), ("a", 4), ("b", 5), ("c", 6)]);
        let res = sim.simulate(EvictionPolicy::Lru, &trace);
        assert_eq!(res.evictions, 0);
        assert_eq!(res.hits, 3);
    }
}
