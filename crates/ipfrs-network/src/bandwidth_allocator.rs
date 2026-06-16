//! Peer bandwidth allocator using weighted max-min fairness.
//!
//! Allocates outbound bandwidth fairly across peers with per-peer minimum
//! guarantees and hard caps. Supports three allocation strategies:
//! `EqualShare`, `WeightedFair`, and `MaxMinFair`.

use std::collections::HashMap;

// ── Strategy ─────────────────────────────────────────────────────────────────

/// Strategy used by [`PeerBandwidthAllocator`] when distributing bandwidth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AllocationStrategy {
    /// Each peer receives `total_bps / peer_count`, clamped to `[min_bps, max_bps]`.
    EqualShare,
    /// Each peer receives `(weight / Σweights) * total_bps`, clamped to `[min_bps, max_bps]`.
    WeightedFair,
    /// Iterative max-min: guarantee `min_bps` first, then distribute the remainder equally
    /// until every peer hits its `max_bps` cap.
    MaxMinFair,
}

// ── PeerAllocation ────────────────────────────────────────────────────────────

/// Per-peer allocation record.
#[derive(Clone, Debug)]
pub struct PeerAllocation {
    /// Unique peer identifier.
    pub peer_id: String,
    /// Relative weight used by `WeightedFair` (default `1.0`).
    pub weight: f64,
    /// Guaranteed minimum bandwidth in bps.
    pub min_bps: u64,
    /// Hard cap in bps (`u64::MAX` = unlimited).
    pub max_bps: u64,
    /// Result of the last [`PeerBandwidthAllocator::run_allocation`] call.
    pub allocated_bps: u64,
}

impl PeerAllocation {
    fn new(peer_id: String, weight: f64, min_bps: u64, max_bps: u64) -> Self {
        Self {
            peer_id,
            weight,
            min_bps,
            max_bps,
            allocated_bps: 0,
        }
    }
}

// ── AllocationStats ───────────────────────────────────────────────────────────

/// Aggregate statistics produced by [`PeerBandwidthAllocator::stats`].
#[derive(Clone, Debug)]
pub struct AllocationStats {
    /// Configured total bandwidth budget in bps.
    pub total_bps: u64,
    /// Sum of all per-peer `allocated_bps` after the last allocation run.
    pub allocated_bps: u64,
    /// `total_bps - allocated_bps`.
    pub unallocated_bps: u64,
    /// Number of peers currently tracked.
    pub peer_count: usize,
}

impl AllocationStats {
    /// Returns the fraction of the total bandwidth that has been allocated
    /// (`allocated_bps / total_bps`), or `0.0` when `total_bps == 0`.
    pub fn utilization(&self) -> f64 {
        if self.total_bps == 0 {
            return 0.0;
        }
        self.allocated_bps as f64 / self.total_bps as f64
    }
}

// ── PeerBandwidthAllocator ────────────────────────────────────────────────────

/// Allocates outbound bandwidth fairly across peers.
pub struct PeerBandwidthAllocator {
    /// All tracked peers, keyed by peer_id.
    pub peers: HashMap<String, PeerAllocation>,
    /// Total available bandwidth budget in bps.
    pub total_bps: u64,
    /// Active allocation strategy.
    pub strategy: AllocationStrategy,
}

impl PeerBandwidthAllocator {
    /// Creates a new allocator with the given budget and strategy.
    pub fn new(total_bps: u64, strategy: AllocationStrategy) -> Self {
        Self {
            peers: HashMap::new(),
            total_bps,
            strategy,
        }
    }

    /// Registers a new peer.  If the peer already exists it is updated in place.
    pub fn add_peer(&mut self, peer_id: String, weight: f64, min_bps: u64, max_bps: u64) {
        self.peers
            .entry(peer_id.clone())
            .and_modify(|p| {
                p.weight = weight;
                p.min_bps = min_bps;
                p.max_bps = max_bps;
            })
            .or_insert_with(|| PeerAllocation::new(peer_id, weight, min_bps, max_bps));
    }

    /// Removes a peer and returns `true` if it existed.
    pub fn remove_peer(&mut self, peer_id: &str) -> bool {
        self.peers.remove(peer_id).is_some()
    }

    /// Runs the configured allocation strategy and stores results in each peer's
    /// `allocated_bps` field.
    pub fn run_allocation(&mut self) {
        let count = self.peers.len();
        if count == 0 {
            return;
        }

        match self.strategy {
            AllocationStrategy::EqualShare => self.run_equal_share(count),
            AllocationStrategy::WeightedFair => self.run_weighted_fair(),
            AllocationStrategy::MaxMinFair => self.run_max_min_fair(count),
        }
    }

    // ── EqualShare ──────────────────────────────────────────────────────────

    fn run_equal_share(&mut self, count: usize) {
        let share = self.total_bps / count as u64;
        for peer in self.peers.values_mut() {
            peer.allocated_bps = share.clamp(peer.min_bps, peer.max_bps);
        }
    }

    // ── WeightedFair ────────────────────────────────────────────────────────

    fn run_weighted_fair(&mut self) {
        let sum_weights: f64 = self.peers.values().map(|p| p.weight.max(0.0)).sum();

        if sum_weights <= 0.0 {
            // All weights are zero — fall back to zero allocation (honour min_bps).
            for peer in self.peers.values_mut() {
                peer.allocated_bps = peer.min_bps.min(peer.max_bps);
            }
            return;
        }

        let total = self.total_bps as f64;
        for peer in self.peers.values_mut() {
            let w = peer.weight.max(0.0);
            let proportional = ((w / sum_weights) * total) as u64;
            peer.allocated_bps = proportional.clamp(peer.min_bps, peer.max_bps);
        }
    }

    // ── MaxMinFair ──────────────────────────────────────────────────────────

    fn run_max_min_fair(&mut self, count: usize) {
        // --- Step 1: satisfy minimum guarantees ---
        // Sum of all min_bps (capped by max_bps so min can never exceed max).
        let total_min: u64 = self.peers.values().map(|p| p.min_bps.min(p.max_bps)).sum();

        // If the total budget cannot even cover the minimums we distribute what
        // we have proportionally to the minimums (best-effort).
        if total_min >= self.total_bps {
            let total_bps = self.total_bps;
            let total_min_f = total_min as f64;
            for peer in self.peers.values_mut() {
                let effective_min = peer.min_bps.min(peer.max_bps) as f64;
                let share = if total_min_f > 0.0 {
                    ((effective_min / total_min_f) * total_bps as f64) as u64
                } else {
                    0
                };
                peer.allocated_bps = share.min(peer.max_bps);
            }
            return;
        }

        // Assign minimums.
        for peer in self.peers.values_mut() {
            peer.allocated_bps = peer.min_bps.min(peer.max_bps);
        }

        // --- Step 2: iterative max-min distribution of remainder ---
        // Remaining budget after satisfying minimums.
        let mut remaining = self.total_bps.saturating_sub(total_min);
        // Peers that have not yet hit their cap.
        let mut uncapped_count = count;

        loop {
            if remaining == 0 || uncapped_count == 0 {
                break;
            }

            let fair_share = remaining / uncapped_count as u64;
            if fair_share == 0 {
                break;
            }

            let mut newly_capped = 0u64; // bandwidth freed back by peers hitting their cap
            let mut still_uncapped = 0usize;

            for peer in self.peers.values_mut() {
                if peer.allocated_bps >= peer.max_bps {
                    // Already capped in a previous iteration.
                    continue;
                }
                let candidate = peer.allocated_bps + fair_share;
                if candidate >= peer.max_bps {
                    newly_capped += candidate - peer.max_bps;
                    peer.allocated_bps = peer.max_bps;
                } else {
                    peer.allocated_bps = candidate;
                    still_uncapped += 1;
                }
            }

            remaining = newly_capped;
            uncapped_count = still_uncapped;
        }

        // If there is leftover and some peers are still under their cap,
        // give whatever is left to the first uncapped peer (avoids wasting budget
        // due to integer truncation).
        if remaining > 0 {
            if let Some(peer) = self
                .peers
                .values_mut()
                .find(|p| p.allocated_bps < p.max_bps)
            {
                let extra = remaining.min(peer.max_bps - peer.allocated_bps);
                peer.allocated_bps += extra;
            }
        }
    }

    // ── Query helpers ────────────────────────────────────────────────────────

    /// Returns the last allocated bandwidth for `peer_id`, or `None` if unknown.
    pub fn allocation_for(&self, peer_id: &str) -> Option<u64> {
        self.peers.get(peer_id).map(|p| p.allocated_bps)
    }

    /// Returns aggregate statistics for the current state.
    pub fn stats(&self) -> AllocationStats {
        let allocated_bps: u64 = self.peers.values().map(|p| p.allocated_bps).sum();
        let unallocated_bps = self.total_bps.saturating_sub(allocated_bps);
        AllocationStats {
            total_bps: self.total_bps,
            allocated_bps,
            unallocated_bps,
            peer_count: self.peers.len(),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // helper: build an allocator with N equal peers (weight=1, min=0, max=unlimited)
    fn alloc_equal_peers(n: usize, total_bps: u64) -> PeerBandwidthAllocator {
        let mut alloc = PeerBandwidthAllocator::new(total_bps, AllocationStrategy::EqualShare);
        for i in 0..n {
            alloc.add_peer(format!("peer{i}"), 1.0, 0, u64::MAX);
        }
        alloc
    }

    // ── EqualShare ────────────────────────────────────────────────────────────

    /// T01 – equal share distributes total_bps / peer_count to every peer.
    #[test]
    fn t01_equal_share_distributes_evenly() {
        let mut alloc = alloc_equal_peers(4, 1000);
        alloc.run_allocation();
        for peer in alloc.peers.values() {
            assert_eq!(
                peer.allocated_bps, 250,
                "peer {} got unexpected share",
                peer.peer_id
            );
        }
    }

    /// T02 – EqualShare respects max_bps cap.
    #[test]
    fn t02_equal_share_respects_max_cap() {
        let mut alloc = PeerBandwidthAllocator::new(1000, AllocationStrategy::EqualShare);
        alloc.add_peer("p0".into(), 1.0, 0, 100); // cap at 100; share would be 500
        alloc.add_peer("p1".into(), 1.0, 0, u64::MAX);
        alloc.run_allocation();
        assert_eq!(alloc.allocation_for("p0"), Some(100));
        assert_eq!(alloc.allocation_for("p1"), Some(500));
    }

    /// T03 – EqualShare respects min_bps guarantee.
    #[test]
    fn t03_equal_share_respects_min_guarantee() {
        let mut alloc = PeerBandwidthAllocator::new(1000, AllocationStrategy::EqualShare);
        alloc.add_peer("p0".into(), 1.0, 600, u64::MAX); // share=500 < min=600
        alloc.add_peer("p1".into(), 1.0, 0, u64::MAX);
        alloc.run_allocation();
        assert_eq!(alloc.allocation_for("p0"), Some(600));
        assert_eq!(alloc.allocation_for("p1"), Some(500));
    }

    // ── WeightedFair ─────────────────────────────────────────────────────────

    /// T04 – WeightedFair is proportional to weights.
    #[test]
    fn t04_weighted_fair_proportional() {
        let mut alloc = PeerBandwidthAllocator::new(1200, AllocationStrategy::WeightedFair);
        alloc.add_peer("p0".into(), 1.0, 0, u64::MAX);
        alloc.add_peer("p1".into(), 2.0, 0, u64::MAX);
        alloc.add_peer("p2".into(), 3.0, 0, u64::MAX);
        alloc.run_allocation();
        // weights 1:2:3 over 1200 bps → 200, 400, 600
        assert_eq!(alloc.allocation_for("p0"), Some(200));
        assert_eq!(alloc.allocation_for("p1"), Some(400));
        assert_eq!(alloc.allocation_for("p2"), Some(600));
    }

    /// T05 – WeightedFair respects max_bps cap.
    #[test]
    fn t05_weighted_fair_respects_max_cap() {
        let mut alloc = PeerBandwidthAllocator::new(1000, AllocationStrategy::WeightedFair);
        alloc.add_peer("heavy".into(), 9.0, 0, 200); // would get 900 but capped at 200
        alloc.add_peer("light".into(), 1.0, 0, u64::MAX);
        alloc.run_allocation();
        assert_eq!(alloc.allocation_for("heavy"), Some(200));
        assert_eq!(alloc.allocation_for("light"), Some(100));
    }

    /// T06 – A peer with weight=0 receives min_bps (no proportional share).
    #[test]
    fn t06_weight_zero_peer_gets_min() {
        let mut alloc = PeerBandwidthAllocator::new(1000, AllocationStrategy::WeightedFair);
        alloc.add_peer("zero".into(), 0.0, 50, u64::MAX);
        alloc.add_peer("normal".into(), 1.0, 0, u64::MAX);
        alloc.run_allocation();
        // weight=0 peer: proportional share = 0, clamped up to min_bps=50
        assert_eq!(alloc.allocation_for("zero"), Some(50));
        // weight=1 peer: gets full 1000
        assert_eq!(alloc.allocation_for("normal"), Some(1000));
    }

    // ── MaxMinFair ────────────────────────────────────────────────────────────

    /// T07 – MaxMinFair satisfies all min_bps guarantees.
    #[test]
    fn t07_max_min_fair_satisfies_minimums() {
        let mut alloc = PeerBandwidthAllocator::new(1000, AllocationStrategy::MaxMinFair);
        alloc.add_peer("p0".into(), 1.0, 100, u64::MAX);
        alloc.add_peer("p1".into(), 1.0, 200, u64::MAX);
        alloc.add_peer("p2".into(), 1.0, 50, u64::MAX);
        alloc.run_allocation();
        // After mins (100+200+50=350) remaining=650, equal share of 650/3≈216 each
        assert!(
            alloc
                .allocation_for("p0")
                .expect("test: peer should be registered and have an allocation")
                >= 100
        );
        assert!(
            alloc
                .allocation_for("p1")
                .expect("test: peer should be registered and have an allocation")
                >= 200
        );
        assert!(
            alloc
                .allocation_for("p2")
                .expect("test: peer should be registered and have an allocation")
                >= 50
        );
    }

    /// T08 – MaxMinFair does not exceed max_bps.
    #[test]
    fn t08_max_min_fair_respects_max_cap() {
        let mut alloc = PeerBandwidthAllocator::new(1000, AllocationStrategy::MaxMinFair);
        alloc.add_peer("capped".into(), 1.0, 0, 100);
        alloc.add_peer("free".into(), 1.0, 0, u64::MAX);
        alloc.run_allocation();
        assert!(
            alloc
                .allocation_for("capped")
                .expect("test: peer should be registered and have an allocation")
                <= 100
        );
    }

    /// T09 – MaxMinFair distributes remainder after minimums equally.
    #[test]
    fn t09_max_min_fair_equal_remainder() {
        // 3 peers, min 0 each, no caps → each should get total/3
        let mut alloc = PeerBandwidthAllocator::new(900, AllocationStrategy::MaxMinFair);
        alloc.add_peer("a".into(), 1.0, 0, u64::MAX);
        alloc.add_peer("b".into(), 1.0, 0, u64::MAX);
        alloc.add_peer("c".into(), 1.0, 0, u64::MAX);
        alloc.run_allocation();
        let a = alloc
            .allocation_for("a")
            .expect("test: peer should be registered and have an allocation");
        let b = alloc
            .allocation_for("b")
            .expect("test: peer should be registered and have an allocation");
        let c = alloc
            .allocation_for("c")
            .expect("test: peer should be registered and have an allocation");
        assert_eq!(a, 300);
        assert_eq!(b, 300);
        assert_eq!(c, 300);
    }

    /// T10 – MaxMinFair: when budget is less than sum of minimums, distributes proportionally.
    #[test]
    fn t10_max_min_fair_budget_below_minimums() {
        let mut alloc = PeerBandwidthAllocator::new(100, AllocationStrategy::MaxMinFair);
        alloc.add_peer("x".into(), 1.0, 200, u64::MAX); // min > total
        alloc.add_peer("y".into(), 1.0, 200, u64::MAX);
        alloc.run_allocation();
        let x = alloc
            .allocation_for("x")
            .expect("test: allocation_for x should return Some after run_allocation");
        let y = alloc
            .allocation_for("y")
            .expect("test: allocation_for y should return Some after run_allocation");
        // Each should get approximately 50 (total/2); neither exceeds total.
        assert!(x + y <= 100, "allocated more than total: x={x} y={y}");
    }

    // ── remove_peer ───────────────────────────────────────────────────────────

    /// T11 – remove_peer returns true for existing peer and false for unknown.
    #[test]
    fn t11_remove_peer_returns_correct_bool() {
        let mut alloc = alloc_equal_peers(2, 1000);
        assert!(alloc.remove_peer("peer0"));
        assert!(!alloc.remove_peer("peer0")); // already removed
        assert!(!alloc.remove_peer("nonexistent"));
    }

    /// T12 – After removing a peer, allocation re-runs without it.
    #[test]
    fn t12_remove_then_rerun() {
        let mut alloc = alloc_equal_peers(3, 900);
        alloc.run_allocation();
        assert_eq!(alloc.allocation_for("peer0"), Some(300));
        alloc.remove_peer("peer2");
        alloc.run_allocation();
        // Now 2 peers share 900
        assert_eq!(alloc.allocation_for("peer0"), Some(450));
        assert_eq!(alloc.allocation_for("peer1"), Some(450));
        assert_eq!(alloc.allocation_for("peer2"), None);
    }

    // ── allocation_for ────────────────────────────────────────────────────────

    /// T13 – allocation_for returns Some for known peer and None for unknown.
    #[test]
    fn t13_allocation_for_some_and_none() {
        let mut alloc = alloc_equal_peers(1, 1000);
        alloc.run_allocation();
        assert!(alloc.allocation_for("peer0").is_some());
        assert!(alloc.allocation_for("ghost").is_none());
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    /// T14 – stats.utilization is correct after allocation.
    #[test]
    fn t14_stats_utilization() {
        let mut alloc = alloc_equal_peers(2, 1000);
        alloc.run_allocation();
        let stats = alloc.stats();
        // 2 peers each get 500 → 1000 allocated → utilization = 1.0
        assert!((stats.utilization() - 1.0).abs() < f64::EPSILON);
    }

    /// T15 – stats fields are consistent.
    #[test]
    fn t15_stats_fields_consistent() {
        let mut alloc = alloc_equal_peers(3, 900);
        alloc.run_allocation();
        let stats = alloc.stats();
        assert_eq!(stats.total_bps, 900);
        assert_eq!(stats.allocated_bps, 900);
        assert_eq!(stats.unallocated_bps, 0);
        assert_eq!(stats.peer_count, 3);
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    /// T16 – Empty peer list: run_allocation is a no-op; stats returns zeros.
    #[test]
    fn t16_empty_peers_no_panic() {
        let mut alloc = PeerBandwidthAllocator::new(1000, AllocationStrategy::MaxMinFair);
        alloc.run_allocation(); // must not panic
        let stats = alloc.stats();
        assert_eq!(stats.peer_count, 0);
        assert_eq!(stats.allocated_bps, 0);
        assert_eq!(stats.utilization(), 0.0);
    }

    /// T17 – Single peer gets the full budget (up to its max_bps).
    #[test]
    fn t17_single_peer_gets_full_budget() {
        let mut alloc = PeerBandwidthAllocator::new(5000, AllocationStrategy::WeightedFair);
        alloc.add_peer("only".into(), 1.0, 0, u64::MAX);
        alloc.run_allocation();
        assert_eq!(alloc.allocation_for("only"), Some(5000));
    }

    /// T18 – Adding a peer and re-running changes allocations.
    #[test]
    fn t18_add_peer_then_rerun() {
        let mut alloc = PeerBandwidthAllocator::new(600, AllocationStrategy::EqualShare);
        alloc.add_peer("a".into(), 1.0, 0, u64::MAX);
        alloc.run_allocation();
        assert_eq!(alloc.allocation_for("a"), Some(600));

        alloc.add_peer("b".into(), 1.0, 0, u64::MAX);
        alloc.run_allocation();
        assert_eq!(alloc.allocation_for("a"), Some(300));
        assert_eq!(alloc.allocation_for("b"), Some(300));
    }

    /// T19 – stats.utilization is 0.0 when total_bps is 0.
    #[test]
    fn t19_utilization_zero_total() {
        let alloc = PeerBandwidthAllocator::new(0, AllocationStrategy::EqualShare);
        assert_eq!(alloc.stats().utilization(), 0.0);
    }

    /// T20 – MaxMinFair: capped peer's surplus is redistributed to free peers.
    #[test]
    fn t20_max_min_fair_surplus_redistributed() {
        // total=1000, peer0 capped at 100, peer1 free
        // After equal step: 500 each → peer0 capped at 100 → 400 surplus → peer1 gets 900
        let mut alloc = PeerBandwidthAllocator::new(1000, AllocationStrategy::MaxMinFair);
        alloc.add_peer("capped".into(), 1.0, 0, 100);
        alloc.add_peer("free".into(), 1.0, 0, u64::MAX);
        alloc.run_allocation();
        assert_eq!(alloc.allocation_for("capped"), Some(100));
        assert_eq!(alloc.allocation_for("free"), Some(900));
    }
}
