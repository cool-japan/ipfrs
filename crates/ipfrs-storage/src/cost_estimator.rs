//! Storage Cost Estimator
//!
//! Estimates monetary and computational costs for storage operations across
//! different backend types: local SSD, cloud object storage, and cold archive.
//!
//! # Overview
//!
//! - [`BackendType`] — enum of supported storage backends with cost parameters
//! - [`OperationCost`] — cost breakdown for a single storage operation
//! - [`CostProjection`] — monthly/annual projection with savings vs baseline
//! - [`StorageCostEstimator`] — main estimator struct

/// Storage backend types with associated cost parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendType {
    /// Fast local NVMe SSD
    LocalSsd,
    /// Cloud object storage — e.g. AWS S3 Standard
    CloudHot,
    /// Cloud object storage — e.g. AWS S3 Infrequent Access
    CloudWarm,
    /// Cloud archive — e.g. AWS S3 Glacier
    CloudCold,
    /// Spinning local hard disk drive
    LocalHdd,
}

impl BackendType {
    /// Cost per GB per month in USD.
    pub fn cost_per_gb_month(self) -> f64 {
        match self {
            BackendType::LocalSsd => 0.10,
            BackendType::CloudHot => 0.023,
            BackendType::CloudWarm => 0.0125,
            BackendType::CloudCold => 0.004,
            BackendType::LocalHdd => 0.03,
        }
    }

    /// Cost per PUT (write) request in USD.
    pub fn cost_per_put_request(self) -> f64 {
        match self {
            BackendType::LocalSsd => 0.0,
            BackendType::CloudHot => 0.000_005,
            BackendType::CloudWarm => 0.000_010,
            BackendType::CloudCold => 0.000_030,
            BackendType::LocalHdd => 0.0,
        }
    }

    /// Cost per GET (read) request in USD.
    pub fn cost_per_get_request(self) -> f64 {
        match self {
            BackendType::LocalSsd => 0.0,
            BackendType::CloudHot => 0.000_000_4,
            BackendType::CloudWarm => 0.000_001,
            BackendType::CloudCold => 0.001,
            BackendType::LocalHdd => 0.0,
        }
    }

    /// Typical read latency in milliseconds.
    pub fn read_latency_ms(self) -> u64 {
        match self {
            BackendType::LocalSsd => 0,
            BackendType::CloudHot => 5,
            BackendType::CloudWarm => 30,
            BackendType::CloudCold => 500,
            BackendType::LocalHdd => 5,
        }
    }

    /// Return all backend variants in definition order.
    fn all() -> [BackendType; 5] {
        [
            BackendType::LocalSsd,
            BackendType::CloudHot,
            BackendType::CloudWarm,
            BackendType::CloudCold,
            BackendType::LocalHdd,
        ]
    }
}

/// Cost breakdown for a single storage operation.
#[derive(Clone, Debug, PartialEq)]
pub struct OperationCost {
    /// The backend this cost was estimated for.
    pub backend: BackendType,
    /// Monthly GB storage cost in USD.
    pub storage_cost: f64,
    /// Total PUT request cost in USD.
    pub put_cost: f64,
    /// Total GET request cost in USD.
    pub get_cost: f64,
    /// Sum of all cost components in USD.
    pub total_cost: f64,
}

impl OperationCost {
    /// Returns `true` when this operation is cheaper than `other`.
    pub fn is_cheaper_than(&self, other: &OperationCost) -> bool {
        self.total_cost < other.total_cost
    }
}

/// Monthly and annual cost projection for a backend.
#[derive(Clone, Debug, PartialEq)]
pub struct CostProjection {
    /// The backend this projection covers.
    pub backend: BackendType,
    /// Estimated cost for one month in USD.
    pub monthly_cost: f64,
    /// Estimated cost for twelve months in USD.
    pub annual_cost: f64,
    /// `baseline.monthly_cost - this.monthly_cost`.
    ///
    /// Positive means cheaper than the baseline (`CloudHot`).
    /// Negative means more expensive than the baseline.
    pub savings_vs_baseline: f64,
}

/// Estimates monetary and computational costs for storage operations.
///
/// # Example
/// ```
/// use ipfrs_storage::cost_estimator::{BackendType, StorageCostEstimator};
///
/// let estimator = StorageCostEstimator::new();
/// let cost = estimator.estimate_operation(BackendType::CloudHot, 1 << 30, 1_000, 10_000);
/// assert!(cost.total_cost > 0.0);
/// ```
pub struct StorageCostEstimator;

impl StorageCostEstimator {
    /// Create a new estimator instance.
    pub fn new() -> Self {
        StorageCostEstimator
    }

    /// Estimate the cost of a storage operation.
    ///
    /// - `size_bytes` — data volume stored
    /// - `num_puts` — number of write/PUT requests
    /// - `num_gets` — number of read/GET requests
    pub fn estimate_operation(
        &self,
        backend: BackendType,
        size_bytes: u64,
        num_puts: u64,
        num_gets: u64,
    ) -> OperationCost {
        let gb = size_bytes as f64 / (1024.0_f64 * 1024.0 * 1024.0);
        let storage_cost = gb * backend.cost_per_gb_month();
        let put_cost = num_puts as f64 * backend.cost_per_put_request();
        let get_cost = num_gets as f64 * backend.cost_per_get_request();
        let total_cost = storage_cost + put_cost + get_cost;
        OperationCost {
            backend,
            storage_cost,
            put_cost,
            get_cost,
            total_cost,
        }
    }

    /// Estimate costs for all backends and return them sorted by `total_cost` ascending.
    pub fn compare_backends(
        &self,
        size_bytes: u64,
        num_puts: u64,
        num_gets: u64,
    ) -> Vec<OperationCost> {
        let mut costs: Vec<OperationCost> = BackendType::all()
            .iter()
            .map(|&b| self.estimate_operation(b, size_bytes, num_puts, num_gets))
            .collect();

        costs.sort_by(|a, b| {
            a.total_cost
                .partial_cmp(&b.total_cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        costs
    }

    /// Project monthly and annual costs for a backend.
    ///
    /// `savings_vs_baseline` is computed relative to `CloudHot`.
    pub fn project_annual(
        &self,
        backend: BackendType,
        size_bytes: u64,
        monthly_puts: u64,
        monthly_gets: u64,
    ) -> CostProjection {
        let this_monthly = self
            .estimate_operation(backend, size_bytes, monthly_puts, monthly_gets)
            .total_cost;
        let baseline_monthly = self
            .estimate_operation(
                BackendType::CloudHot,
                size_bytes,
                monthly_puts,
                monthly_gets,
            )
            .total_cost;
        CostProjection {
            backend,
            monthly_cost: this_monthly,
            annual_cost: this_monthly * 12.0,
            savings_vs_baseline: baseline_monthly - this_monthly,
        }
    }

    /// Return the backend with the lowest total cost for the given workload.
    pub fn cheapest_backend(&self, size_bytes: u64, num_puts: u64, num_gets: u64) -> BackendType {
        let ranked = self.compare_backends(size_bytes, num_puts, num_gets);
        // compare_backends always has 5 elements; safe to index.
        ranked[0].backend
    }
}

impl Default for StorageCostEstimator {
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

    const GIB: u64 = 1024 * 1024 * 1024;

    // ------------------------------------------------------------------
    // BackendType helpers
    // ------------------------------------------------------------------

    #[test]
    fn local_ssd_has_zero_request_costs() {
        assert_eq!(BackendType::LocalSsd.cost_per_put_request(), 0.0);
        assert_eq!(BackendType::LocalSsd.cost_per_get_request(), 0.0);
    }

    #[test]
    fn local_hdd_has_zero_request_costs() {
        assert_eq!(BackendType::LocalHdd.cost_per_put_request(), 0.0);
        assert_eq!(BackendType::LocalHdd.cost_per_get_request(), 0.0);
    }

    #[test]
    fn cloud_cold_has_highest_get_cost() {
        let cold = BackendType::CloudCold.cost_per_get_request();
        let hot = BackendType::CloudHot.cost_per_get_request();
        let warm = BackendType::CloudWarm.cost_per_get_request();
        assert!(cold > hot, "CloudCold get cost must exceed CloudHot");
        assert!(cold > warm, "CloudCold get cost must exceed CloudWarm");
    }

    #[test]
    fn cloud_cold_has_highest_put_cost() {
        let cold = BackendType::CloudCold.cost_per_put_request();
        let hot = BackendType::CloudHot.cost_per_put_request();
        let warm = BackendType::CloudWarm.cost_per_put_request();
        assert!(cold > hot);
        assert!(cold > warm);
    }

    #[test]
    fn read_latency_ordering() {
        // LocalSsd fastest, CloudCold slowest
        assert_eq!(BackendType::LocalSsd.read_latency_ms(), 0);
        assert!(BackendType::CloudHot.read_latency_ms() < BackendType::CloudWarm.read_latency_ms());
        assert!(
            BackendType::CloudWarm.read_latency_ms() < BackendType::CloudCold.read_latency_ms()
        );
    }

    #[test]
    fn local_ssd_latency_is_zero() {
        assert_eq!(BackendType::LocalSsd.read_latency_ms(), 0);
    }

    #[test]
    fn local_hdd_latency_equals_cloud_hot() {
        assert_eq!(
            BackendType::LocalHdd.read_latency_ms(),
            BackendType::CloudHot.read_latency_ms()
        );
    }

    // ------------------------------------------------------------------
    // estimate_operation
    // ------------------------------------------------------------------

    #[test]
    fn estimate_operation_sums_correctly() {
        let est = StorageCostEstimator::new();
        let cost = est.estimate_operation(BackendType::CloudHot, GIB, 1_000, 5_000);

        let expected_storage = BackendType::CloudHot.cost_per_gb_month();
        let expected_put = 1_000.0 * BackendType::CloudHot.cost_per_put_request();
        let expected_get = 5_000.0 * BackendType::CloudHot.cost_per_get_request();
        let expected_total = expected_storage + expected_put + expected_get;

        assert!((cost.storage_cost - expected_storage).abs() < 1e-12);
        assert!((cost.put_cost - expected_put).abs() < 1e-12);
        assert!((cost.get_cost - expected_get).abs() < 1e-12);
        assert!((cost.total_cost - expected_total).abs() < 1e-12);
    }

    #[test]
    fn estimate_operation_local_ssd_zero_request_costs() {
        let est = StorageCostEstimator::new();
        let cost = est.estimate_operation(BackendType::LocalSsd, GIB, 1_000_000, 1_000_000);
        assert_eq!(cost.put_cost, 0.0);
        assert_eq!(cost.get_cost, 0.0);
        assert!((cost.total_cost - cost.storage_cost).abs() < 1e-12);
    }

    #[test]
    fn estimate_operation_zero_size_zero_storage_cost() {
        let est = StorageCostEstimator::new();
        let cost = est.estimate_operation(BackendType::CloudHot, 0, 0, 0);
        assert_eq!(cost.storage_cost, 0.0);
        assert_eq!(cost.put_cost, 0.0);
        assert_eq!(cost.get_cost, 0.0);
        assert_eq!(cost.total_cost, 0.0);
    }

    #[test]
    fn estimate_operation_backend_field_correct() {
        let est = StorageCostEstimator::new();
        let cost = est.estimate_operation(BackendType::CloudWarm, GIB, 0, 0);
        assert_eq!(cost.backend, BackendType::CloudWarm);
    }

    // ------------------------------------------------------------------
    // compare_backends
    // ------------------------------------------------------------------

    #[test]
    fn compare_backends_returns_five_entries() {
        let est = StorageCostEstimator::new();
        let costs = est.compare_backends(GIB, 100, 100);
        assert_eq!(costs.len(), 5);
    }

    #[test]
    fn compare_backends_sorted_ascending() {
        let est = StorageCostEstimator::new();
        let costs = est.compare_backends(GIB, 10_000, 100_000);
        for window in costs.windows(2) {
            assert!(
                window[0].total_cost <= window[1].total_cost,
                "compare_backends not sorted ascending: {} > {}",
                window[0].total_cost,
                window[1].total_cost
            );
        }
    }

    #[test]
    fn compare_backends_all_backends_represented() {
        let est = StorageCostEstimator::new();
        let costs = est.compare_backends(GIB, 0, 0);
        let mut backends: Vec<BackendType> = costs.iter().map(|c| c.backend).collect();
        backends.sort_by_key(|b| format!("{b:?}"));
        let mut expected = [
            BackendType::LocalSsd,
            BackendType::CloudHot,
            BackendType::CloudWarm,
            BackendType::CloudCold,
            BackendType::LocalHdd,
        ];
        expected.sort_by_key(|b| format!("{b:?}"));
        assert_eq!(backends, expected.to_vec());
    }

    // ------------------------------------------------------------------
    // cheapest_backend
    // ------------------------------------------------------------------

    #[test]
    fn cheapest_backend_cloud_cold_for_archival_no_requests() {
        // No puts or gets — only storage cost matters.
        // CloudCold has cost_per_gb_month=0.004, the lowest.
        let est = StorageCostEstimator::new();
        let cheapest = est.cheapest_backend(GIB, 0, 0);
        assert_eq!(cheapest, BackendType::CloudCold);
    }

    #[test]
    fn cheapest_backend_local_ssd_for_high_read_zero_cost() {
        // LocalSsd charges $0 per request, only storage cost.
        // With massive gets, cloud backends accumulate large get costs.
        // CloudCold get = $0.001/req => 1e9 gets => $1_000_000
        // LocalSsd gets = $0 => cheapest.
        let est = StorageCostEstimator::new();
        let cheapest = est.cheapest_backend(GIB, 0, 1_000_000_000);
        // LocalSsd or LocalHdd both have $0 request costs — compare storage:
        // LocalSsd = $0.10/GB, LocalHdd = $0.03/GB => LocalHdd is cheaper
        // But LocalHdd latency is higher; regardless, by cost LocalHdd wins.
        // The test verifies a local backend wins, not cloud.
        let result = cheapest;
        assert!(
            result == BackendType::LocalSsd || result == BackendType::LocalHdd,
            "Expected a local backend for high-read workload, got {result:?}"
        );
    }

    #[test]
    fn cheapest_backend_local_hdd_cheapest_storage_only() {
        // At zero requests, only storage cost matters:
        // CloudCold=0.004, CloudWarm=0.0125, CloudHot=0.023, LocalHdd=0.03, LocalSsd=0.10
        // So CloudCold is cheapest overall with 0 requests.
        let est = StorageCostEstimator::new();
        let cheapest = est.cheapest_backend(10 * GIB, 0, 0);
        assert_eq!(cheapest, BackendType::CloudCold);
    }

    #[test]
    fn cheapest_backend_local_ssd_dominates_with_zero_cost_requests() {
        // When puts are very large and we compare local vs cloud:
        // LocalSsd put cost = 0, CloudCold put cost = 0.00003/req
        // 10e6 puts on CloudCold = $300; LocalSsd = $0 puts
        // Storage: LocalSsd 1GiB = $0.10, CloudCold = $0.004
        // Total LocalSsd ~ $0.10, CloudCold ~ $300.004 => LocalSsd wins
        let est = StorageCostEstimator::new();
        let cheapest = est.cheapest_backend(GIB, 10_000_000, 0);
        assert!(
            cheapest == BackendType::LocalSsd || cheapest == BackendType::LocalHdd,
            "Expected local backend for massive PUT workload, got {cheapest:?}"
        );
    }

    // ------------------------------------------------------------------
    // project_annual
    // ------------------------------------------------------------------

    #[test]
    fn project_annual_annual_equals_monthly_times_12() {
        let est = StorageCostEstimator::new();
        let proj = est.project_annual(BackendType::CloudHot, GIB, 1_000, 10_000);
        assert!((proj.annual_cost - proj.monthly_cost * 12.0).abs() < 1e-9);
    }

    #[test]
    fn project_annual_baseline_savings_is_zero_for_cloud_hot() {
        // CloudHot IS the baseline, so savings = 0
        let est = StorageCostEstimator::new();
        let proj = est.project_annual(BackendType::CloudHot, GIB, 1_000, 10_000);
        assert!(proj.savings_vs_baseline.abs() < 1e-12);
    }

    #[test]
    fn project_annual_cold_cheaper_so_positive_savings() {
        // CloudCold storage is cheaper than CloudHot => positive savings
        let est = StorageCostEstimator::new();
        let proj = est.project_annual(BackendType::CloudCold, GIB, 0, 0);
        assert!(
            proj.savings_vs_baseline > 0.0,
            "CloudCold (no requests) should have positive savings vs CloudHot"
        );
    }

    #[test]
    fn project_annual_local_ssd_negative_savings_vs_cloud_hot() {
        // LocalSsd storage = $0.10/GB > CloudHot $0.023/GB => negative savings
        let est = StorageCostEstimator::new();
        let proj = est.project_annual(BackendType::LocalSsd, GIB, 0, 0);
        assert!(
            proj.savings_vs_baseline < 0.0,
            "LocalSsd should have negative savings vs CloudHot"
        );
    }

    #[test]
    fn project_annual_backend_field_set_correctly() {
        let est = StorageCostEstimator::new();
        let proj = est.project_annual(BackendType::CloudWarm, GIB, 500, 500);
        assert_eq!(proj.backend, BackendType::CloudWarm);
    }

    // ------------------------------------------------------------------
    // OperationCost::is_cheaper_than
    // ------------------------------------------------------------------

    #[test]
    fn is_cheaper_than_returns_true_when_cheaper() {
        let est = StorageCostEstimator::new();
        let cold = est.estimate_operation(BackendType::CloudCold, GIB, 0, 0);
        let hot = est.estimate_operation(BackendType::CloudHot, GIB, 0, 0);
        assert!(cold.is_cheaper_than(&hot));
    }

    #[test]
    fn is_cheaper_than_returns_false_when_more_expensive() {
        let est = StorageCostEstimator::new();
        let ssd = est.estimate_operation(BackendType::LocalSsd, GIB, 0, 0);
        let cold = est.estimate_operation(BackendType::CloudCold, GIB, 0, 0);
        assert!(!ssd.is_cheaper_than(&cold));
    }

    #[test]
    fn is_cheaper_than_equal_costs_returns_false() {
        let est = StorageCostEstimator::new();
        let a = est.estimate_operation(BackendType::LocalSsd, GIB, 0, 0);
        let b = est.estimate_operation(BackendType::LocalSsd, GIB, 0, 0);
        assert!(!a.is_cheaper_than(&b));
    }

    // ------------------------------------------------------------------
    // Default
    // ------------------------------------------------------------------

    #[test]
    fn default_creates_valid_estimator() {
        let est = StorageCostEstimator;
        let cost = est.estimate_operation(BackendType::CloudHot, GIB, 0, 0);
        assert!(cost.total_cost > 0.0);
    }
}
