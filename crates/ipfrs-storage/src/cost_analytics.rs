//! Cost Analytics for storage optimization
//!
//! This module provides cost tracking and optimization for storage:
//! - Per-tier cost tracking (hot/warm/cold/archive)
//! - Storage operation cost analysis (reads, writes, retrievals)
//! - Cloud storage cost modeling (AWS S3, Azure, GCP)
//! - Cost-aware data placement recommendations
//! - Budget tracking and alerting
//! - Cost projection and forecasting

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

/// Storage tier for cost calculation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CostTier {
    /// Hot storage (frequently accessed, expensive)
    Hot,
    /// Standard storage (balanced cost/performance)
    Standard,
    /// Infrequent access (cheaper storage, retrieval fees)
    InfrequentAccess,
    /// Archive (very cheap storage, high retrieval cost)
    Archive,
    /// Glacier (coldest tier, highest retrieval cost)
    Glacier,
}

/// Cloud provider
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CloudProvider {
    AWS,
    Azure,
    GCP,
    Custom,
}

/// Cost model for a storage tier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierCostModel {
    /// Storage cost per GB per month
    pub storage_cost_per_gb_month: f64,
    /// PUT/POST/LIST request cost per 1000 requests
    pub write_request_cost_per_1k: f64,
    /// GET/HEAD request cost per 1000 requests
    pub read_request_cost_per_1k: f64,
    /// Data retrieval cost per GB (for cold tiers)
    pub retrieval_cost_per_gb: f64,
    /// Data transfer out cost per GB
    pub transfer_cost_per_gb: f64,
    /// Minimum storage duration in days
    pub min_storage_days: u32,
    /// Early deletion fee per GB
    pub early_deletion_cost_per_gb: f64,
}

impl TierCostModel {
    /// AWS S3 Standard tier pricing (approximate)
    pub fn aws_s3_standard() -> Self {
        Self {
            storage_cost_per_gb_month: 0.023,
            write_request_cost_per_1k: 0.005,
            read_request_cost_per_1k: 0.0004,
            retrieval_cost_per_gb: 0.0,
            transfer_cost_per_gb: 0.09,
            min_storage_days: 0,
            early_deletion_cost_per_gb: 0.0,
        }
    }

    /// AWS S3 Infrequent Access tier pricing
    pub fn aws_s3_infrequent() -> Self {
        Self {
            storage_cost_per_gb_month: 0.0125,
            write_request_cost_per_1k: 0.01,
            read_request_cost_per_1k: 0.001,
            retrieval_cost_per_gb: 0.01,
            transfer_cost_per_gb: 0.09,
            min_storage_days: 30,
            early_deletion_cost_per_gb: 0.0125,
        }
    }

    /// AWS S3 Glacier tier pricing
    pub fn aws_s3_glacier() -> Self {
        Self {
            storage_cost_per_gb_month: 0.004,
            write_request_cost_per_1k: 0.03,
            read_request_cost_per_1k: 0.0004,
            retrieval_cost_per_gb: 0.02, // Standard retrieval
            transfer_cost_per_gb: 0.09,
            min_storage_days: 90,
            early_deletion_cost_per_gb: 0.012,
        }
    }

    /// Azure Blob Hot tier pricing
    pub fn azure_hot() -> Self {
        Self {
            storage_cost_per_gb_month: 0.018,
            write_request_cost_per_1k: 0.0055,
            read_request_cost_per_1k: 0.00044,
            retrieval_cost_per_gb: 0.0,
            transfer_cost_per_gb: 0.087,
            min_storage_days: 0,
            early_deletion_cost_per_gb: 0.0,
        }
    }

    /// Azure Blob Cool tier pricing
    pub fn azure_cool() -> Self {
        Self {
            storage_cost_per_gb_month: 0.01,
            write_request_cost_per_1k: 0.01,
            read_request_cost_per_1k: 0.001,
            retrieval_cost_per_gb: 0.01,
            transfer_cost_per_gb: 0.087,
            min_storage_days: 30,
            early_deletion_cost_per_gb: 0.01,
        }
    }

    /// Azure Blob Archive tier pricing
    pub fn azure_archive() -> Self {
        Self {
            storage_cost_per_gb_month: 0.002,
            write_request_cost_per_1k: 0.011,
            read_request_cost_per_1k: 0.0055,
            retrieval_cost_per_gb: 0.02,
            transfer_cost_per_gb: 0.087,
            min_storage_days: 180,
            early_deletion_cost_per_gb: 0.006,
        }
    }

    /// GCP Standard storage pricing
    pub fn gcp_standard() -> Self {
        Self {
            storage_cost_per_gb_month: 0.020,
            write_request_cost_per_1k: 0.005,
            read_request_cost_per_1k: 0.0004,
            retrieval_cost_per_gb: 0.0,
            transfer_cost_per_gb: 0.12,
            min_storage_days: 0,
            early_deletion_cost_per_gb: 0.0,
        }
    }

    /// GCP Nearline storage pricing
    pub fn gcp_nearline() -> Self {
        Self {
            storage_cost_per_gb_month: 0.010,
            write_request_cost_per_1k: 0.01,
            read_request_cost_per_1k: 0.001,
            retrieval_cost_per_gb: 0.01,
            transfer_cost_per_gb: 0.12,
            min_storage_days: 30,
            early_deletion_cost_per_gb: 0.010,
        }
    }

    /// GCP Coldline storage pricing
    pub fn gcp_coldline() -> Self {
        Self {
            storage_cost_per_gb_month: 0.004,
            write_request_cost_per_1k: 0.01,
            read_request_cost_per_1k: 0.005,
            retrieval_cost_per_gb: 0.02,
            transfer_cost_per_gb: 0.12,
            min_storage_days: 90,
            early_deletion_cost_per_gb: 0.012,
        }
    }
}

/// Usage tracking for cost calculation
#[derive(Debug)]
pub struct UsageMetrics {
    /// Total storage in bytes
    pub storage_bytes: AtomicU64,
    /// Total write requests
    pub write_requests: AtomicU64,
    /// Total read requests
    pub read_requests: AtomicU64,
    /// Total bytes retrieved
    pub bytes_retrieved: AtomicU64,
    /// Total bytes transferred out
    pub bytes_transferred: AtomicU64,
    /// Start time for this period
    pub period_start: parking_lot::Mutex<SystemTime>,
}

impl UsageMetrics {
    fn new() -> Self {
        Self {
            storage_bytes: AtomicU64::new(0),
            write_requests: AtomicU64::new(0),
            read_requests: AtomicU64::new(0),
            bytes_retrieved: AtomicU64::new(0),
            bytes_transferred: AtomicU64::new(0),
            period_start: parking_lot::Mutex::new(SystemTime::now()),
        }
    }

    fn record_write(&self, bytes: u64) {
        self.storage_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.write_requests.fetch_add(1, Ordering::Relaxed);
    }

    fn record_read(&self, bytes: u64) {
        self.read_requests.fetch_add(1, Ordering::Relaxed);
        self.bytes_retrieved.fetch_add(bytes, Ordering::Relaxed);
    }

    fn record_delete(&self, bytes: u64) {
        self.storage_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }

    fn record_transfer(&self, bytes: u64) {
        self.bytes_transferred.fetch_add(bytes, Ordering::Relaxed);
    }
}

/// Cost breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBreakdown {
    /// Storage costs
    pub storage_cost: f64,
    /// Write request costs
    pub write_request_cost: f64,
    /// Read request costs
    pub read_request_cost: f64,
    /// Data retrieval costs
    pub retrieval_cost: f64,
    /// Data transfer costs
    pub transfer_cost: f64,
    /// Early deletion costs
    pub early_deletion_cost: f64,
    /// Total cost
    pub total_cost: f64,
}

impl CostBreakdown {
    fn calculate_total(&mut self) {
        self.total_cost = self.storage_cost
            + self.write_request_cost
            + self.read_request_cost
            + self.retrieval_cost
            + self.transfer_cost
            + self.early_deletion_cost;
    }
}

/// Cost optimizer
pub struct CostAnalyzer {
    /// Cost models by tier
    cost_models: HashMap<CostTier, TierCostModel>,
    /// Usage metrics by tier
    usage_metrics: HashMap<CostTier, UsageMetrics>,
    /// Cloud provider
    #[allow(dead_code)]
    provider: CloudProvider,
}

impl CostAnalyzer {
    /// Create a new cost analyzer for a specific provider
    pub fn new(provider: CloudProvider) -> Self {
        let mut cost_models = HashMap::new();

        match provider {
            CloudProvider::AWS => {
                cost_models.insert(CostTier::Standard, TierCostModel::aws_s3_standard());
                cost_models.insert(
                    CostTier::InfrequentAccess,
                    TierCostModel::aws_s3_infrequent(),
                );
                cost_models.insert(CostTier::Glacier, TierCostModel::aws_s3_glacier());
            }
            CloudProvider::Azure => {
                cost_models.insert(CostTier::Hot, TierCostModel::azure_hot());
                cost_models.insert(CostTier::InfrequentAccess, TierCostModel::azure_cool());
                cost_models.insert(CostTier::Archive, TierCostModel::azure_archive());
            }
            CloudProvider::GCP => {
                cost_models.insert(CostTier::Standard, TierCostModel::gcp_standard());
                cost_models.insert(CostTier::InfrequentAccess, TierCostModel::gcp_nearline());
                cost_models.insert(CostTier::Archive, TierCostModel::gcp_coldline());
            }
            CloudProvider::Custom => {}
        }

        Self {
            cost_models,
            usage_metrics: HashMap::new(),
            provider,
        }
    }

    /// Set custom cost model for a tier
    pub fn set_cost_model(&mut self, tier: CostTier, model: TierCostModel) {
        self.cost_models.insert(tier, model);
    }

    /// Record a write operation
    pub fn record_write(&mut self, tier: CostTier, bytes: u64) {
        self.usage_metrics
            .entry(tier)
            .or_insert_with(UsageMetrics::new)
            .record_write(bytes);
    }

    /// Record a read operation
    pub fn record_read(&mut self, tier: CostTier, bytes: u64) {
        self.usage_metrics
            .entry(tier)
            .or_insert_with(UsageMetrics::new)
            .record_read(bytes);
    }

    /// Record a delete operation
    pub fn record_delete(&mut self, tier: CostTier, bytes: u64) {
        self.usage_metrics
            .entry(tier)
            .or_insert_with(UsageMetrics::new)
            .record_delete(bytes);
    }

    /// Record data transfer
    pub fn record_transfer(&mut self, tier: CostTier, bytes: u64) {
        self.usage_metrics
            .entry(tier)
            .or_insert_with(UsageMetrics::new)
            .record_transfer(bytes);
    }

    /// Calculate cost for a specific tier
    pub fn calculate_tier_cost(&self, tier: CostTier, days: u32) -> Option<CostBreakdown> {
        let model = self.cost_models.get(&tier)?;
        let metrics = self.usage_metrics.get(&tier)?;

        let storage_gb = metrics.storage_bytes.load(Ordering::Relaxed) as f64 / 1_073_741_824.0;
        let write_requests = metrics.write_requests.load(Ordering::Relaxed) as f64;
        let read_requests = metrics.read_requests.load(Ordering::Relaxed) as f64;
        let retrieved_gb = metrics.bytes_retrieved.load(Ordering::Relaxed) as f64 / 1_073_741_824.0;
        let transferred_gb =
            metrics.bytes_transferred.load(Ordering::Relaxed) as f64 / 1_073_741_824.0;

        let months = days as f64 / 30.0;

        let mut breakdown = CostBreakdown {
            storage_cost: storage_gb * model.storage_cost_per_gb_month * months,
            write_request_cost: (write_requests / 1000.0) * model.write_request_cost_per_1k,
            read_request_cost: (read_requests / 1000.0) * model.read_request_cost_per_1k,
            retrieval_cost: retrieved_gb * model.retrieval_cost_per_gb,
            transfer_cost: transferred_gb * model.transfer_cost_per_gb,
            early_deletion_cost: 0.0, // Would need deletion tracking
            total_cost: 0.0,
        };

        breakdown.calculate_total();
        Some(breakdown)
    }

    /// Calculate total cost across all tiers
    pub fn calculate_total_cost(&self, days: u32) -> CostBreakdown {
        let mut total = CostBreakdown {
            storage_cost: 0.0,
            write_request_cost: 0.0,
            read_request_cost: 0.0,
            retrieval_cost: 0.0,
            transfer_cost: 0.0,
            early_deletion_cost: 0.0,
            total_cost: 0.0,
        };

        for tier in self.usage_metrics.keys() {
            if let Some(breakdown) = self.calculate_tier_cost(*tier, days) {
                total.storage_cost += breakdown.storage_cost;
                total.write_request_cost += breakdown.write_request_cost;
                total.read_request_cost += breakdown.read_request_cost;
                total.retrieval_cost += breakdown.retrieval_cost;
                total.transfer_cost += breakdown.transfer_cost;
                total.early_deletion_cost += breakdown.early_deletion_cost;
            }
        }

        total.calculate_total();
        total
    }

    /// Get tier recommendation based on access pattern
    pub fn recommend_tier(&self, access_frequency: f64, data_size_gb: f64) -> TierRecommendation {
        let mut recommendations = Vec::new();

        for (tier, model) in &self.cost_models {
            // Calculate cost for this tier over 30 days
            let storage_cost = data_size_gb * model.storage_cost_per_gb_month;

            // Estimate request costs based on access frequency
            let monthly_accesses = access_frequency * 30.0;
            let read_cost = (monthly_accesses / 1000.0) * model.read_request_cost_per_1k;
            let retrieval_cost = monthly_accesses * data_size_gb * model.retrieval_cost_per_gb;

            let total_cost = storage_cost + read_cost + retrieval_cost;

            recommendations.push(TierOption {
                tier: *tier,
                estimated_monthly_cost: total_cost,
                storage_cost,
                access_cost: read_cost + retrieval_cost,
            });
        }

        // Sort by cost
        recommendations.sort_by(|a, b| {
            a.estimated_monthly_cost
                .partial_cmp(&b.estimated_monthly_cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        TierRecommendation {
            recommended_tier: recommendations[0].tier,
            options: recommendations,
            access_frequency,
            data_size_gb,
        }
    }

    /// Project costs for next N days
    pub fn project_costs(&self, days: u32) -> CostProjection {
        let current_cost = self.calculate_total_cost(days);
        let daily_rate = current_cost.total_cost / days as f64;

        CostProjection {
            current_total: current_cost.total_cost,
            daily_rate,
            projected_monthly: daily_rate * 30.0,
            projected_yearly: daily_rate * 365.0,
            breakdown: current_cost,
        }
    }
}

/// Tier recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierRecommendation {
    pub recommended_tier: CostTier,
    pub options: Vec<TierOption>,
    pub access_frequency: f64,
    pub data_size_gb: f64,
}

/// Tier option
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierOption {
    pub tier: CostTier,
    pub estimated_monthly_cost: f64,
    pub storage_cost: f64,
    pub access_cost: f64,
}

/// Cost projection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostProjection {
    pub current_total: f64,
    pub daily_rate: f64,
    pub projected_monthly: f64,
    pub projected_yearly: f64,
    pub breakdown: CostBreakdown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_calculation() {
        let mut analyzer = CostAnalyzer::new(CloudProvider::AWS);

        // Simulate 100 GB in standard tier
        analyzer.record_write(CostTier::Standard, 100 * 1024 * 1024 * 1024);

        // Calculate monthly cost
        let breakdown = analyzer
            .calculate_tier_cost(CostTier::Standard, 30)
            .unwrap();

        // Should be approximately $2.30 for 100GB at $0.023/GB/month
        assert!(breakdown.storage_cost > 2.0 && breakdown.storage_cost < 2.5);
    }

    #[test]
    fn test_tier_recommendation() {
        let analyzer = CostAnalyzer::new(CloudProvider::AWS);

        // High access frequency - should recommend standard
        let rec = analyzer.recommend_tier(100.0, 10.0);
        assert_eq!(rec.recommended_tier, CostTier::Standard);

        // Very low access frequency - should recommend cheaper tier
        // Glacier retrieval = monthly_accesses * data_gb * $0.02/GB
        // Storage savings vs Standard = data_gb * ($0.023 - $0.004)/month
        // For Glacier to be cheaper: monthly_accesses < 0.95
        // With access_freq=0.01, monthly_accesses=0.3 < 0.95, so Glacier wins
        let rec = analyzer.recommend_tier(0.01, 10.0);
        // With very low access, Glacier or IA should be cheaper
        assert_ne!(rec.recommended_tier, CostTier::Standard);
    }

    #[test]
    fn test_cost_models() {
        let model = TierCostModel::aws_s3_standard();
        assert_eq!(model.storage_cost_per_gb_month, 0.023);

        let model = TierCostModel::azure_hot();
        assert_eq!(model.storage_cost_per_gb_month, 0.018);

        let model = TierCostModel::gcp_standard();
        assert_eq!(model.storage_cost_per_gb_month, 0.020);
    }

    #[test]
    fn test_cost_projection() {
        let mut analyzer = CostAnalyzer::new(CloudProvider::AWS);
        analyzer.record_write(CostTier::Standard, 100 * 1024 * 1024 * 1024);

        let projection = analyzer.project_costs(30);
        assert!(projection.projected_monthly > 0.0);
        assert!(projection.projected_yearly > projection.projected_monthly);
    }
}
