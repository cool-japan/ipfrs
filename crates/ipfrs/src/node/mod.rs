//! IPFRS Node - Unified node implementation (module root)

use ipfrs_core::Cid;
use ipfrs_interface::metrics::IpfrsMetrics;
use ipfrs_network::NetworkNode;
use ipfrs_semantic::{DistanceMetric, SemanticRouter};
use ipfrs_storage::{CachedBlockStore, SledBlockStore};
use ipfrs_tensorlogic::TensorLogicStore;
use once_cell::sync::OnceCell;
use std::sync::Arc;
use std::time::SystemTime;

use crate::auth::AuthManager;
use crate::pin::PinManager;
use crate::tls::TlsManager;

mod auth_ops;
mod block_ops;
mod core;
mod dag_ops;
mod network_ops;
mod pin_ops;
mod repo_ops;
mod semantic_ops;
mod tensorlogic_ops;

pub use core::{NodeConfig, NodeStatus};
pub use tensorlogic_ops::DistributedInferResult;

/// Type alias for the hot-cache-wrapped Sled block store used inside Node.
pub(super) type NodeStore = CachedBlockStore<SledBlockStore>;

/// IPFRS unified node combining all layers
pub struct Node {
    pub(super) config: NodeConfig,
    pub(super) network: Option<NetworkNode>,
    pub(super) storage: Option<Arc<NodeStore>>,
    pub(super) semantic: OnceCell<Arc<SemanticRouter>>,
    pub(super) tensorlogic: OnceCell<Arc<TensorLogicStore<NodeStore>>>,
    pub(super) auth_manager: Option<Arc<AuthManager>>,
    pub(super) tls_manager: Option<Arc<TlsManager>>,
    pub(super) pin_manager: Arc<PinManager>,
    pub(super) startup_time: Option<SystemTime>,
    /// Prometheus metrics for this node instance.
    ///
    /// All block operations, DHT calls, inference sessions, GossipSub events,
    /// storage stats, and GC runs are recorded here.
    pub metrics: Arc<IpfrsMetrics>,
}

impl Node {
    /// Get storage handle or return error if not started.
    pub(super) fn storage(&self) -> ipfrs_core::Result<&Arc<NodeStore>> {
        self.storage.as_ref().ok_or_else(|| {
            ipfrs_core::Error::Initialization("Node not started - call start() first".to_string())
        })
    }

    /// Get semantic router handle or return error if not enabled.
    /// Lazily initializes the semantic router on first access.
    pub(super) fn semantic(&self) -> ipfrs_core::Result<&Arc<SemanticRouter>> {
        use ipfrs_semantic::SemanticRouter as SR;
        if !self.config.enable_semantic {
            return Err(ipfrs_core::Error::Initialization(
                "Semantic routing not enabled - set enable_semantic=true in config".to_string(),
            ));
        }

        self.semantic
            .get_or_try_init(|| SR::new(self.config.semantic.clone()).map(Arc::new))
    }

    /// Get TensorLogic store handle or return error if not enabled.
    /// Lazily initializes the TensorLogic store on first access.
    pub(super) fn tensorlogic(&self) -> ipfrs_core::Result<&Arc<TensorLogicStore<NodeStore>>> {
        if !self.config.enable_tensorlogic {
            return Err(ipfrs_core::Error::Initialization(
                "TensorLogic not enabled - set enable_tensorlogic=true in config".to_string(),
            ));
        }

        let storage = self.storage()?;

        self.tensorlogic
            .get_or_try_init(|| TensorLogicStore::new(storage.clone()).map(Arc::new))
    }

    /// Get network handle or return error if not started
    pub(super) fn network(&self) -> ipfrs_core::Result<&NetworkNode> {
        self.network.as_ref().ok_or_else(|| {
            ipfrs_core::Error::Initialization("Node not started - call start() first".to_string())
        })
    }

    /// Get mutable network handle or return error if not started
    pub(super) fn network_mut(&mut self) -> ipfrs_core::Result<&mut NetworkNode> {
        self.network.as_mut().ok_or_else(|| {
            ipfrs_core::Error::Initialization("Node not started - call start() first".to_string())
        })
    }
}

/// Storage statistics
#[derive(Debug, Clone)]
pub struct StorageStats {
    /// Number of blocks stored
    pub num_blocks: usize,
    /// Whether storage is empty
    pub is_empty: bool,
    /// Deduplication statistics snapshot (Clone-friendly view of AtomicU64 counters)
    pub dedup: ipfrs_storage::DeduplicationStatsSnapshot,
}

/// Block statistics
#[derive(Debug, Clone)]
pub struct BlockStat {
    /// Content identifier
    pub cid: Cid,
    /// Size in bytes
    pub size: usize,
}

/// Semantic router statistics
#[derive(Debug, Clone)]
pub struct SemanticStats {
    /// Number of indexed vectors
    pub num_vectors: usize,
    /// Vector dimension
    pub dimension: usize,
    /// Distance metric used
    pub metric: DistanceMetric,
    /// Current cache size
    pub cache_size: usize,
    /// Maximum cache capacity
    pub cache_capacity: usize,
}

/// TensorLogic statistics
#[derive(Debug, Clone)]
pub struct TensorLogicStats {
    /// Whether TensorLogic is enabled
    pub enabled: bool,
    /// Number of facts in knowledge base
    pub num_facts: usize,
    /// Number of rules in knowledge base
    pub num_rules: usize,
}

/// Re-export the orphan GC result from ipfrs-storage so callers only need to
/// import from `ipfrs`.
pub use ipfrs_storage::OrphanGcResult;

/// Result of garbage collection
#[derive(Debug, Clone)]
pub struct GcResult {
    /// Number of blocks collected (deleted)
    pub blocks_collected: u64,
    /// Bytes freed
    pub bytes_freed: u64,
    /// Number of blocks marked as reachable
    pub blocks_marked: u64,
    /// Number of blocks scanned
    pub blocks_scanned: u64,
    /// Duration of GC run
    pub duration: std::time::Duration,
    /// Whether GC was cancelled
    pub cancelled: bool,
}

/// Result of filesystem check
#[derive(Debug, Clone)]
pub struct FsckResult {
    /// Number of blocks checked
    pub blocks_checked: u64,
    /// Number of valid blocks
    pub blocks_valid: u64,
    /// List of corrupt blocks
    pub blocks_corrupt: Vec<Cid>,
    /// List of missing blocks
    pub blocks_missing: Vec<Cid>,
}

/// Result of DAG export operation
#[derive(Debug, Clone)]
pub struct DagExportStats {
    /// Number of blocks exported
    pub blocks_exported: u64,
    /// Total bytes exported
    pub bytes_exported: u64,
}

/// Result of DAG import operation
#[derive(Debug, Clone)]
pub struct DagImportStats {
    /// Number of blocks imported
    pub blocks_imported: u64,
    /// Total bytes imported
    pub bytes_imported: u64,
}
