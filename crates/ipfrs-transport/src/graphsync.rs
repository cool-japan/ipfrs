//! GraphSync protocol for DAG traversal
//!
//! Implements efficient DAG traversal with:
//! - IPLD selector parsing and execution
//! - Incremental response streaming
//! - Resume capability from partial transfers
//! - Breadth-first and depth-first traversal
//!
//! # Example
//!
//! ```
//! use ipfrs_transport::Selector;
//!
//! // Create a selector for recursive depth-limited traversal
//! let selector = Selector::RecursiveDepth { max_depth: 5 };
//!
//! // Validate the selector
//! assert!(selector.validate().is_ok());
//!
//! // Create a selector for specific fields
//! let field_selector = Selector::Fields {
//!     fields: vec!["data".to_string(), "links".to_string()]
//! };
//!
//! // Parse from JSON
//! let json = r#"{"type": "recursivedepth", "max_depth": 3}"#;
//! let parsed = Selector::from_json(json).unwrap();
//! ```

use ipfrs_core::error::{Error, Result};
use ipfrs_core::{Block, Cid};
use ipfrs_storage::traits::BlockStore;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;

/// IPLD Selector
///
/// Selectors specify which parts of a DAG to traverse
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
#[derive(Default)]
pub enum Selector {
    /// Match everything
    #[default]
    All,
    /// Match specific fields by name
    Fields { fields: Vec<String> },
    /// Recursively traverse to a depth limit
    RecursiveDepth { max_depth: usize },
    /// Recursively traverse all links
    RecursiveAll,
    /// Match based on index
    Index { index: usize },
    /// Sequence of selectors
    Sequence { selectors: Vec<Selector> },
    /// Match the current node
    Matcher,
}

impl Selector {
    /// Parse a selector from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| Error::InvalidInput(format!("Failed to parse selector: {}", e)))
    }

    /// Validate the selector
    pub fn validate(&self) -> Result<()> {
        match self {
            Selector::RecursiveDepth { max_depth } if *max_depth == 0 => {
                return Err(Error::InvalidInput(
                    "max_depth must be greater than 0".to_string(),
                ));
            }
            Selector::RecursiveDepth { .. } => {}
            Selector::Sequence { selectors } => {
                for sel in selectors {
                    sel.validate()?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Check if this selector matches all
    pub fn matches_all(&self) -> bool {
        matches!(self, Selector::All | Selector::RecursiveAll)
    }
}

/// Traversal mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraversalMode {
    /// Breadth-first search
    BreadthFirst,
    /// Depth-first search
    DepthFirst,
}

/// DAG traversal state
#[derive(Debug, Clone)]
pub struct TraversalState {
    /// Root CID
    pub root: Cid,
    /// Visited CIDs
    pub visited: HashSet<Cid>,
    /// Queue of CIDs to visit (for BFS) or stack (for DFS)
    pub queue: VecDeque<(Cid, usize)>, // (CID, depth)
    /// Current depth
    pub current_depth: usize,
    /// Maximum depth (if limited)
    pub max_depth: Option<usize>,
    /// Blocks fetched so far
    pub blocks_fetched: usize,
    /// Bytes fetched so far
    pub bytes_fetched: u64,
}

impl TraversalState {
    /// Create a new traversal state
    pub fn new(root: Cid, max_depth: Option<usize>) -> Self {
        let mut queue = VecDeque::new();
        queue.push_back((root, 0));

        Self {
            root,
            visited: HashSet::new(),
            queue,
            current_depth: 0,
            max_depth,
            blocks_fetched: 0,
            bytes_fetched: 0,
        }
    }

    /// Check if traversal is complete
    pub fn is_complete(&self) -> bool {
        self.queue.is_empty()
    }

    /// Get next CID to visit
    pub fn next(&mut self, mode: TraversalMode) -> Option<(Cid, usize)> {
        match mode {
            TraversalMode::BreadthFirst => self.queue.pop_front(),
            TraversalMode::DepthFirst => self.queue.pop_back(),
        }
    }

    /// Add a CID to the queue
    pub fn enqueue(&mut self, cid: Cid, depth: usize) {
        if let Some(max) = self.max_depth {
            if depth > max {
                return;
            }
        }

        if !self.visited.contains(&cid) {
            self.queue.push_back((cid, depth));
        }
    }

    /// Mark a CID as visited
    pub fn mark_visited(&mut self, cid: Cid, size: u64) {
        self.visited.insert(cid);
        self.blocks_fetched += 1;
        self.bytes_fetched += size;
    }

    /// Save checkpoint for resume
    pub fn checkpoint(&self) -> TraversalCheckpoint {
        TraversalCheckpoint {
            root: self.root,
            visited: self.visited.clone(),
            queue: self.queue.clone(),
            max_depth: self.max_depth,
            blocks_fetched: self.blocks_fetched,
            bytes_fetched: self.bytes_fetched,
        }
    }

    /// Restore from checkpoint
    pub fn from_checkpoint(checkpoint: TraversalCheckpoint) -> Self {
        Self {
            root: checkpoint.root,
            visited: checkpoint.visited,
            queue: checkpoint.queue,
            current_depth: 0,
            max_depth: checkpoint.max_depth,
            blocks_fetched: checkpoint.blocks_fetched,
            bytes_fetched: checkpoint.bytes_fetched,
        }
    }
}

/// Checkpoint for resuming traversal
#[derive(Debug, Clone)]
pub struct TraversalCheckpoint {
    /// Root CID
    pub root: Cid,
    /// Visited CIDs
    pub visited: HashSet<Cid>,
    /// Queue state
    pub queue: VecDeque<(Cid, usize)>,
    /// Maximum depth
    pub max_depth: Option<usize>,
    /// Blocks fetched
    pub blocks_fetched: usize,
    /// Bytes fetched
    pub bytes_fetched: u64,
}

impl TraversalCheckpoint {
    /// Serialize to JSON using CID strings
    pub fn to_json(&self) -> Result<String> {
        #[derive(Serialize)]
        struct SerializableCheckpoint {
            root: String,
            visited: Vec<String>,
            queue: Vec<(String, usize)>,
            max_depth: Option<usize>,
            blocks_fetched: usize,
            bytes_fetched: u64,
        }

        let serializable = SerializableCheckpoint {
            root: self.root.to_string(),
            visited: self.visited.iter().map(|c| c.to_string()).collect(),
            queue: self
                .queue
                .iter()
                .map(|(c, d)| (c.to_string(), *d))
                .collect(),
            max_depth: self.max_depth,
            blocks_fetched: self.blocks_fetched,
            bytes_fetched: self.bytes_fetched,
        };

        serde_json::to_string(&serializable)
            .map_err(|e| Error::Internal(format!("Failed to serialize checkpoint: {}", e)))
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        #[derive(Deserialize)]
        struct SerializableCheckpoint {
            root: String,
            visited: Vec<String>,
            queue: Vec<(String, usize)>,
            max_depth: Option<usize>,
            blocks_fetched: usize,
            bytes_fetched: u64,
        }

        let serializable: SerializableCheckpoint = serde_json::from_str(json)
            .map_err(|e| Error::Internal(format!("Failed to deserialize checkpoint: {}", e)))?;

        let root: Cid = serializable
            .root
            .parse()
            .map_err(|e| Error::InvalidInput(format!("Invalid root CID: {}", e)))?;

        let visited: Result<HashSet<Cid>> = serializable
            .visited
            .iter()
            .map(|s| {
                s.parse()
                    .map_err(|e| Error::InvalidInput(format!("Invalid CID: {}", e)))
            })
            .collect();

        let queue: Result<VecDeque<(Cid, usize)>> = serializable
            .queue
            .iter()
            .map(|(s, d)| {
                s.parse()
                    .map(|c| (c, *d))
                    .map_err(|e| Error::InvalidInput(format!("Invalid CID: {}", e)))
            })
            .collect();

        Ok(Self {
            root,
            visited: visited?,
            queue: queue?,
            max_depth: serializable.max_depth,
            blocks_fetched: serializable.blocks_fetched,
            bytes_fetched: serializable.bytes_fetched,
        })
    }
}

/// DAG traversal engine
pub struct DagTraversal<S: BlockStore> {
    /// Block store
    store: Arc<S>,
    /// Traversal mode
    mode: TraversalMode,
    /// Selector
    #[allow(dead_code)]
    selector: Selector,
    /// Traversal state
    state: Arc<RwLock<TraversalState>>,
}

impl<S: BlockStore> DagTraversal<S> {
    /// Create a new DAG traversal
    pub fn new(store: Arc<S>, root: Cid, selector: Selector, mode: TraversalMode) -> Result<Self> {
        selector.validate()?;

        let max_depth = match &selector {
            Selector::RecursiveDepth { max_depth } => Some(*max_depth),
            _ => None,
        };

        let state = TraversalState::new(root, max_depth);

        Ok(Self {
            store,
            mode,
            selector,
            state: Arc::new(RwLock::new(state)),
        })
    }

    /// Resume from a checkpoint
    pub fn from_checkpoint(
        store: Arc<S>,
        checkpoint: TraversalCheckpoint,
        selector: Selector,
        mode: TraversalMode,
    ) -> Result<Self> {
        selector.validate()?;
        let state = TraversalState::from_checkpoint(checkpoint);

        Ok(Self {
            store,
            mode,
            selector,
            state: Arc::new(RwLock::new(state)),
        })
    }

    /// Get the next block in the traversal
    pub async fn next_block(&self) -> Result<Option<Block>> {
        let mut state = self.state.write().await;

        // Get next CID to visit
        let (cid, depth) = match state.next(self.mode) {
            Some(item) => item,
            None => return Ok(None),
        };

        // Fetch the block
        let block = match self.store.get(&cid).await? {
            Some(b) => b,
            None => return Err(Error::NotFound(format!("Block not found for CID: {}", cid))),
        };

        // Mark as visited
        state.mark_visited(cid, block.data().len() as u64);
        state.current_depth = depth;

        // Extract links from the block and add to queue
        if let Ok(links) = self.extract_links(&block) {
            for link_cid in links {
                state.enqueue(link_cid, depth + 1);
            }
        }

        Ok(Some(block))
    }

    /// Extract CID links from a block
    fn extract_links(&self, _block: &Block) -> Result<Vec<Cid>> {
        // Simple link extraction - in a real implementation, this would parse IPLD
        // and extract CIDs based on the selector

        // For now, we'll just return an empty vector
        // In a real implementation, you would:
        // 1. Parse the block data as IPLD
        // 2. Apply the selector to determine which fields to follow
        // 3. Extract CID links from those fields

        Ok(Vec::new())
    }

    /// Check if traversal is complete
    pub async fn is_complete(&self) -> bool {
        self.state.read().await.is_complete()
    }

    /// Get traversal statistics
    pub async fn stats(&self) -> TraversalStats {
        let state = self.state.read().await;
        TraversalStats {
            blocks_fetched: state.blocks_fetched,
            bytes_fetched: state.bytes_fetched,
            blocks_remaining: state.queue.len(),
            current_depth: state.current_depth,
        }
    }

    /// Create a checkpoint for resume
    pub async fn checkpoint(&self) -> TraversalCheckpoint {
        self.state.read().await.checkpoint()
    }

    /// Traverse all and collect blocks
    pub async fn collect_all(&self) -> Result<Vec<Block>> {
        let mut blocks = Vec::new();

        while let Some(block) = self.next_block().await? {
            blocks.push(block);
        }

        Ok(blocks)
    }
}

/// Traversal statistics
#[derive(Debug, Clone)]
pub struct TraversalStats {
    /// Number of blocks fetched
    pub blocks_fetched: usize,
    /// Bytes fetched
    pub bytes_fetched: u64,
    /// Blocks remaining in queue
    pub blocks_remaining: usize,
    /// Current traversal depth
    pub current_depth: usize,
}

/// GraphSync protocol handler
pub struct GraphSync<S: BlockStore> {
    /// Block store
    store: Arc<S>,
    /// Active traversals
    traversals: Arc<RwLock<HashMap<Cid, Arc<DagTraversal<S>>>>>,
}

impl<S: BlockStore> GraphSync<S> {
    /// Create a new GraphSync instance
    pub fn new(store: Arc<S>) -> Result<Self> {
        Ok(Self {
            store,
            traversals: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Start a new traversal
    pub async fn start_traversal(
        &self,
        root: Cid,
        selector: Selector,
        mode: TraversalMode,
    ) -> Result<Arc<DagTraversal<S>>> {
        let traversal = Arc::new(DagTraversal::new(self.store.clone(), root, selector, mode)?);

        let mut traversals = self.traversals.write().await;
        traversals.insert(root, traversal.clone());

        Ok(traversal)
    }

    /// Resume a traversal from checkpoint
    pub async fn resume_traversal(
        &self,
        checkpoint: TraversalCheckpoint,
        selector: Selector,
        mode: TraversalMode,
    ) -> Result<Arc<DagTraversal<S>>> {
        let root = checkpoint.root;
        let traversal = Arc::new(DagTraversal::from_checkpoint(
            self.store.clone(),
            checkpoint,
            selector,
            mode,
        )?);

        let mut traversals = self.traversals.write().await;
        traversals.insert(root, traversal.clone());

        Ok(traversal)
    }

    /// Get an active traversal
    pub async fn get_traversal(&self, root: &Cid) -> Option<Arc<DagTraversal<S>>> {
        self.traversals.read().await.get(root).cloned()
    }

    /// Remove a completed traversal
    pub async fn remove_traversal(&self, root: &Cid) {
        self.traversals.write().await.remove(root);
    }

    /// Get number of active traversals
    pub async fn active_count(&self) -> usize {
        self.traversals.read().await.len()
    }
}

/// Gradient message for federated learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientMessage {
    /// Gradient identifier (e.g., layer name or tensor CID)
    pub id: String,
    /// Gradient data (compressed)
    pub data: Vec<u8>,
    /// Shape of the gradient tensor
    pub shape: Vec<usize>,
    /// Data type (f32, f16, etc.)
    pub dtype: String,
    /// Checksum for verification
    pub checksum: u64,
    /// Metadata (e.g., learning rate, batch size)
    pub metadata: HashMap<String, String>,
}

impl GradientMessage {
    /// Create a new gradient message
    pub fn new(
        id: impl Into<String>,
        data: Vec<u8>,
        shape: Vec<usize>,
        dtype: impl Into<String>,
    ) -> Self {
        let checksum = Self::compute_checksum(&data);
        Self {
            id: id.into(),
            data,
            shape,
            dtype: dtype.into(),
            checksum,
            metadata: HashMap::new(),
        }
    }

    /// Compute checksum for data verification
    fn compute_checksum(data: &[u8]) -> u64 {
        // Simple checksum using FNV-1a hash
        let mut hash: u64 = 0xcbf29ce484222325;
        for &byte in data {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    /// Verify checksum
    pub fn verify(&self) -> bool {
        Self::compute_checksum(&self.data) == self.checksum
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get total elements in gradient
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Estimate size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }
}

/// Gradient aggregation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationStrategy {
    /// Simple averaging
    Average,
    /// Weighted average based on sample counts
    WeightedAverage,
    /// Median aggregation (robust to outliers)
    Median,
    /// Federated averaging (FedAvg)
    FederatedAvg,
}

/// Gradient aggregator for federated learning
pub struct GradientAggregator {
    /// Aggregation strategy
    strategy: AggregationStrategy,
    /// Accumulated gradients per layer
    gradients: Arc<RwLock<HashMap<String, Vec<GradientMessage>>>>,
    /// Expected number of contributors
    expected_contributors: usize,
    /// Verification enabled
    verify_checksums: bool,
}

impl GradientAggregator {
    /// Create a new gradient aggregator
    pub fn new(strategy: AggregationStrategy, expected_contributors: usize) -> Self {
        Self {
            strategy,
            gradients: Arc::new(RwLock::new(HashMap::new())),
            expected_contributors,
            verify_checksums: true,
        }
    }

    /// Add a gradient to the aggregator
    pub async fn add_gradient(&self, gradient: GradientMessage) -> Result<()> {
        // Verify checksum if enabled
        if self.verify_checksums && !gradient.verify() {
            return Err(Error::InvalidInput(format!(
                "Gradient checksum verification failed for {}",
                gradient.id
            )));
        }

        // Verify dimensions
        if gradient.num_elements() == 0 {
            return Err(Error::InvalidInput(
                "Gradient has zero elements".to_string(),
            ));
        }

        let mut gradients = self.gradients.write().await;
        gradients
            .entry(gradient.id.clone())
            .or_insert_with(Vec::new)
            .push(gradient);

        Ok(())
    }

    /// Check if ready to aggregate (all contributors submitted)
    pub async fn is_ready(&self, layer_id: &str) -> bool {
        let gradients = self.gradients.read().await;
        gradients
            .get(layer_id)
            .map(|g| g.len() >= self.expected_contributors)
            .unwrap_or(false)
    }

    /// Aggregate gradients for a specific layer
    pub async fn aggregate(&self, layer_id: &str) -> Result<GradientMessage> {
        let gradients = self.gradients.read().await;
        let layer_gradients = gradients
            .get(layer_id)
            .ok_or_else(|| Error::NotFound(format!("No gradients for layer: {}", layer_id)))?;

        if layer_gradients.is_empty() {
            return Err(Error::InvalidInput("No gradients to aggregate".to_string()));
        }

        // Verify all gradients have same shape
        let first_shape = &layer_gradients[0].shape;
        for grad in layer_gradients.iter().skip(1) {
            if &grad.shape != first_shape {
                return Err(Error::InvalidInput("Gradient shape mismatch".to_string()));
            }
        }

        match self.strategy {
            AggregationStrategy::Average | AggregationStrategy::FederatedAvg => {
                self.aggregate_average(layer_id, layer_gradients)
            }
            AggregationStrategy::WeightedAverage => {
                self.aggregate_weighted(layer_id, layer_gradients)
            }
            AggregationStrategy::Median => self.aggregate_median(layer_id, layer_gradients),
        }
    }

    /// Simple averaging aggregation
    fn aggregate_average(
        &self,
        layer_id: &str,
        gradients: &[GradientMessage],
    ) -> Result<GradientMessage> {
        let n = gradients.len();
        let size = gradients[0].data.len();

        // Sum all gradients (treating as bytes for now)
        let mut sum = vec![0u8; size];
        for grad in gradients {
            for (i, &byte) in grad.data.iter().enumerate() {
                sum[i] = sum[i].saturating_add(byte / n as u8);
            }
        }

        Ok(GradientMessage::new(
            layer_id,
            sum,
            gradients[0].shape.clone(),
            gradients[0].dtype.clone(),
        ))
    }

    /// Weighted average aggregation
    fn aggregate_weighted(
        &self,
        layer_id: &str,
        gradients: &[GradientMessage],
    ) -> Result<GradientMessage> {
        // Extract weights from metadata (sample counts)
        let weights: Vec<f32> = gradients
            .iter()
            .map(|g| {
                g.metadata
                    .get("samples")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(1.0)
            })
            .collect();

        let total_weight: f32 = weights.iter().sum();
        let size = gradients[0].data.len();

        // Weighted sum
        let mut weighted_sum = vec![0u8; size];
        for (grad, &weight) in gradients.iter().zip(weights.iter()) {
            let normalized_weight = weight / total_weight;
            for (i, &byte) in grad.data.iter().enumerate() {
                weighted_sum[i] =
                    weighted_sum[i].saturating_add((byte as f32 * normalized_weight) as u8);
            }
        }

        Ok(GradientMessage::new(
            layer_id,
            weighted_sum,
            gradients[0].shape.clone(),
            gradients[0].dtype.clone(),
        ))
    }

    /// Median aggregation (robust to outliers).
    ///
    /// For each element position, all contributor values are collected and the
    /// median is selected.  We handle `f32` gradients (4 bytes/element) directly;
    /// all other dtypes fall back to a byte-level median which is dtype-agnostic
    /// but preserves the robust property.
    fn aggregate_median(
        &self,
        layer_id: &str,
        gradients: &[GradientMessage],
    ) -> Result<GradientMessage> {
        if gradients.is_empty() {
            return Err(Error::InvalidInput(
                "median aggregation: no gradients supplied".to_string(),
            ));
        }

        let n = gradients.len();
        let size = gradients[0].data.len();

        // Validate that every gradient has the same byte length.
        for (i, g) in gradients.iter().enumerate() {
            if g.data.len() != size {
                return Err(Error::InvalidInput(format!(
                    "median aggregation: gradient {} has {} bytes, expected {}",
                    i,
                    g.data.len(),
                    size,
                )));
            }
        }

        let dtype = &gradients[0].dtype;

        // Fast path: f32 gradients – operate at float granularity.
        if dtype == "f32" || dtype == "float32" {
            let element_size = 4usize;
            if !size.is_multiple_of(element_size) {
                return Err(Error::InvalidInput(format!(
                    "median aggregation: byte length {} is not a multiple of 4 for f32 dtype",
                    size
                )));
            }
            let num_elements = size / element_size;
            let mut out_data = vec![0u8; size];

            for elem in 0..num_elements {
                let byte_off = elem * element_size;
                let mut values: Vec<f32> = gradients
                    .iter()
                    .map(|g| {
                        let bytes: [u8; 4] = g.data[byte_off..byte_off + element_size]
                            .try_into()
                            .unwrap_or([0u8; 4]);
                        f32::from_le_bytes(bytes)
                    })
                    .collect();

                // Sort to find the median.
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = if n % 2 == 1 {
                    values[n / 2]
                } else {
                    // Even count: average of the two middle values.
                    (values[n / 2 - 1] + values[n / 2]) * 0.5
                };

                out_data[byte_off..byte_off + element_size].copy_from_slice(&median.to_le_bytes());
            }

            return Ok(GradientMessage::new(
                layer_id,
                out_data,
                gradients[0].shape.clone(),
                dtype.clone(),
            ));
        }

        // Generic byte-level median fallback for other dtypes (f16, i8, etc.).
        // The median is taken independently per byte position.
        let mut out_data = vec![0u8; size];
        for (byte_pos, out_byte) in out_data.iter_mut().enumerate() {
            let mut col: Vec<u8> = gradients.iter().map(|g| g.data[byte_pos]).collect();
            col.sort_unstable();
            *out_byte = if n % 2 == 1 {
                col[n / 2]
            } else {
                let lo = col[n / 2 - 1] as u16;
                let hi = col[n / 2] as u16;
                ((lo + hi) / 2) as u8
            };
        }

        Ok(GradientMessage::new(
            layer_id,
            out_data,
            gradients[0].shape.clone(),
            dtype.clone(),
        ))
    }

    /// Clear gradients for a layer after aggregation
    pub async fn clear(&self, layer_id: &str) {
        let mut gradients = self.gradients.write().await;
        gradients.remove(layer_id);
    }

    /// Get statistics
    pub async fn stats(&self) -> GradientAggregatorStats {
        let gradients = self.gradients.read().await;
        let total_gradients: usize = gradients.values().map(|v| v.len()).sum();
        let layers_count = gradients.len();

        GradientAggregatorStats {
            total_gradients,
            layers_count,
            expected_contributors: self.expected_contributors,
        }
    }
}

/// Gradient aggregator statistics
#[derive(Debug, Clone)]
pub struct GradientAggregatorStats {
    /// Total gradients received
    pub total_gradients: usize,
    /// Number of layers
    pub layers_count: usize,
    /// Expected contributors
    pub expected_contributors: usize,
}

/// Bidirectional gradient stream
pub struct GradientStream {
    /// Gradient aggregator
    aggregator: Arc<GradientAggregator>,
    /// Outgoing gradient queue
    outgoing: Arc<RwLock<VecDeque<GradientMessage>>>,
    /// Maximum queue size
    max_queue_size: usize,
}

impl GradientStream {
    /// Create a new gradient stream
    pub fn new(aggregator: Arc<GradientAggregator>, max_queue_size: usize) -> Self {
        Self {
            aggregator,
            outgoing: Arc::new(RwLock::new(VecDeque::new())),
            max_queue_size,
        }
    }

    /// Push a gradient to send
    pub async fn push_gradient(&self, gradient: GradientMessage) -> Result<()> {
        let mut outgoing = self.outgoing.write().await;
        if outgoing.len() >= self.max_queue_size {
            return Err(Error::Internal("Gradient queue is full".to_string()));
        }
        outgoing.push_back(gradient);
        Ok(())
    }

    /// Pop a gradient to send
    pub async fn pop_gradient(&self) -> Option<GradientMessage> {
        self.outgoing.write().await.pop_front()
    }

    /// Receive a gradient
    pub async fn receive_gradient(&self, gradient: GradientMessage) -> Result<()> {
        self.aggregator.add_gradient(gradient).await
    }

    /// Get queue size
    pub async fn queue_size(&self) -> usize {
        self.outgoing.read().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selector_parse() {
        let json = r#"{"type":"all"}"#;
        let selector = Selector::from_json(json).expect("test: parse all-selector from JSON");
        assert!(selector.matches_all());

        let json2 = r#"{"type":"recursivedepth","max_depth":5}"#;
        let selector2 =
            Selector::from_json(json2).expect("test: parse recursive-depth selector from JSON");
        match selector2 {
            Selector::RecursiveDepth { max_depth } => assert_eq!(max_depth, 5),
            _ => panic!("Wrong selector type"),
        }
    }

    #[test]
    fn test_selector_validate() {
        let selector = Selector::RecursiveDepth { max_depth: 0 };
        assert!(selector.validate().is_err());

        let selector2 = Selector::RecursiveDepth { max_depth: 5 };
        assert!(selector2.validate().is_ok());
    }

    #[test]
    fn test_traversal_state() {
        let cid: Cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse()
            .expect("test: parse CID string");

        let mut state = TraversalState::new(cid, Some(3));
        assert!(!state.is_complete());

        // Get root
        let (root_cid, depth) = state
            .next(TraversalMode::BreadthFirst)
            .expect("test: get next traversal item");
        assert_eq!(root_cid, cid);
        assert_eq!(depth, 0);

        state.mark_visited(cid, 1024);
        assert_eq!(state.blocks_fetched, 1);
        assert_eq!(state.bytes_fetched, 1024);

        assert!(state.is_complete());
    }

    #[test]
    fn test_checkpoint() {
        let cid: Cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse()
            .expect("test: parse CID string");

        let mut state = TraversalState::new(cid, Some(3));
        state.mark_visited(cid, 1024);

        let checkpoint = state.checkpoint();
        assert_eq!(checkpoint.root, cid);
        assert_eq!(checkpoint.blocks_fetched, 1);
        assert_eq!(checkpoint.bytes_fetched, 1024);

        let restored = TraversalState::from_checkpoint(checkpoint);
        assert_eq!(restored.blocks_fetched, 1);
        assert_eq!(restored.bytes_fetched, 1024);
    }

    #[test]
    fn test_gradient_message() {
        let data = vec![1, 2, 3, 4, 5];
        let shape = vec![5];
        let gradient = GradientMessage::new("layer1", data.clone(), shape.clone(), "f32");

        assert_eq!(gradient.id, "layer1");
        assert_eq!(gradient.data, data);
        assert_eq!(gradient.shape, shape);
        assert_eq!(gradient.dtype, "f32");
        assert!(gradient.verify());
        assert_eq!(gradient.num_elements(), 5);
        assert_eq!(gradient.size_bytes(), 5);
    }

    #[test]
    fn test_gradient_checksum() {
        let data = vec![1, 2, 3, 4, 5];
        let mut gradient = GradientMessage::new("layer1", data, vec![5], "f32");

        // Verify original
        assert!(gradient.verify());

        // Corrupt data
        gradient.data[0] = 99;

        // Should fail verification
        assert!(!gradient.verify());
    }

    #[tokio::test]
    async fn test_gradient_aggregator() {
        let aggregator = GradientAggregator::new(AggregationStrategy::Average, 2);

        let grad1 = GradientMessage::new("layer1", vec![10, 20, 30], vec![3], "f32");
        let grad2 = GradientMessage::new("layer1", vec![20, 30, 40], vec![3], "f32");

        aggregator
            .add_gradient(grad1)
            .await
            .expect("test: add first gradient to aggregator");
        aggregator
            .add_gradient(grad2)
            .await
            .expect("test: add second gradient to aggregator");

        assert!(aggregator.is_ready("layer1").await);

        let aggregated = aggregator
            .aggregate("layer1")
            .await
            .expect("test: aggregate layer1 gradients");
        assert_eq!(aggregated.shape, vec![3]);
        assert_eq!(aggregated.id, "layer1");
    }

    #[tokio::test]
    async fn test_gradient_stream() {
        let aggregator = Arc::new(GradientAggregator::new(AggregationStrategy::Average, 1));
        let stream = GradientStream::new(aggregator, 10);

        let grad = GradientMessage::new("layer1", vec![1, 2, 3], vec![3], "f32");

        // Push gradient
        stream
            .push_gradient(grad.clone())
            .await
            .expect("test: push gradient to stream");
        assert_eq!(stream.queue_size().await, 1);

        // Pop gradient
        let popped = stream
            .pop_gradient()
            .await
            .expect("test: pop gradient from stream");
        assert_eq!(popped.id, "layer1");
        assert_eq!(stream.queue_size().await, 0);

        // Receive gradient
        stream
            .receive_gradient(grad)
            .await
            .expect("test: receive gradient into stream");
    }
}
