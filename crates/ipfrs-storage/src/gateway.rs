//! IPFS gateway fallback for fetching blocks from public gateways
//!
//! Provides a hybrid local/remote storage model where blocks not found locally
//! can be fetched from public IPFS gateways and cached for future use.

use crate::traits::BlockStore;
use async_trait::async_trait;
use ipfrs_core::{Block, Cid, Error, Result};
use reqwest::Client as HttpClient;
use std::sync::Arc;
use std::time::Duration;
use url::Url;

/// Default IPFS gateways
const DEFAULT_GATEWAYS: &[&str] = &[
    "https://ipfs.io",
    "https://dweb.link",
    "https://cloudflare-ipfs.com",
    "https://gateway.pinata.cloud",
];

/// Gateway block store configuration
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    /// List of gateway URLs to try
    pub gateways: Vec<String>,
    /// Request timeout in seconds
    pub timeout: Duration,
    /// Maximum number of retries per gateway
    pub max_retries: usize,
    /// Whether to cache fetched blocks locally
    pub cache_fetched: bool,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            gateways: DEFAULT_GATEWAYS.iter().map(|s| s.to_string()).collect(),
            timeout: Duration::from_secs(30),
            max_retries: 2,
            cache_fetched: true,
        }
    }
}

impl GatewayConfig {
    /// Create a new gateway configuration with default gateways
    pub fn new() -> Self {
        Self::default()
    }

    /// Set custom gateways
    pub fn with_gateways(mut self, gateways: Vec<String>) -> Self {
        self.gateways = gateways;
        self
    }

    /// Add a gateway to the list
    pub fn add_gateway(mut self, gateway: String) -> Self {
        self.gateways.push(gateway);
        self
    }

    /// Set request timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set maximum retries per gateway
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Set whether to cache fetched blocks
    pub fn with_cache_fetched(mut self, cache_fetched: bool) -> Self {
        self.cache_fetched = cache_fetched;
        self
    }
}

/// Gateway block store that fetches from IPFS gateways
pub struct GatewayBlockStore {
    http_client: Arc<HttpClient>,
    config: GatewayConfig,
}

impl GatewayBlockStore {
    /// Create a new gateway block store
    pub fn new(config: GatewayConfig) -> Result<Self> {
        let http_client = HttpClient::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| Error::Storage(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            http_client: Arc::new(http_client),
            config,
        })
    }

    /// Fetch a block from a specific gateway
    async fn fetch_from_gateway(&self, gateway: &str, cid: &Cid) -> Result<Option<Block>> {
        // Build gateway URL: <gateway>/ipfs/<cid>
        let url = format!("{}/ipfs/{}", gateway.trim_end_matches('/'), cid);

        let url =
            Url::parse(&url).map_err(|e| Error::Storage(format!("Invalid gateway URL: {e}")))?;

        tracing::debug!("Fetching block {} from gateway {}", cid, gateway);

        match self.http_client.get(url).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    let data = response.bytes().await.map_err(|e| {
                        Error::Storage(format!("Failed to read response body: {e}"))
                    })?;

                    tracing::debug!(
                        "Successfully fetched block {} ({} bytes) from {}",
                        cid,
                        data.len(),
                        gateway
                    );

                    Ok(Some(Block::from_parts(*cid, data)))
                } else if response.status().as_u16() == 404 {
                    tracing::debug!("Block {} not found on gateway {}", cid, gateway);
                    Ok(None)
                } else {
                    Err(Error::Storage(format!(
                        "Gateway {} returned error: {}",
                        gateway,
                        response.status()
                    )))
                }
            }
            Err(e) => Err(Error::Storage(format!(
                "Failed to fetch from gateway {gateway}: {e}"
            ))),
        }
    }

    /// Fetch a block from any available gateway
    async fn fetch_from_any_gateway(&self, cid: &Cid) -> Result<Option<Block>> {
        for gateway in &self.config.gateways {
            for attempt in 0..self.config.max_retries {
                match self.fetch_from_gateway(gateway, cid).await {
                    Ok(Some(block)) => return Ok(Some(block)),
                    Ok(None) => break, // 404, try next gateway
                    Err(e) => {
                        if attempt < self.config.max_retries - 1 {
                            tracing::warn!(
                                "Attempt {} failed for gateway {}: {}. Retrying...",
                                attempt + 1,
                                gateway,
                                e
                            );
                            tokio::time::sleep(Duration::from_millis(100 * (attempt as u64 + 1)))
                                .await;
                        } else {
                            tracing::warn!(
                                "All {} attempts failed for gateway {}: {}",
                                self.config.max_retries,
                                gateway,
                                e
                            );
                        }
                    }
                }
            }
        }

        Ok(None)
    }

    /// Get configuration
    pub fn config(&self) -> &GatewayConfig {
        &self.config
    }
}

#[async_trait]
impl BlockStore for GatewayBlockStore {
    /// Store a block (not supported for gateway-only store)
    async fn put(&self, _block: &Block) -> Result<()> {
        Err(Error::Storage(
            "Put operation not supported for gateway-only block store".to_string(),
        ))
    }

    /// Retrieve a block by fetching from gateways
    async fn get(&self, cid: &Cid) -> Result<Option<Block>> {
        self.fetch_from_any_gateway(cid).await
    }

    /// Check if a block exists (attempts to fetch)
    async fn has(&self, cid: &Cid) -> Result<bool> {
        Ok(self.get(cid).await?.is_some())
    }

    /// Delete a block (not supported for gateway-only store)
    async fn delete(&self, _cid: &Cid) -> Result<()> {
        Err(Error::Storage(
            "Delete operation not supported for gateway-only block store".to_string(),
        ))
    }

    /// Get the number of blocks (always 0 for gateway store)
    fn len(&self) -> usize {
        0
    }

    /// Check if the store is empty (always true for gateway store)
    fn is_empty(&self) -> bool {
        true
    }

    /// List CIDs (not supported for gateway store)
    fn list_cids(&self) -> Result<Vec<Cid>> {
        Ok(Vec::new())
    }

    /// Retrieve multiple blocks
    async fn get_many(&self, cids: &[Cid]) -> Result<Vec<Option<Block>>> {
        let mut results = Vec::with_capacity(cids.len());
        for cid in cids {
            results.push(self.get(cid).await?);
        }
        Ok(results)
    }
}

/// Hybrid block store that combines a local store with gateway fallback
pub struct HybridBlockStore<T: BlockStore> {
    local: Arc<T>,
    gateway: Arc<GatewayBlockStore>,
}

impl<T: BlockStore> HybridBlockStore<T> {
    /// Create a new hybrid block store
    pub fn new(local: T, gateway_config: GatewayConfig) -> Result<Self> {
        let gateway = GatewayBlockStore::new(gateway_config)?;

        Ok(Self {
            local: Arc::new(local),
            gateway: Arc::new(gateway),
        })
    }

    /// Get reference to local store
    pub fn local(&self) -> &Arc<T> {
        &self.local
    }

    /// Get reference to gateway store
    pub fn gateway(&self) -> &Arc<GatewayBlockStore> {
        &self.gateway
    }
}

#[async_trait]
impl<T: BlockStore> BlockStore for HybridBlockStore<T> {
    /// Store a block in the local store
    async fn put(&self, block: &Block) -> Result<()> {
        self.local.put(block).await
    }

    /// Retrieve a block from local store, fallback to gateway
    async fn get(&self, cid: &Cid) -> Result<Option<Block>> {
        // Try local first
        if let Some(block) = self.local.get(cid).await? {
            return Ok(Some(block));
        }

        // Fallback to gateway
        tracing::debug!("Block {} not found locally, trying gateways", cid);
        if let Some(block) = self.gateway.get(cid).await? {
            // Cache the fetched block locally if configured
            if self.gateway.config.cache_fetched {
                tracing::debug!("Caching fetched block {} locally", cid);
                if let Err(e) = self.local.put(&block).await {
                    tracing::warn!("Failed to cache fetched block locally: {}", e);
                }
            }
            return Ok(Some(block));
        }

        Ok(None)
    }

    /// Check if a block exists in local store or gateway
    async fn has(&self, cid: &Cid) -> Result<bool> {
        // Check local first
        if self.local.has(cid).await? {
            return Ok(true);
        }

        // Check gateway (this will fetch the block)
        Ok(self.gateway.has(cid).await?)
    }

    /// Delete a block from local store
    async fn delete(&self, cid: &Cid) -> Result<()> {
        self.local.delete(cid).await
    }

    /// Get number of blocks in local store
    fn len(&self) -> usize {
        self.local.len()
    }

    /// Check if local store is empty
    fn is_empty(&self) -> bool {
        self.local.is_empty()
    }

    /// List CIDs in local store
    fn list_cids(&self) -> Result<Vec<Cid>> {
        self.local.list_cids()
    }

    /// Store multiple blocks in local store
    async fn put_many(&self, blocks: &[Block]) -> Result<()> {
        self.local.put_many(blocks).await
    }

    /// Retrieve multiple blocks from local store with gateway fallback
    async fn get_many(&self, cids: &[Cid]) -> Result<Vec<Option<Block>>> {
        let mut results = Vec::with_capacity(cids.len());
        let mut missing_cids = Vec::new();
        let mut missing_indices = Vec::new();

        // Try local first
        let local_results = self.local.get_many(cids).await?;

        for (i, result) in local_results.into_iter().enumerate() {
            if let Some(block) = result {
                results.push(Some(block));
            } else {
                results.push(None);
                missing_cids.push(cids[i]);
                missing_indices.push(i);
            }
        }

        // Fetch missing blocks from gateway
        if !missing_cids.is_empty() {
            let gateway_results = self.gateway.get_many(&missing_cids).await?;

            for (idx, block_opt) in gateway_results.into_iter().enumerate() {
                let result_idx = missing_indices[idx];

                if let Some(block) = block_opt {
                    // Cache locally if configured
                    if self.gateway.config.cache_fetched {
                        let _ = self.local.put(&block).await;
                    }
                    results[result_idx] = Some(block);
                }
            }
        }

        Ok(results)
    }

    /// Check if multiple blocks exist
    async fn has_many(&self, cids: &[Cid]) -> Result<Vec<bool>> {
        let mut results = self.local.has_many(cids).await?;

        // Check missing blocks in gateway
        for i in 0..results.len() {
            if !results[i] {
                results[i] = self.gateway.has(&cids[i]).await?;
            }
        }

        Ok(results)
    }

    /// Delete multiple blocks from local store
    async fn delete_many(&self, cids: &[Cid]) -> Result<()> {
        self.local.delete_many(cids).await
    }

    /// Flush local store
    async fn flush(&self) -> Result<()> {
        self.local.flush().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gateway_config() {
        let config = GatewayConfig::new()
            .with_timeout(Duration::from_secs(10))
            .with_max_retries(3)
            .add_gateway("https://custom.gateway.example".to_string());

        assert_eq!(config.timeout, Duration::from_secs(10));
        assert_eq!(config.max_retries, 3);
        assert!(config
            .gateways
            .contains(&"https://custom.gateway.example".to_string()));
    }

    #[test]
    fn test_gateway_config_default() {
        let config = GatewayConfig::default();
        assert!(!config.gateways.is_empty());
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert_eq!(config.max_retries, 2);
        assert!(config.cache_fetched);
    }
}
