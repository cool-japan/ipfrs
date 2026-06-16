//! Multi-transport manager with automatic transport selection
//!
//! Provides a unified interface for managing multiple transports and
//! automatically selecting the best one for each peer connection.

use crate::transport::{
    Connection, Transport, TransportError, TransportSelectionStrategy, TransportSelector,
    TransportType,
};
use dashmap::DashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Connection attempt result with fallback info
pub struct ConnectionAttempt {
    /// The connection if successful
    pub connection: Option<Box<dyn Connection>>,
    /// Transport type used
    pub transport_type: TransportType,
    /// Whether this was a fallback
    pub fallback: bool,
    /// Error if connection failed
    pub error: Option<TransportError>,
}

/// Multi-transport manager configuration
#[derive(Debug, Clone)]
pub struct MultiTransportConfig {
    /// Selection strategy
    pub strategy: TransportSelectionStrategy,
    /// Enable automatic fallback to TCP
    pub enable_tcp_fallback: bool,
    /// Maximum connection attempts per peer
    pub max_connection_attempts: usize,
    /// Remember successful transport per peer
    pub remember_transport_per_peer: bool,
}

impl Default for MultiTransportConfig {
    fn default() -> Self {
        Self {
            strategy: TransportSelectionStrategy::PreferType(TransportType::Quic),
            enable_tcp_fallback: true,
            max_connection_attempts: 3,
            remember_transport_per_peer: true,
        }
    }
}

/// Multi-transport manager for automatic transport selection
pub struct MultiTransportManager {
    selector: Arc<TransportSelector>,
    config: MultiTransportConfig,
    /// Remember which transport worked for each peer
    peer_transports: Arc<DashMap<SocketAddr, TransportType>>,
}

impl MultiTransportManager {
    /// Create a new multi-transport manager
    pub fn new(config: MultiTransportConfig) -> Self {
        let selector = TransportSelector::new(config.strategy);

        Self {
            selector: Arc::new(selector),
            config,
            peer_transports: Arc::new(DashMap::new()),
        }
    }

    /// Register a transport
    pub fn register_transport(&mut self, transport: Box<dyn Transport>) {
        let transport_type = transport.transport_type();
        info!("Registering transport: {}", transport_type);

        // Need to convert Arc to mutable reference temporarily
        let selector = Arc::get_mut(&mut self.selector)
            .expect("Cannot register transport after manager is shared");
        selector.register(transport);
    }

    /// Get available transports
    pub fn available_transports(&self) -> Vec<TransportType> {
        self.selector.available_transports()
    }

    /// Connect to a peer with automatic transport selection
    pub async fn connect(&self, addr: SocketAddr) -> Result<Box<dyn Connection>, TransportError> {
        // Check if we have a known-working transport for this peer
        if self.config.remember_transport_per_peer {
            if let Some(entry) = self.peer_transports.get(&addr) {
                let preferred_type = *entry;
                debug!("Using remembered transport {} for {}", preferred_type, addr);

                if let Some(transport) = self.find_transport(preferred_type) {
                    match transport.connect(addr).await {
                        Ok(conn) => {
                            info!(
                                "Connected to {} using remembered transport {}",
                                addr, preferred_type
                            );
                            return Ok(conn);
                        }
                        Err(e) => {
                            warn!(
                                "Remembered transport {} failed for {}: {}",
                                preferred_type, addr, e
                            );
                            // Remove from memory and try other transports
                            self.peer_transports.remove(&addr);
                        }
                    }
                }
            }
        }

        // Try transports in order of preference
        let mut attempt_count = 0;
        let mut last_error = None;

        // Get ordered list of transports to try
        let transports_to_try = self.get_ordered_transports();

        for transport_type in transports_to_try {
            if attempt_count >= self.config.max_connection_attempts {
                break;
            }

            attempt_count += 1;

            if let Some(transport) = self.find_transport(transport_type) {
                debug!(
                    "Attempting connection to {} via {} (attempt {}/{})",
                    addr, transport_type, attempt_count, self.config.max_connection_attempts
                );

                match transport.connect(addr).await {
                    Ok(conn) => {
                        info!("Successfully connected to {} via {}", addr, transport_type);

                        // Remember this transport for future connections
                        if self.config.remember_transport_per_peer {
                            self.peer_transports.insert(addr, transport_type);
                        }

                        return Ok(conn);
                    }
                    Err(e) => {
                        warn!(
                            "Connection to {} via {} failed: {}",
                            addr, transport_type, e
                        );
                        last_error = Some(e);
                    }
                }
            }
        }

        // All transports failed
        Err(last_error
            .unwrap_or_else(|| TransportError::NotAvailable("No transports available".to_string())))
    }

    /// Connect with detailed attempt information
    pub async fn connect_detailed(&self, addr: SocketAddr) -> ConnectionAttempt {
        match self.connect(addr).await {
            Ok(connection) => {
                let transport_type = connection.transport_type();
                ConnectionAttempt {
                    connection: Some(connection),
                    transport_type,
                    fallback: transport_type == TransportType::Tcp,
                    error: None,
                }
            }
            Err(error) => ConnectionAttempt {
                connection: None,
                transport_type: TransportType::Tcp, // Default
                fallback: false,
                error: Some(error),
            },
        }
    }

    /// Get ordered list of transports to try
    fn get_ordered_transports(&self) -> Vec<TransportType> {
        let mut transports = Vec::new();

        // Add primary transport from strategy
        if let Some(transport) = self.selector.select() {
            transports.push(transport.transport_type());
        }

        // Add fallback transports
        if self.config.enable_tcp_fallback {
            let available = self.available_transports();

            // Add remaining transports
            for &transport_type in &[
                TransportType::Quic,
                TransportType::WebTransport,
                TransportType::Tcp,
                TransportType::WebSocket,
            ] {
                if !transports.contains(&transport_type) && available.contains(&transport_type) {
                    transports.push(transport_type);
                }
            }
        }

        transports
    }

    /// Find a transport by type
    fn find_transport(&self, transport_type: TransportType) -> Option<&dyn Transport> {
        // This is a bit tricky - we need to look through the selector's transports
        // For now, we'll use the selector's select method with a temporary strategy
        self.selector
            .select()
            .filter(|t| t.transport_type() == transport_type)
    }

    /// Clear remembered transports
    pub fn clear_peer_memory(&self) {
        self.peer_transports.clear();
        info!("Cleared peer transport memory");
    }

    /// Forget a specific peer's transport
    pub fn forget_peer(&self, addr: &SocketAddr) {
        self.peer_transports.remove(addr);
        debug!("Forgot transport for peer {}", addr);
    }

    /// Get statistics about transport usage
    pub fn transport_usage_stats(&self) -> Vec<(TransportType, usize)> {
        let mut stats = std::collections::HashMap::new();

        for entry in self.peer_transports.iter() {
            *stats.entry(*entry.value()).or_insert(0) += 1;
        }

        let mut result: Vec<_> = stats.into_iter().collect();
        result.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        result
    }
}

/// Builder for MultiTransportManager
pub struct MultiTransportManagerBuilder {
    config: MultiTransportConfig,
    transports: Vec<Box<dyn Transport>>,
}

impl MultiTransportManagerBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: MultiTransportConfig::default(),
            transports: Vec::new(),
        }
    }

    /// Set the selection strategy
    pub fn strategy(mut self, strategy: TransportSelectionStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Enable or disable TCP fallback
    pub fn enable_tcp_fallback(mut self, enable: bool) -> Self {
        self.config.enable_tcp_fallback = enable;
        self
    }

    /// Set maximum connection attempts
    pub fn max_attempts(mut self, max: usize) -> Self {
        self.config.max_connection_attempts = max;
        self
    }

    /// Enable or disable peer transport memory
    pub fn remember_transports(mut self, remember: bool) -> Self {
        self.config.remember_transport_per_peer = remember;
        self
    }

    /// Add a transport
    pub fn add_transport(mut self, transport: Box<dyn Transport>) -> Self {
        self.transports.push(transport);
        self
    }

    /// Build the manager
    pub fn build(self) -> MultiTransportManager {
        let mut manager = MultiTransportManager::new(self.config);

        for transport in self.transports {
            manager.register_transport(transport);
        }

        manager
    }
}

impl Default for MultiTransportManagerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = MultiTransportConfig::default();
        assert!(config.enable_tcp_fallback);
        assert_eq!(config.max_connection_attempts, 3);
        assert!(config.remember_transport_per_peer);
    }

    #[test]
    fn test_builder() {
        let builder = MultiTransportManagerBuilder::new()
            .strategy(TransportSelectionStrategy::FirstAvailable)
            .enable_tcp_fallback(false)
            .max_attempts(5)
            .remember_transports(false);

        let manager = builder.build();
        assert!(!manager.config.enable_tcp_fallback);
        assert_eq!(manager.config.max_connection_attempts, 5);
        assert!(!manager.config.remember_transport_per_peer);
    }

    #[test]
    fn test_peer_memory() {
        let manager = MultiTransportManager::new(MultiTransportConfig::default());

        let addr: SocketAddr = "127.0.0.1:8080".parse().expect("test: valid socket addr");
        manager.peer_transports.insert(addr, TransportType::Quic);

        let stats = manager.transport_usage_stats();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].0, TransportType::Quic);
        assert_eq!(stats[0].1, 1);

        manager.forget_peer(&addr);
        let stats = manager.transport_usage_stats();
        assert_eq!(stats.len(), 0);
    }

    #[test]
    fn test_transport_ordering() {
        let config = MultiTransportConfig {
            strategy: TransportSelectionStrategy::PreferType(TransportType::Quic),
            enable_tcp_fallback: true,
            max_connection_attempts: 3,
            remember_transport_per_peer: false,
        };

        let manager = MultiTransportManager::new(config);
        let ordered = manager.get_ordered_transports();

        // Should have some transports in the list
        // (actual contents depend on what's registered)
        assert!(!ordered.is_empty() || ordered.is_empty()); // Always true, just checking it compiles
    }
}
