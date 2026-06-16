//! Protocol handler registry and version negotiation
//!
//! Provides:
//! - Protocol handler registration and lifecycle
//! - Protocol version negotiation
//! - Protocol capability advertisement
//! - Dynamic handler loading

use ipfrs_core::error::{Error, Result};
use std::collections::HashMap;
use std::sync::Arc;

/// Protocol version using semantic versioning
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProtocolVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

impl ProtocolVersion {
    /// Create a new protocol version
    pub fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parse version from string (e.g., "1.2.3")
    pub fn parse(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return Err(Error::Network(format!("Invalid version string: {}", s)));
        }

        let major = parts[0]
            .parse()
            .map_err(|e| Error::Network(format!("Invalid major version: {}", e)))?;
        let minor = parts[1]
            .parse()
            .map_err(|e| Error::Network(format!("Invalid minor version: {}", e)))?;
        let patch = parts[2]
            .parse()
            .map_err(|e| Error::Network(format!("Invalid patch version: {}", e)))?;

        Ok(Self::new(major, minor, patch))
    }

    /// Check if this version is compatible with another
    /// Compatible if major versions match and this minor >= other minor
    pub fn is_compatible_with(&self, other: &ProtocolVersion) -> bool {
        self.major == other.major && self.minor >= other.minor
    }
}

impl std::fmt::Display for ProtocolVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Protocol identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProtocolId {
    /// Protocol name
    pub name: String,
    /// Protocol version
    pub version: ProtocolVersion,
}

impl ProtocolId {
    /// Create a new protocol ID
    pub fn new(name: String, version: ProtocolVersion) -> Self {
        Self { name, version }
    }

    /// Get the full protocol string (e.g., "/ipfrs/tensorswap/1.0.0")
    pub fn to_protocol_string(&self) -> String {
        format!("/ipfrs/{}/{}", self.name, self.version)
    }

    /// Parse protocol ID from string
    pub fn parse(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.trim_matches('/').split('/').collect();
        if parts.len() != 3 || parts[0] != "ipfrs" {
            return Err(Error::Network(format!("Invalid protocol string: {}", s)));
        }

        let name = parts[1].to_string();
        let version = ProtocolVersion::parse(parts[2])?;

        Ok(Self::new(name, version))
    }
}

impl std::fmt::Display for ProtocolId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_protocol_string())
    }
}

/// Protocol capabilities
#[derive(Debug, Clone)]
pub struct ProtocolCapabilities {
    /// Supported features
    pub features: Vec<String>,
    /// Maximum message size
    pub max_message_size: usize,
    /// Whether protocol supports streaming
    pub supports_streaming: bool,
}

impl Default for ProtocolCapabilities {
    fn default() -> Self {
        Self {
            features: Vec::new(),
            max_message_size: 1024 * 1024, // 1MB default
            supports_streaming: false,
        }
    }
}

/// Type alias for boxed protocol handler
type BoxedProtocolHandler = Arc<parking_lot::RwLock<Box<dyn ProtocolHandler>>>;

/// Protocol handler trait
pub trait ProtocolHandler: Send + Sync {
    /// Get protocol ID
    fn protocol_id(&self) -> ProtocolId;

    /// Get protocol capabilities
    fn capabilities(&self) -> ProtocolCapabilities {
        ProtocolCapabilities::default()
    }

    /// Handle incoming protocol request
    fn handle_request(&mut self, request: &[u8]) -> Result<Vec<u8>>;

    /// Initialize the handler
    fn initialize(&mut self) -> Result<()> {
        Ok(())
    }

    /// Shutdown the handler
    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Protocol handler registry
pub struct ProtocolRegistry {
    /// Registered handlers by protocol ID
    handlers: parking_lot::RwLock<HashMap<ProtocolId, BoxedProtocolHandler>>,
    /// Protocol aliases (name -> list of versions)
    aliases: parking_lot::RwLock<HashMap<String, Vec<ProtocolVersion>>>,
}

impl ProtocolRegistry {
    /// Create a new protocol registry
    pub fn new() -> Self {
        Self {
            handlers: parking_lot::RwLock::new(HashMap::new()),
            aliases: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// Register a protocol handler
    pub fn register(&self, handler: Box<dyn ProtocolHandler>) -> Result<()> {
        let protocol_id = handler.protocol_id();
        let mut handlers = self.handlers.write();

        if handlers.contains_key(&protocol_id) {
            return Err(Error::Network(format!(
                "Protocol already registered: {}",
                protocol_id
            )));
        }

        // Add to aliases
        let mut aliases = self.aliases.write();
        aliases
            .entry(protocol_id.name.clone())
            .or_default()
            .push(protocol_id.version.clone());

        handlers.insert(protocol_id, Arc::new(parking_lot::RwLock::new(handler)));

        Ok(())
    }

    /// Unregister a protocol handler
    pub fn unregister(&self, protocol_id: &ProtocolId) -> Result<()> {
        let mut handlers = self.handlers.write();

        if let Some(handler) = handlers.remove(protocol_id) {
            // Shutdown the handler
            let mut handler = handler.write();
            handler.shutdown()?;

            // Remove from aliases
            let mut aliases = self.aliases.write();
            if let Some(versions) = aliases.get_mut(&protocol_id.name) {
                versions.retain(|v| v != &protocol_id.version);
                if versions.is_empty() {
                    aliases.remove(&protocol_id.name);
                }
            }

            Ok(())
        } else {
            Err(Error::Network(format!(
                "Protocol not registered: {}",
                protocol_id
            )))
        }
    }

    /// Get a protocol handler
    pub fn get(&self, protocol_id: &ProtocolId) -> Option<BoxedProtocolHandler> {
        let handlers = self.handlers.read();
        handlers.get(protocol_id).cloned()
    }

    /// Find compatible protocol version
    pub fn find_compatible(&self, name: &str, min_version: &ProtocolVersion) -> Option<ProtocolId> {
        let aliases = self.aliases.read();
        if let Some(versions) = aliases.get(name) {
            // Find the highest compatible version
            let mut compatible_versions: Vec<_> = versions
                .iter()
                .filter(|v| v.is_compatible_with(min_version))
                .collect();

            compatible_versions.sort_by(|a, b| b.cmp(a)); // Sort descending

            if let Some(version) = compatible_versions.first() {
                return Some(ProtocolId::new(name.to_string(), (*version).clone()));
            }
        }
        None
    }

    /// Get all registered protocol IDs
    pub fn list_protocols(&self) -> Vec<ProtocolId> {
        let handlers = self.handlers.read();
        handlers.keys().cloned().collect()
    }

    /// Handle a request with the appropriate protocol handler
    pub fn handle_request(&self, protocol_id: &ProtocolId, request: &[u8]) -> Result<Vec<u8>> {
        if let Some(handler) = self.get(protocol_id) {
            let mut handler = handler.write();
            handler.handle_request(request)
        } else {
            Err(Error::Network(format!(
                "No handler registered for protocol: {}",
                protocol_id
            )))
        }
    }

    /// Get protocol capabilities
    pub fn get_capabilities(&self, protocol_id: &ProtocolId) -> Option<ProtocolCapabilities> {
        if let Some(handler) = self.get(protocol_id) {
            let handler = handler.read();
            Some(handler.capabilities())
        } else {
            None
        }
    }

    /// Shutdown all handlers
    pub fn shutdown_all(&self) -> Result<()> {
        let handlers = self.handlers.write();
        for handler in handlers.values() {
            let mut handler = handler.write();
            handler.shutdown()?;
        }
        Ok(())
    }
}

impl Default for ProtocolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock protocol handler for testing
    struct MockProtocolHandler {
        id: ProtocolId,
    }

    impl MockProtocolHandler {
        fn new(name: &str, version: ProtocolVersion) -> Self {
            Self {
                id: ProtocolId::new(name.to_string(), version),
            }
        }
    }

    impl ProtocolHandler for MockProtocolHandler {
        fn protocol_id(&self) -> ProtocolId {
            self.id.clone()
        }

        fn handle_request(&mut self, request: &[u8]) -> Result<Vec<u8>> {
            Ok(request.to_vec())
        }
    }

    #[test]
    fn test_protocol_version_creation() {
        let version = ProtocolVersion::new(1, 2, 3);
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 2);
        assert_eq!(version.patch, 3);
    }

    #[test]
    fn test_protocol_version_parse() {
        let version = ProtocolVersion::parse("1.2.3")
            .expect("test: valid version string '1.2.3' should parse");
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 2);
        assert_eq!(version.patch, 3);

        assert!(ProtocolVersion::parse("invalid").is_err());
        assert!(ProtocolVersion::parse("1.2").is_err());
    }

    #[test]
    fn test_protocol_version_compatibility() {
        let v1_0_0 = ProtocolVersion::new(1, 0, 0);
        let v1_1_0 = ProtocolVersion::new(1, 1, 0);
        let v1_2_0 = ProtocolVersion::new(1, 2, 0);
        let v2_0_0 = ProtocolVersion::new(2, 0, 0);

        // Same major, higher or equal minor is compatible
        assert!(v1_2_0.is_compatible_with(&v1_0_0));
        assert!(v1_1_0.is_compatible_with(&v1_0_0));
        assert!(v1_0_0.is_compatible_with(&v1_0_0));

        // Lower minor is not compatible
        assert!(!v1_0_0.is_compatible_with(&v1_1_0));

        // Different major is not compatible
        assert!(!v2_0_0.is_compatible_with(&v1_0_0));
        assert!(!v1_0_0.is_compatible_with(&v2_0_0));
    }

    #[test]
    fn test_protocol_version_display() {
        let version = ProtocolVersion::new(1, 2, 3);
        assert_eq!(format!("{}", version), "1.2.3");
    }

    #[test]
    fn test_protocol_id_creation() {
        let version = ProtocolVersion::new(1, 0, 0);
        let id = ProtocolId::new("test".to_string(), version);
        assert_eq!(id.name, "test");
        assert_eq!(id.version.major, 1);
    }

    #[test]
    fn test_protocol_id_to_string() {
        let version = ProtocolVersion::new(1, 0, 0);
        let id = ProtocolId::new("tensorswap".to_string(), version);
        assert_eq!(id.to_protocol_string(), "/ipfrs/tensorswap/1.0.0");
    }

    #[test]
    fn test_protocol_id_parse() {
        let id = ProtocolId::parse("/ipfrs/tensorswap/1.0.0")
            .expect("test: valid protocol string '/ipfrs/tensorswap/1.0.0' should parse");
        assert_eq!(id.name, "tensorswap");
        assert_eq!(id.version.major, 1);

        assert!(ProtocolId::parse("/invalid/tensorswap/1.0.0").is_err());
        assert!(ProtocolId::parse("/ipfrs/tensorswap").is_err());
    }

    #[test]
    fn test_registry_creation() {
        let registry = ProtocolRegistry::new();
        assert_eq!(registry.list_protocols().len(), 0);
    }

    #[test]
    fn test_register_handler() {
        let registry = ProtocolRegistry::new();
        let handler = Box::new(MockProtocolHandler::new(
            "test",
            ProtocolVersion::new(1, 0, 0),
        ));

        registry
            .register(handler)
            .expect("test: registering a fresh handler should succeed");
        assert_eq!(registry.list_protocols().len(), 1);
    }

    #[test]
    fn test_register_duplicate() {
        let registry = ProtocolRegistry::new();
        let handler1 = Box::new(MockProtocolHandler::new(
            "test",
            ProtocolVersion::new(1, 0, 0),
        ));
        let handler2 = Box::new(MockProtocolHandler::new(
            "test",
            ProtocolVersion::new(1, 0, 0),
        ));

        registry
            .register(handler1)
            .expect("test: registering handler1 should succeed");
        assert!(registry.register(handler2).is_err());
    }

    #[test]
    fn test_get_handler() {
        let registry = ProtocolRegistry::new();
        let version = ProtocolVersion::new(1, 0, 0);
        let protocol_id = ProtocolId::new("test".to_string(), version.clone());

        let handler = Box::new(MockProtocolHandler::new("test", version));
        registry
            .register(handler)
            .expect("test: registering handler should succeed");

        let retrieved = registry.get(&protocol_id);
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_find_compatible() {
        let registry = ProtocolRegistry::new();

        let handler1 = Box::new(MockProtocolHandler::new(
            "test",
            ProtocolVersion::new(1, 0, 0),
        ));
        let handler2 = Box::new(MockProtocolHandler::new(
            "test",
            ProtocolVersion::new(1, 1, 0),
        ));
        let handler3 = Box::new(MockProtocolHandler::new(
            "test",
            ProtocolVersion::new(1, 2, 0),
        ));

        registry
            .register(handler1)
            .expect("test: registering handler1 should succeed");
        registry
            .register(handler2)
            .expect("test: registering handler2 should succeed");
        registry
            .register(handler3)
            .expect("test: registering handler3 should succeed");

        // Should find the highest compatible version
        let min_version = ProtocolVersion::new(1, 0, 0);
        let compatible = registry.find_compatible("test", &min_version);

        assert!(compatible.is_some());
        let compatible = compatible
            .expect("test: find_compatible should return a result for a registered version");
        assert_eq!(compatible.version.major, 1);
        assert_eq!(compatible.version.minor, 2);
    }

    #[test]
    fn test_unregister_handler() {
        let registry = ProtocolRegistry::new();
        let version = ProtocolVersion::new(1, 0, 0);
        let protocol_id = ProtocolId::new("test".to_string(), version.clone());

        let handler = Box::new(MockProtocolHandler::new("test", version));
        registry
            .register(handler)
            .expect("test: registering handler should succeed");

        registry
            .unregister(&protocol_id)
            .expect("test: unregistering a registered protocol should succeed");
        assert_eq!(registry.list_protocols().len(), 0);
    }

    #[test]
    fn test_handle_request() {
        let registry = ProtocolRegistry::new();
        let version = ProtocolVersion::new(1, 0, 0);
        let protocol_id = ProtocolId::new("test".to_string(), version.clone());

        let handler = Box::new(MockProtocolHandler::new("test", version));
        registry
            .register(handler)
            .expect("test: registering handler should succeed");

        let request = b"test request";
        let response = registry
            .handle_request(&protocol_id, request)
            .expect("test: handle_request should return response for registered protocol");

        assert_eq!(response, request);
    }

    #[test]
    fn test_protocol_capabilities_default() {
        let caps = ProtocolCapabilities::default();
        assert_eq!(caps.max_message_size, 1024 * 1024);
        assert!(!caps.supports_streaming);
    }
}
