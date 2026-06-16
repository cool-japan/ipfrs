//! Network interface monitoring and switch detection
//!
//! This module provides functionality to detect network interface changes,
//! which is crucial for mobile devices that switch between WiFi and cellular
//! networks, or for any device where network connectivity may change.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::{debug, info, warn};

/// Network interface information
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetworkInterface {
    /// Interface name (e.g., "eth0", "wlan0", "cellular0")
    pub name: String,
    /// IP addresses assigned to this interface
    pub addresses: Vec<IpAddr>,
    /// Interface type
    pub interface_type: InterfaceType,
    /// Whether the interface is currently active
    pub is_active: bool,
}

/// Type of network interface
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterfaceType {
    /// Wired Ethernet connection
    Ethernet,
    /// WiFi connection
    WiFi,
    /// Cellular/mobile data connection
    Cellular,
    /// Loopback interface
    Loopback,
    /// Unknown or other type
    Other,
}

impl InterfaceType {
    /// Determine interface type from interface name
    pub fn from_name(name: &str) -> Self {
        let lower = name.to_lowercase();
        if lower.contains("eth") || lower.contains("en") && lower.contains("p") {
            InterfaceType::Ethernet
        } else if lower.contains("wlan") || lower.contains("wifi") || lower.contains("wl") {
            InterfaceType::WiFi
        } else if lower.contains("cellular") || lower.contains("wwan") || lower.contains("ppp") {
            InterfaceType::Cellular
        } else if lower.contains("lo") {
            InterfaceType::Loopback
        } else {
            InterfaceType::Other
        }
    }

    /// Get priority for this interface type (higher = preferred)
    pub fn priority(&self) -> u8 {
        match self {
            InterfaceType::Ethernet => 3,
            InterfaceType::WiFi => 2,
            InterfaceType::Cellular => 1,
            InterfaceType::Loopback => 0,
            InterfaceType::Other => 0,
        }
    }
}

/// Network change event
#[derive(Debug, Clone)]
pub enum NetworkChange {
    /// New interface became available
    InterfaceAdded(NetworkInterface),
    /// Interface was removed
    InterfaceRemoved(NetworkInterface),
    /// Primary interface changed (e.g., switched from WiFi to Cellular)
    PrimaryInterfaceChanged {
        old: Option<NetworkInterface>,
        new: NetworkInterface,
    },
    /// IP address changed on an interface
    AddressChanged {
        interface: String,
        old_addresses: Vec<IpAddr>,
        new_addresses: Vec<IpAddr>,
    },
    /// Interface became active
    InterfaceUp(NetworkInterface),
    /// Interface became inactive
    InterfaceDown(NetworkInterface),
}

/// Network monitor configuration
#[derive(Debug, Clone)]
pub struct NetworkMonitorConfig {
    /// Polling interval for checking network changes
    pub poll_interval: Duration,
    /// Minimum time between network change notifications (debouncing)
    pub debounce_duration: Duration,
    /// Whether to monitor loopback interfaces
    pub monitor_loopback: bool,
}

impl Default for NetworkMonitorConfig {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_secs(5),
            debounce_duration: Duration::from_millis(500),
            monitor_loopback: false,
        }
    }
}

impl NetworkMonitorConfig {
    /// Mobile device configuration with frequent polling
    pub fn mobile() -> Self {
        Self {
            poll_interval: Duration::from_secs(2), // Poll more frequently
            debounce_duration: Duration::from_millis(1000), // Longer debounce
            monitor_loopback: false,
        }
    }

    /// Server configuration with less frequent polling
    pub fn server() -> Self {
        Self {
            poll_interval: Duration::from_secs(30), // Poll less frequently
            debounce_duration: Duration::from_millis(100),
            monitor_loopback: false,
        }
    }
}

/// Network monitor for detecting interface changes
pub struct NetworkMonitor {
    config: NetworkMonitorConfig,
    /// Current state of network interfaces
    interfaces: Arc<RwLock<HashMap<String, NetworkInterface>>>,
    /// Primary (preferred) interface
    primary_interface: Arc<RwLock<Option<NetworkInterface>>>,
    /// Last time a change was detected
    last_change: Arc<RwLock<Instant>>,
    /// Statistics
    stats: Arc<RwLock<NetworkMonitorStats>>,
}

/// Network monitor statistics
#[derive(Debug, Clone, Default)]
pub struct NetworkMonitorStats {
    /// Number of interface additions detected
    pub interfaces_added: usize,
    /// Number of interface removals detected
    pub interfaces_removed: usize,
    /// Number of primary interface changes
    pub primary_changes: usize,
    /// Number of address changes detected
    pub address_changes: usize,
    /// Total network change events
    pub total_changes: usize,
}

/// Errors that can occur during network monitoring
#[derive(Debug, Error)]
pub enum NetworkMonitorError {
    #[error("Failed to get network interfaces: {0}")]
    InterfaceQueryFailed(String),

    #[error("No active network interfaces found")]
    NoActiveInterfaces,
}

impl NetworkMonitor {
    /// Create a new network monitor
    pub fn new(config: NetworkMonitorConfig) -> Self {
        Self {
            config,
            interfaces: Arc::new(RwLock::new(HashMap::new())),
            primary_interface: Arc::new(RwLock::new(None)),
            last_change: Arc::new(RwLock::new(Instant::now())),
            stats: Arc::new(RwLock::new(NetworkMonitorStats::default())),
        }
    }

    /// Get current network interfaces
    pub fn get_interfaces(&self) -> HashMap<String, NetworkInterface> {
        self.interfaces.read().clone()
    }

    /// Get the primary (preferred) interface
    pub fn get_primary_interface(&self) -> Option<NetworkInterface> {
        self.primary_interface.read().clone()
    }

    /// Check for network changes and return any detected changes
    ///
    /// This should be called periodically (e.g., from a background task)
    pub fn check_for_changes(&self) -> Result<Vec<NetworkChange>, NetworkMonitorError> {
        // Get current interfaces from the system
        let current_interfaces = self.query_system_interfaces()?;

        // Debounce: skip if too soon after last change
        {
            let last_change = *self.last_change.read();
            if last_change.elapsed() < self.config.debounce_duration {
                return Ok(Vec::new());
            }
        }

        let mut changes = Vec::new();
        let mut interfaces_lock = self.interfaces.write();

        // Detect added and changed interfaces
        for (name, new_iface) in &current_interfaces {
            if let Some(old_iface) = interfaces_lock.get(name) {
                // Check for address changes
                if old_iface.addresses != new_iface.addresses {
                    debug!(
                        "Address changed on interface {}: {:?} -> {:?}",
                        name, old_iface.addresses, new_iface.addresses
                    );
                    changes.push(NetworkChange::AddressChanged {
                        interface: name.clone(),
                        old_addresses: old_iface.addresses.clone(),
                        new_addresses: new_iface.addresses.clone(),
                    });
                    self.stats.write().address_changes += 1;
                }

                // Check for status changes
                if old_iface.is_active != new_iface.is_active {
                    if new_iface.is_active {
                        info!("Interface {} is now active", name);
                        changes.push(NetworkChange::InterfaceUp(new_iface.clone()));
                    } else {
                        info!("Interface {} is now inactive", name);
                        changes.push(NetworkChange::InterfaceDown(new_iface.clone()));
                    }
                }
            } else {
                // New interface
                info!("New interface detected: {}", name);
                changes.push(NetworkChange::InterfaceAdded(new_iface.clone()));
                self.stats.write().interfaces_added += 1;
            }
        }

        // Detect removed interfaces
        for (name, old_iface) in interfaces_lock.iter() {
            if !current_interfaces.contains_key(name) {
                info!("Interface removed: {}", name);
                changes.push(NetworkChange::InterfaceRemoved(old_iface.clone()));
                self.stats.write().interfaces_removed += 1;
            }
        }

        // Update stored interfaces
        *interfaces_lock = current_interfaces.clone();
        drop(interfaces_lock);

        // Determine primary interface
        let old_primary = self.primary_interface.read().clone();
        let new_primary = self.select_primary_interface(&current_interfaces);

        if old_primary != new_primary {
            if let Some(new) = &new_primary {
                info!(
                    "Primary interface changed: {:?} -> {}",
                    old_primary.as_ref().map(|i| i.name.as_str()),
                    new.name
                );
                changes.push(NetworkChange::PrimaryInterfaceChanged {
                    old: old_primary,
                    new: new.clone(),
                });
                self.stats.write().primary_changes += 1;
            }
            *self.primary_interface.write() = new_primary;
        }

        if !changes.is_empty() {
            *self.last_change.write() = Instant::now();
            self.stats.write().total_changes += changes.len();
        }

        Ok(changes)
    }

    /// Select the primary interface based on availability and priority
    fn select_primary_interface(
        &self,
        interfaces: &HashMap<String, NetworkInterface>,
    ) -> Option<NetworkInterface> {
        interfaces
            .values()
            .filter(|iface| {
                iface.is_active
                    && !iface.addresses.is_empty()
                    && (self.config.monitor_loopback
                        || iface.interface_type != InterfaceType::Loopback)
            })
            .max_by_key(|iface| iface.interface_type.priority())
            .cloned()
    }

    /// Query system for current network interfaces
    ///
    /// This is a simplified implementation that creates mock data for demonstration.
    /// In a real implementation, this would use platform-specific APIs to query
    /// actual network interfaces (e.g., getifaddrs on Unix, GetAdaptersAddresses on Windows)
    fn query_system_interfaces(
        &self,
    ) -> Result<HashMap<String, NetworkInterface>, NetworkMonitorError> {
        // This is a placeholder implementation
        // In production, use platform-specific APIs or crates like `if-addrs` or `pnet`

        warn!("Using mock network interface detection - implement platform-specific querying for production use");

        let interfaces = HashMap::new();

        // Mock interface - in real implementation, query actual interfaces
        // For now, return empty to avoid errors

        Ok(interfaces)
    }

    /// Get statistics
    pub fn stats(&self) -> NetworkMonitorStats {
        self.stats.read().clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        *self.stats.write() = NetworkMonitorStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interface_type_from_name() {
        assert_eq!(InterfaceType::from_name("eth0"), InterfaceType::Ethernet);
        assert_eq!(InterfaceType::from_name("wlan0"), InterfaceType::WiFi);
        assert_eq!(InterfaceType::from_name("wwan0"), InterfaceType::Cellular);
        assert_eq!(InterfaceType::from_name("lo"), InterfaceType::Loopback);
    }

    #[test]
    fn test_interface_priority() {
        assert!(InterfaceType::Ethernet.priority() > InterfaceType::WiFi.priority());
        assert!(InterfaceType::WiFi.priority() > InterfaceType::Cellular.priority());
        assert!(InterfaceType::Cellular.priority() > InterfaceType::Loopback.priority());
    }

    #[test]
    fn test_network_monitor_creation() {
        let monitor = NetworkMonitor::new(NetworkMonitorConfig::default());
        assert!(monitor.get_interfaces().is_empty());
        assert!(monitor.get_primary_interface().is_none());
    }

    #[test]
    fn test_select_primary_interface() {
        let monitor = NetworkMonitor::new(NetworkMonitorConfig::default());
        let mut interfaces = HashMap::new();

        // Add WiFi interface
        interfaces.insert(
            "wlan0".to_string(),
            NetworkInterface {
                name: "wlan0".to_string(),
                addresses: vec!["192.168.1.100"
                    .parse()
                    .expect("test: valid IP address literal should parse")],
                interface_type: InterfaceType::WiFi,
                is_active: true,
            },
        );

        // Add cellular interface
        interfaces.insert(
            "wwan0".to_string(),
            NetworkInterface {
                name: "wwan0".to_string(),
                addresses: vec!["10.0.0.100"
                    .parse()
                    .expect("test: valid IP address literal should parse")],
                interface_type: InterfaceType::Cellular,
                is_active: true,
            },
        );

        let primary = monitor.select_primary_interface(&interfaces);
        assert!(primary.is_some());
        // WiFi should be preferred over cellular
        assert_eq!(
            primary
                .expect("test: primary interface should be Some for active interfaces")
                .interface_type,
            InterfaceType::WiFi
        );
    }

    #[test]
    fn test_ethernet_preferred_over_wifi() {
        let monitor = NetworkMonitor::new(NetworkMonitorConfig::default());
        let mut interfaces = HashMap::new();

        interfaces.insert(
            "eth0".to_string(),
            NetworkInterface {
                name: "eth0".to_string(),
                addresses: vec!["192.168.1.50"
                    .parse()
                    .expect("test: valid IP address literal should parse")],
                interface_type: InterfaceType::Ethernet,
                is_active: true,
            },
        );

        interfaces.insert(
            "wlan0".to_string(),
            NetworkInterface {
                name: "wlan0".to_string(),
                addresses: vec!["192.168.1.100"
                    .parse()
                    .expect("test: valid IP address literal should parse")],
                interface_type: InterfaceType::WiFi,
                is_active: true,
            },
        );

        let primary = monitor.select_primary_interface(&interfaces);
        assert!(primary.is_some());
        assert_eq!(
            primary
                .expect("test: primary interface should be Some when ethernet is active")
                .interface_type,
            InterfaceType::Ethernet
        );
    }

    #[test]
    fn test_inactive_interface_not_selected() {
        let monitor = NetworkMonitor::new(NetworkMonitorConfig::default());
        let mut interfaces = HashMap::new();

        interfaces.insert(
            "wlan0".to_string(),
            NetworkInterface {
                name: "wlan0".to_string(),
                addresses: vec!["192.168.1.100"
                    .parse()
                    .expect("test: valid IP address literal should parse")],
                interface_type: InterfaceType::WiFi,
                is_active: false, // Inactive
            },
        );

        let primary = monitor.select_primary_interface(&interfaces);
        assert!(primary.is_none());
    }

    #[test]
    fn test_interface_without_addresses_not_selected() {
        let monitor = NetworkMonitor::new(NetworkMonitorConfig::default());
        let mut interfaces = HashMap::new();

        interfaces.insert(
            "wlan0".to_string(),
            NetworkInterface {
                name: "wlan0".to_string(),
                addresses: vec![], // No addresses
                interface_type: InterfaceType::WiFi,
                is_active: true,
            },
        );

        let primary = monitor.select_primary_interface(&interfaces);
        assert!(primary.is_none());
    }

    #[test]
    fn test_loopback_filtering() {
        let config = NetworkMonitorConfig {
            monitor_loopback: false,
            ..Default::default()
        };
        let monitor = NetworkMonitor::new(config);
        let mut interfaces = HashMap::new();

        interfaces.insert(
            "lo".to_string(),
            NetworkInterface {
                name: "lo".to_string(),
                addresses: vec!["127.0.0.1"
                    .parse()
                    .expect("test: valid loopback IP literal should parse")],
                interface_type: InterfaceType::Loopback,
                is_active: true,
            },
        );

        let primary = monitor.select_primary_interface(&interfaces);
        assert!(primary.is_none()); // Loopback should be filtered out
    }

    #[test]
    fn test_stats_initialization() {
        let monitor = NetworkMonitor::new(NetworkMonitorConfig::default());
        let stats = monitor.stats();
        assert_eq!(stats.interfaces_added, 0);
        assert_eq!(stats.interfaces_removed, 0);
        assert_eq!(stats.primary_changes, 0);
        assert_eq!(stats.address_changes, 0);
        assert_eq!(stats.total_changes, 0);
    }

    #[test]
    fn test_mobile_config() {
        let config = NetworkMonitorConfig::mobile();
        assert_eq!(config.poll_interval, Duration::from_secs(2));
        assert!(!config.monitor_loopback);
    }

    #[test]
    fn test_server_config() {
        let config = NetworkMonitorConfig::server();
        assert_eq!(config.poll_interval, Duration::from_secs(30));
        assert!(!config.monitor_loopback);
    }
}
