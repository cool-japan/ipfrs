//! Configuration management for IPFRS CLI
//!
//! Handles reading, writing, and validating IPFRS configuration files.

#![allow(dead_code)]

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Global cached configuration
static CACHED_CONFIG: OnceLock<Config> = OnceLock::new();

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    /// General settings
    #[serde(default)]
    pub general: GeneralConfig,

    /// Storage settings
    #[serde(default)]
    pub storage: StorageConfig,

    /// Network settings
    #[serde(default)]
    pub network: NetworkConfig,

    /// Gateway settings
    #[serde(default)]
    pub gateway: GatewayConfig,

    /// API settings
    #[serde(default)]
    pub api: ApiConfig,

    /// Shell settings
    #[serde(default)]
    pub shell: ShellConfig,
}

/// General configuration options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Data directory path
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,

    /// Log level (error, warn, info, debug, trace)
    #[serde(default = "default_log_level")]
    pub log_level: String,

    /// Enable colored output
    #[serde(default = "default_true")]
    pub color: bool,

    /// Output format (text, json)
    #[serde(default = "default_format")]
    pub format: String,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
            log_level: default_log_level(),
            color: true,
            format: default_format(),
        }
    }
}

/// Storage configuration options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Block store path (relative to data_dir)
    #[serde(default = "default_blocks_path")]
    pub blocks_path: String,

    /// Maximum cache size in bytes
    #[serde(default = "default_cache_size")]
    pub cache_size: u64,

    /// Enable write-ahead logging
    #[serde(default = "default_true")]
    pub wal_enabled: bool,

    /// Garbage collection interval in seconds
    #[serde(default = "default_gc_interval")]
    pub gc_interval: u64,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            blocks_path: default_blocks_path(),
            cache_size: default_cache_size(),
            wal_enabled: true,
            gc_interval: default_gc_interval(),
        }
    }
}

/// Network configuration options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Listen addresses
    #[serde(default = "default_listen_addrs")]
    pub listen_addrs: Vec<String>,

    /// Bootstrap peers
    #[serde(default)]
    pub bootstrap_peers: Vec<String>,

    /// Maximum number of connections
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,

    /// Enable DHT
    #[serde(default = "default_true")]
    pub dht_enabled: bool,

    /// Enable mDNS discovery
    #[serde(default = "default_true")]
    pub mdns_enabled: bool,

    /// Connection timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout: u64,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            listen_addrs: default_listen_addrs(),
            bootstrap_peers: Vec::new(),
            max_connections: default_max_connections(),
            dht_enabled: true,
            mdns_enabled: true,
            timeout: default_timeout(),
        }
    }
}

/// Gateway configuration options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    /// Gateway listen address
    #[serde(default = "default_gateway_addr")]
    pub listen_addr: String,

    /// Enable CORS
    #[serde(default = "default_true")]
    pub cors_enabled: bool,

    /// CORS allowed origins
    #[serde(default)]
    pub cors_origins: Vec<String>,

    /// Enable TLS
    #[serde(default)]
    pub tls_enabled: bool,

    /// TLS certificate path
    #[serde(default)]
    pub tls_cert_path: Option<String>,

    /// TLS key path
    #[serde(default)]
    pub tls_key_path: Option<String>,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            listen_addr: default_gateway_addr(),
            cors_enabled: true,
            cors_origins: Vec::new(),
            tls_enabled: false,
            tls_cert_path: None,
            tls_key_path: None,
        }
    }
}

/// API configuration options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// API listen address (for local daemon)
    #[serde(default = "default_api_addr")]
    pub listen_addr: String,

    /// Remote API URL (for connecting to remote daemon)
    /// Format: http://hostname:port or https://hostname:port
    /// Overrides listen_addr when set
    #[serde(default)]
    pub remote_url: Option<String>,

    /// Enable authentication
    #[serde(default)]
    pub auth_enabled: bool,

    /// API token (if auth is enabled)
    #[serde(default)]
    pub api_token: Option<String>,

    /// Connection timeout for remote API in seconds
    #[serde(default = "default_api_timeout")]
    pub timeout: u64,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            listen_addr: default_api_addr(),
            remote_url: None,
            auth_enabled: false,
            api_token: None,
            timeout: default_api_timeout(),
        }
    }
}

// Default value functions
fn default_data_dir() -> PathBuf {
    PathBuf::from(".ipfrs")
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_true() -> bool {
    true
}

fn default_format() -> String {
    "text".to_string()
}

fn default_blocks_path() -> String {
    "blocks".to_string()
}

fn default_cache_size() -> u64 {
    100 * 1024 * 1024 // 100MB
}

fn default_gc_interval() -> u64 {
    3600 // 1 hour
}

fn default_listen_addrs() -> Vec<String> {
    vec![
        "/ip4/0.0.0.0/tcp/4001".to_string(),
        "/ip6/::/tcp/4001".to_string(),
    ]
}

fn default_max_connections() -> u32 {
    256
}

fn default_timeout() -> u64 {
    30
}

fn default_gateway_addr() -> String {
    "127.0.0.1:8080".to_string()
}

fn default_api_addr() -> String {
    "127.0.0.1:5001".to_string()
}

fn default_api_timeout() -> u64 {
    60 // 60 seconds for remote API calls
}

/// Shell configuration options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellConfig {
    /// User-defined command aliases
    #[serde(default)]
    pub aliases: std::collections::HashMap<String, String>,

    /// Enable command hints
    #[serde(default = "default_true")]
    pub hints_enabled: bool,

    /// Enable syntax highlighting
    #[serde(default = "default_true")]
    pub highlighting_enabled: bool,

    /// Maximum history entries
    #[serde(default = "default_history_size")]
    pub history_size: usize,
}

impl Default for ShellConfig {
    fn default() -> Self {
        Self {
            aliases: std::collections::HashMap::new(),
            hints_enabled: true,
            highlighting_enabled: true,
            history_size: default_history_size(),
        }
    }
}

fn default_history_size() -> usize {
    1000
}

impl Config {
    /// Load configuration from default locations (with caching)
    ///
    /// This method uses a global cache to avoid repeated disk reads.
    /// The config is loaded once and reused for subsequent calls.
    ///
    /// Search order:
    /// 1. .ipfrs/config.toml (current directory)
    /// 2. ~/.ipfrs/config.toml (user home)
    /// 3. /etc/ipfrs/config.toml (system)
    pub fn load() -> Result<Self> {
        Ok(CACHED_CONFIG
            .get_or_init(|| Self::load_uncached().unwrap_or_default())
            .clone())
    }

    /// Load configuration from default locations without caching
    ///
    /// Use this when you need to force a fresh load (e.g., after config changes).
    ///
    /// Search order:
    /// 1. .ipfrs/config.toml (current directory)
    /// 2. ~/.ipfrs/config.toml (user home)
    /// 3. /etc/ipfrs/config.toml (system)
    ///
    /// Environment variables (override config file):
    /// - IPFRS_PATH: Data directory
    /// - IPFRS_LOG_LEVEL: Log level
    /// - IPFRS_API_URL: Remote API URL
    /// - IPFRS_API_TOKEN: API authentication token
    pub fn load_uncached() -> Result<Self> {
        // Try local config first
        let local_config = PathBuf::from(".ipfrs/config.toml");
        let mut config = if local_config.exists() {
            Self::load_from(&local_config)?
        } else if let Some(home) = dirs::home_dir() {
            // Try user home config
            let user_config = home.join(".ipfrs/config.toml");
            if user_config.exists() {
                Self::load_from(&user_config)?
            } else {
                // Try system config
                let system_config = PathBuf::from("/etc/ipfrs/config.toml");
                if system_config.exists() {
                    Self::load_from(&system_config)?
                } else {
                    Self::default()
                }
            }
        } else {
            Self::default()
        };

        // Apply environment variable overrides
        config.apply_env_overrides();

        Ok(config)
    }

    /// Apply environment variable overrides to configuration
    ///
    /// Supported environment variables:
    /// - IPFRS_PATH: Data directory
    /// - IPFRS_LOG_LEVEL: Log level (error, warn, info, debug, trace)
    /// - IPFRS_API_URL: Remote API URL (http://host:port or https://host:port)
    /// - IPFRS_API_TOKEN: API authentication token
    fn apply_env_overrides(&mut self) {
        use std::env;

        // Data directory
        if let Ok(path) = env::var("IPFRS_PATH") {
            self.general.data_dir = PathBuf::from(path);
        }

        // Log level
        if let Ok(level) = env::var("IPFRS_LOG_LEVEL") {
            self.general.log_level = level;
        }

        // Remote API URL
        if let Ok(url) = env::var("IPFRS_API_URL") {
            self.api.remote_url = Some(url);
        }

        // API token
        if let Ok(token) = env::var("IPFRS_API_TOKEN") {
            self.api.api_token = Some(token);
            self.api.auth_enabled = true;
        }
    }

    /// Get the effective API URL
    ///
    /// Returns the remote URL if set, otherwise returns the local listen address
    /// formatted as http://address
    pub fn api_url(&self) -> String {
        self.api
            .remote_url
            .clone()
            .unwrap_or_else(|| format!("http://{}", self.api.listen_addr))
    }

    /// Check if connecting to a remote daemon
    pub fn is_remote(&self) -> bool {
        self.api.remote_url.is_some()
    }

    /// Clear the cached configuration
    ///
    /// This forces the next `load()` call to reload from disk.
    /// Note: Due to OnceLock limitations, this doesn't actually clear the cache
    /// but is provided for API consistency. Use `load_uncached()` instead.
    pub fn clear_cache() {
        // OnceLock doesn't support clearing, but we provide this for API consistency
        // Users should use load_uncached() if they need a fresh load
    }

    /// Load configuration from a specific path
    pub fn load_from(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        let config: Config = toml::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {}", path.display()))?;

        Ok(config)
    }

    /// Save configuration to a file
    pub fn save(&self, path: &Path) -> Result<()> {
        let content = toml::to_string_pretty(self).context("Failed to serialize config")?;

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
        }

        std::fs::write(path, content)
            .with_context(|| format!("Failed to write config file: {}", path.display()))?;

        Ok(())
    }

    /// Get the default configuration file path
    ///
    /// Returns the path to the user's configuration file (~/.ipfrs/config.toml)
    pub fn default_path() -> Result<PathBuf> {
        let home = dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?;
        Ok(home.join(".ipfrs/config.toml"))
    }

    /// Generate default config file with comments
    pub fn generate_default_config() -> String {
        r#"# IPFRS Configuration File
# Generated by ipfrs init

[general]
# Data directory path
data_dir = ".ipfrs"
# Log level: error, warn, info, debug, trace
log_level = "info"
# Enable colored output
color = true
# Default output format: text, json
format = "text"

[storage]
# Block store path (relative to data_dir)
blocks_path = "blocks"
# Maximum cache size in bytes (default: 100MB)
cache_size = 104857600
# Enable write-ahead logging
wal_enabled = true
# Garbage collection interval in seconds
gc_interval = 3600

[network]
# Listen addresses
listen_addrs = [
    "/ip4/0.0.0.0/tcp/4001",
    "/ip6/::/tcp/4001"
]
# Bootstrap peers (add your own or use public IPFS bootstrap)
bootstrap_peers = []
# Maximum number of connections
max_connections = 256
# Enable DHT for peer/content discovery
dht_enabled = true
# Enable mDNS for local peer discovery
mdns_enabled = true
# Connection timeout in seconds
timeout = 30

[gateway]
# HTTP Gateway listen address
listen_addr = "127.0.0.1:8080"
# Enable CORS
cors_enabled = true
# CORS allowed origins (empty = all)
cors_origins = []
# Enable TLS
tls_enabled = false
# TLS certificate path
# tls_cert_path = "/path/to/cert.pem"
# TLS key path
# tls_key_path = "/path/to/key.pem"

[api]
# API server listen address (for local daemon)
listen_addr = "127.0.0.1:5001"
# Remote API URL (for connecting to remote daemon)
# Format: http://hostname:port or https://hostname:port
# When set, overrides listen_addr for client commands
# Example: remote_url = "http://192.168.1.100:5001"
# remote_url = ""
# Connection timeout for remote API in seconds
timeout = 60
# Enable API authentication
auth_enabled = false
# API token (required if auth_enabled = true or connecting to authenticated remote)
# api_token = "your-secret-token"

[shell]
# Enable command hints (suggestions as you type)
hints_enabled = true
# Enable syntax highlighting
highlighting_enabled = true
# Maximum number of history entries
history_size = 1000
# User-defined command aliases
# Example: aliases = { "myalias" = "full command here" }
# [shell.aliases]
# ll = "ls -la"
# gs = "git status"
"#
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.general.log_level, "info");
        assert!(config.general.color);
        assert_eq!(config.storage.blocks_path, "blocks");
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let toml_str =
            toml::to_string_pretty(&config).expect("test: config serialization should succeed");
        let parsed: Config =
            toml::from_str(&toml_str).expect("test: config deserialization should succeed");
        assert_eq!(parsed.general.log_level, config.general.log_level);
    }

    #[test]
    fn test_generate_default_config() {
        let config_str = Config::generate_default_config();
        assert!(config_str.contains("[general]"));
        assert!(config_str.contains("[storage]"));
        assert!(config_str.contains("[network]"));
        assert!(config_str.contains("[shell]"));
    }

    #[test]
    fn test_shell_config_default() {
        let shell_config = super::ShellConfig::default();
        assert!(shell_config.hints_enabled);
        assert!(shell_config.highlighting_enabled);
        assert_eq!(shell_config.history_size, 1000);
        assert!(shell_config.aliases.is_empty());
    }

    #[test]
    fn test_shell_config_serialization() {
        let mut shell_config = super::ShellConfig::default();
        shell_config
            .aliases
            .insert("ll".to_string(), "ls -la".to_string());
        shell_config
            .aliases
            .insert("gs".to_string(), "git status".to_string());

        let toml_str = toml::to_string_pretty(&shell_config)
            .expect("test: shell config serialization should succeed");
        let parsed: super::ShellConfig =
            toml::from_str(&toml_str).expect("test: shell config deserialization should succeed");

        assert_eq!(parsed.hints_enabled, shell_config.hints_enabled);
        assert_eq!(parsed.history_size, shell_config.history_size);
        assert_eq!(parsed.aliases.len(), 2);
        assert_eq!(parsed.aliases.get("ll"), Some(&"ls -la".to_string()));
        assert_eq!(parsed.aliases.get("gs"), Some(&"git status".to_string()));
    }

    #[test]
    fn test_config_caching() {
        // Load config twice - should return the same instance (via caching)
        let config1 = Config::load().expect("test: config load should succeed");
        let config2 = Config::load().expect("test: config load should succeed");

        // Both should have the same default values
        assert_eq!(config1.general.log_level, config2.general.log_level);
        assert_eq!(config1.storage.cache_size, config2.storage.cache_size);
    }

    #[test]
    fn test_config_uncached_load() {
        // Load uncached config multiple times
        let config1 = Config::load_uncached().expect("test: config uncached load should succeed");
        let config2 = Config::load_uncached().expect("test: config uncached load should succeed");

        // Both should have the same default values
        assert_eq!(config1.general.log_level, config2.general.log_level);
        assert_eq!(config1.storage.cache_size, config2.storage.cache_size);
    }

    #[test]
    fn test_clear_cache() {
        // Test that clear_cache doesn't panic (it's a no-op currently)
        Config::clear_cache();

        // Config should still load successfully
        let config = Config::load().expect("test: config load should succeed");
        assert_eq!(config.general.log_level, "info");
    }
}
