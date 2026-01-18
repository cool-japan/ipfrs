//! Plugin system for IPFRS CLI
//!
//! This module provides a plugin system that allows users to extend the IPFRS CLI
//! with custom commands. Plugins can be written in any language and interfaced
//! via a simple executable-based protocol.
//!
//! # Plugin Discovery
//!
//! Plugins are discovered in the following locations (in order):
//! 1. `~/.ipfrs/plugins/` - User plugins
//! 2. `/usr/local/lib/ipfrs/plugins/` - System-wide plugins (Unix)
//! 3. `$IPFRS_PLUGIN_PATH` - Custom plugin directories (colon-separated)
//!
//! # Plugin Protocol
//!
//! Plugins are executables that follow this naming convention:
//! - `ipfrs-plugin-<name>` for the executable
//!
//! When invoked, plugins receive:
//! - Arguments passed after the plugin name
//! - Environment variables:
//!   - `IPFRS_API_URL` - Daemon API endpoint
//!   - `IPFRS_DATA_DIR` - Repository data directory
//!   - `IPFRS_CONFIG` - Config file path
//!
//! # Examples
//!
//! ```rust
//! use ipfrs_cli::plugin::PluginManager;
//!
//! // Discover all available plugins
//! let mut manager = PluginManager::new();
//! let plugins = manager.discover_plugins();
//!
//! for plugin in plugins {
//!     println!("Found plugin: {}", plugin.name());
//! }
//! ```
//!
//! ## Creating a Plugin
//!
//! Create an executable named `ipfrs-plugin-hello`:
//!
//! ```bash
//! #!/bin/bash
//! # ipfrs-plugin-hello
//! echo "Hello from plugin!"
//! echo "API URL: $IPFRS_API_URL"
//! ```
//!
//! Make it executable and place in `~/.ipfrs/plugins/`:
//!
//! ```bash
//! chmod +x ipfrs-plugin-hello
//! mv ipfrs-plugin-hello ~/.ipfrs/plugins/
//! ```
//!
//! Then use it:
//!
//! ```bash
//! ipfrs plugin hello
//! ```

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Plugin manager for discovering and executing plugins
pub struct PluginManager {
    plugin_paths: Vec<PathBuf>,
    plugins: HashMap<String, Plugin>,
}

/// Represents a single plugin
#[derive(Debug, Clone)]
pub struct Plugin {
    name: String,
    path: PathBuf,
    description: Option<String>,
}

impl Plugin {
    /// Create a new plugin
    pub fn new(name: String, path: PathBuf) -> Self {
        Self {
            name,
            path,
            description: None,
        }
    }

    /// Get the plugin name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the plugin executable path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the plugin description (if available)
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Set the plugin description
    pub fn with_description(mut self, desc: String) -> Self {
        self.description = Some(desc);
        self
    }

    /// Execute the plugin with the given arguments
    pub fn execute(&self, args: &[String], config: &crate::config::Config) -> Result<i32> {
        let mut cmd = Command::new(&self.path);
        cmd.args(args);

        // Set environment variables for plugin
        cmd.env("IPFRS_DATA_DIR", &config.general.data_dir);
        cmd.env("IPFRS_LOG_LEVEL", &config.general.log_level);

        if let Some(api_url) = &config.api.remote_url {
            cmd.env("IPFRS_API_URL", api_url);
        }

        if let Some(api_token) = &config.api.api_token {
            cmd.env("IPFRS_API_TOKEN", api_token);
        }

        // Inherit stdio so plugin output goes directly to terminal
        cmd.stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());

        let status = cmd
            .status()
            .with_context(|| format!("Failed to execute plugin '{}'", self.name))?;

        Ok(status.code().unwrap_or(1))
    }

    /// Query plugin metadata (description, version, etc.)
    /// Plugins should support `--plugin-info` flag to return JSON metadata
    pub fn query_metadata(&mut self) -> Result<()> {
        let output = Command::new(&self.path).arg("--plugin-info").output();

        if let Ok(output) = output {
            if output.status.success() {
                if let Ok(info) = serde_json::from_slice::<HashMap<String, String>>(&output.stdout)
                {
                    if let Some(desc) = info.get("description") {
                        self.description = Some(desc.clone());
                    }
                }
            }
        }

        Ok(())
    }
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        let mut plugin_paths = Vec::new();

        // Add user plugin directory
        if let Some(home) = dirs::home_dir() {
            plugin_paths.push(home.join(".ipfrs").join("plugins"));
        }

        // Add system plugin directory (Unix)
        #[cfg(unix)]
        {
            plugin_paths.push(PathBuf::from("/usr/local/lib/ipfrs/plugins"));
        }

        // Add custom plugin path from environment
        if let Ok(custom_paths) = env::var("IPFRS_PLUGIN_PATH") {
            for path in custom_paths.split(':') {
                if !path.is_empty() {
                    plugin_paths.push(PathBuf::from(path));
                }
            }
        }

        Self {
            plugin_paths,
            plugins: HashMap::new(),
        }
    }

    /// Add a custom plugin search path
    pub fn add_plugin_path(&mut self, path: PathBuf) {
        if !self.plugin_paths.contains(&path) {
            self.plugin_paths.push(path);
        }
    }

    /// Discover all available plugins in the plugin paths
    pub fn discover_plugins(&mut self) -> Vec<&Plugin> {
        self.plugins.clear();

        for plugin_dir in &self.plugin_paths {
            if !plugin_dir.exists() {
                continue;
            }

            if let Ok(entries) = std::fs::read_dir(plugin_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();

                    // Check if it's an executable file
                    if !path.is_file() {
                        continue;
                    }

                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        if let Ok(metadata) = path.metadata() {
                            let permissions = metadata.permissions();
                            // Check if executable bit is set
                            if permissions.mode() & 0o111 == 0 {
                                continue;
                            }
                        }
                    }

                    // Extract plugin name from filename
                    if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                        if let Some(plugin_name) = filename.strip_prefix("ipfrs-plugin-") {
                            let mut plugin = Plugin::new(plugin_name.to_string(), path.clone());

                            // Try to query metadata
                            let _ = plugin.query_metadata();

                            self.plugins.insert(plugin_name.to_string(), plugin);
                        }
                    }
                }
            }
        }

        self.plugins.values().collect()
    }

    /// Get a plugin by name
    pub fn get_plugin(&self, name: &str) -> Option<&Plugin> {
        self.plugins.get(name)
    }

    /// List all discovered plugin names
    pub fn list_plugins(&self) -> Vec<&str> {
        self.plugins.keys().map(|s| s.as_str()).collect()
    }

    /// Execute a plugin with the given arguments
    pub fn execute_plugin(
        &self,
        name: &str,
        args: &[String],
        config: &crate::config::Config,
    ) -> Result<i32> {
        let plugin = self
            .get_plugin(name)
            .with_context(|| format!("Plugin '{}' not found", name))?;

        plugin.execute(args, config)
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_creation() {
        let plugin = Plugin::new(
            "test".to_string(),
            PathBuf::from("/usr/local/lib/ipfrs/plugins/ipfrs-plugin-test"),
        );

        assert_eq!(plugin.name(), "test");
        assert_eq!(
            plugin.path(),
            Path::new("/usr/local/lib/ipfrs/plugins/ipfrs-plugin-test")
        );
        assert!(plugin.description().is_none());
    }

    #[test]
    fn test_plugin_with_description() {
        let plugin = Plugin::new("test".to_string(), PathBuf::from("/tmp/plugin"))
            .with_description("A test plugin".to_string());

        assert_eq!(plugin.description(), Some("A test plugin"));
    }

    #[test]
    fn test_plugin_manager_creation() {
        let manager = PluginManager::new();
        assert!(!manager.plugin_paths.is_empty());
    }

    #[test]
    fn test_add_plugin_path() {
        let mut manager = PluginManager::new();
        let custom_path = PathBuf::from("/custom/plugins");

        manager.add_plugin_path(custom_path.clone());
        assert!(manager.plugin_paths.contains(&custom_path));

        // Adding duplicate should not duplicate
        let initial_count = manager.plugin_paths.len();
        manager.add_plugin_path(custom_path.clone());
        assert_eq!(manager.plugin_paths.len(), initial_count);
    }

    #[test]
    fn test_plugin_manager_default() {
        let manager = PluginManager::default();
        assert!(!manager.plugin_paths.is_empty());
    }

    #[test]
    fn test_list_plugins_empty() {
        let manager = PluginManager::new();
        assert_eq!(manager.list_plugins().len(), 0);
    }

    #[test]
    fn test_get_plugin_not_found() {
        let manager = PluginManager::new();
        assert!(manager.get_plugin("nonexistent").is_none());
    }
}
