//! Configuration management commands
//!
//! This module provides functions for managing IPFRS configuration:
//! - `config_show` - Display current configuration
//! - `config_export` - Export configuration to file
//! - `config_import` - Import configuration from file
//! - `config_edit` - Open configuration in editor

use crate::config::Config;
use crate::output::{self, error, print_kv, success};
use anyhow::{Context, Result};

/// Show current configuration
pub async fn config_show(format: String) -> Result<()> {
    let config = Config::load()?;

    if format == "json" {
        let json = serde_json::to_string_pretty(&config)?;
        println!("{}", json);
    } else {
        println!("IPFRS Configuration");
        println!("===================");
        println!();

        println!("General:");
        println!("--------");
        print_kv(
            "  Data directory",
            &config.general.data_dir.display().to_string(),
        );
        print_kv("  Log level", &config.general.log_level);
        print_kv("  Color output", &config.general.color.to_string());
        print_kv("  Default format", &config.general.format);
        println!();

        println!("Storage:");
        println!("--------");
        print_kv("  Blocks path", &config.storage.blocks_path);
        print_kv("  Cache size", &format_bytes(config.storage.cache_size));
        print_kv("  WAL enabled", &config.storage.wal_enabled.to_string());
        print_kv("  GC interval", &format!("{}s", config.storage.gc_interval));
        println!();

        println!("Network:");
        println!("--------");
        print_kv(
            "  Max connections",
            &config.network.max_connections.to_string(),
        );
        println!("  Listen addresses:");
        for addr in &config.network.listen_addrs {
            println!("    - {}", addr);
        }
        println!();

        println!("Gateway:");
        println!("--------");
        print_kv("  Listen address", &config.gateway.listen_addr);
        println!();

        println!("API:");
        println!("----");
        print_kv("  Listen address", &config.api.listen_addr);
        print_kv("  Auth enabled", &config.api.auth_enabled.to_string());
        print_kv("  Timeout", &format!("{}s", config.api.timeout));
        if let Some(ref remote_url) = config.api.remote_url {
            print_kv("  Remote URL", remote_url);
        }
        println!();
    }

    Ok(())
}

/// Export configuration to file
pub async fn config_export(output: String, format: String) -> Result<()> {
    let config = Config::load()?;

    // Export configuration
    let content = if format == "json" {
        serde_json::to_string_pretty(&config)?
    } else if format == "toml" {
        toml::to_string_pretty(&config)?
    } else if format == "yaml" {
        serde_yaml::to_string(&config)?
    } else {
        anyhow::bail!(
            "Unsupported export format: {}. Use json, toml, or yaml.",
            format
        );
    };

    let content_len = content.len();

    std::fs::write(&output, content)
        .with_context(|| format!("Failed to write configuration to {}", output))?;

    success(&format!("Configuration exported to {}", output));
    print_kv("Format", &format);
    print_kv("Size", &format!("{} bytes", content_len));

    Ok(())
}

/// Import configuration from file
pub async fn config_import(input: String, dry_run: bool) -> Result<()> {
    // Read input file
    let content = std::fs::read_to_string(&input)
        .with_context(|| format!("Failed to read configuration from {}", input))?;

    // Detect format from file extension
    let format = if input.ends_with(".json") {
        "json"
    } else if input.ends_with(".toml") {
        "toml"
    } else if input.ends_with(".yaml") || input.ends_with(".yml") {
        "yaml"
    } else {
        // Try to auto-detect
        if content.trim().starts_with('{') {
            "json"
        } else if content.contains('[') && content.contains(']') {
            "toml"
        } else {
            "yaml"
        }
    };

    // Parse configuration
    let config: Config = match format {
        "json" => serde_json::from_str(&content)?,
        "toml" => toml::from_str(&content)?,
        "yaml" => serde_yaml::from_str(&content)?,
        _ => anyhow::bail!("Could not detect configuration format"),
    };

    if dry_run {
        output::info("Dry run mode - configuration validation only");
        success("✓ Configuration is valid");
        print_kv("Format detected", format);
        println!();
        println!("Configuration summary:");
        print_kv(
            "  Data directory",
            &config.general.data_dir.display().to_string(),
        );
        print_kv("  Log level", &config.general.log_level);
        print_kv(
            "  Max connections",
            &config.network.max_connections.to_string(),
        );
        println!();
        output::info("Use without --dry-run to apply the configuration");
    } else {
        // Get default config path
        let config_path = Config::default_path()?;

        // Backup existing config
        if config_path.exists() {
            let backup_path = config_path.with_extension("toml.backup");
            std::fs::copy(&config_path, &backup_path)?;
            output::info(&format!("Backed up existing config to {:?}", backup_path));
        }

        // Write new configuration
        let toml_content = toml::to_string_pretty(&config)?;
        std::fs::write(&config_path, toml_content)?;

        success("Configuration imported successfully");
        print_kv("Source", &input);
        print_kv("Destination", &config_path.display().to_string());
        print_kv("Format", format);
        println!();
        output::info("Restart the daemon to apply changes: ipfrs daemon restart");
    }

    Ok(())
}

/// Edit configuration file
pub async fn config_edit() -> Result<()> {
    let config_path = Config::default_path()?;

    if !config_path.exists() {
        error("Configuration file does not exist");
        println!();
        output::info("Initialize a new configuration:");
        println!("  ipfrs init");
        return Ok(());
    }

    // Get editor from environment or use default
    let editor = std::env::var("EDITOR")
        .or_else(|_| std::env::var("VISUAL"))
        .unwrap_or_else(|_| "vi".to_string());

    println!("Opening configuration in {}...", editor);
    print_kv("Config file", &config_path.display().to_string());
    println!();

    // Open editor
    let status = std::process::Command::new(&editor)
        .arg(&config_path)
        .status()
        .with_context(|| format!("Failed to open editor: {}", editor))?;

    if status.success() {
        success("Configuration editor closed");
        println!();
        output::info("Restart the daemon to apply changes: ipfrs daemon restart");
    } else {
        error("Editor exited with error");
    }

    Ok(())
}

/// Helper function to format bytes
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}
