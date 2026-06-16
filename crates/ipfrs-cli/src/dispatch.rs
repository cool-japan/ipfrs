//! Command dispatch helpers for IPFRS CLI
//!
//! This module contains handler functions that are called from `main()` for
//! commands that require non-trivial logic or significant imports.  Splitting
//! them here keeps `main.rs` under the 2 000-line threshold.

use anyhow::Result;

// ============================================================================
// Validation Helpers
// ============================================================================

/// Validate that a CID string has valid format
#[allow(dead_code)]
pub fn validate_cid_format(cid_str: &str) -> Result<()> {
    if cid_str.is_empty() {
        return Err(anyhow::anyhow!("CID cannot be empty"));
    }

    // Check common CID prefixes
    if !cid_str.starts_with("Qm") && !cid_str.starts_with("bafy") && !cid_str.starts_with("bafk") {
        crate::output::warning(
            "CID may have invalid format. Expected to start with 'Qm', 'bafy', or 'bafk'",
        );
    }

    Ok(())
}

/// Check if a path is readable
#[allow(dead_code)]
pub fn validate_path_readable(path: &str) -> Result<()> {
    let path_obj = std::path::Path::new(path);

    if !path_obj.exists() {
        return Err(anyhow::anyhow!(
            "Path does not exist: {}\nPlease check the path and try again.",
            path
        ));
    }

    // Try to open the file to check read permissions
    if path_obj.is_file() {
        std::fs::File::open(path_obj).map_err(|e| {
            anyhow::anyhow!(
                "Cannot read file: {}\nError: {}\n\nCheck file permissions.",
                path,
                e
            )
        })?;
    }

    Ok(())
}

/// Warn about potentially large operations
#[allow(dead_code)]
pub fn check_file_size_warning(size: u64) {
    use crate::output::format_bytes;

    const LARGE_FILE_THRESHOLD: u64 = 100 * 1024 * 1024; // 100 MB
    const HUGE_FILE_THRESHOLD: u64 = 1024 * 1024 * 1024; // 1 GB

    if size > HUGE_FILE_THRESHOLD {
        crate::output::warning(&format!(
            "Very large file detected: {}. This operation may take significant time and memory.",
            format_bytes(size)
        ));
    } else if size > LARGE_FILE_THRESHOLD {
        crate::output::warning(&format!(
            "Large file detected: {}. This may take a while.",
            format_bytes(size)
        ));
    }
}

// ============================================================================
// Identity Management
// ============================================================================

/// Subcommands for `ipfrs identity` — mirrored here to break the circular
/// dependency between `main.rs` types and the handler.
///
/// The actual enum lives in `main.rs`; this function receives it as a concrete
/// type alias via the public re-export below.
pub async fn handle_identity_command(subcommand: super::IdentityCommands) -> Result<()> {
    use ipfrs_network::identity::PeerIdentityManager;

    match subcommand {
        super::IdentityCommands::Show { data_dir, format } => {
            let key_path = std::path::PathBuf::from(&data_dir).join("identity.key");
            let mgr = PeerIdentityManager::load_or_generate(&key_path)
                .map_err(|e| anyhow::anyhow!("Failed to load identity: {}", e))?;

            let peer_id = mgr.peer_id();
            let rotations = mgr.rotation_count();

            if format == "json" {
                use serde_json::json;
                let obj = json!({
                    "peer_id": peer_id.to_string(),
                    "rotation_count": rotations,
                    "key_file": key_path.display().to_string(),
                });
                println!("{}", serde_json::to_string_pretty(&obj)?);
            } else {
                crate::output::print_kv("PeerId", &peer_id.to_string());
                crate::output::print_kv("Key file", &key_path.display().to_string());
                crate::output::print_kv("Rotations", &rotations.to_string());
            }
        }

        super::IdentityCommands::Rotate { data_dir } => {
            let key_path = std::path::PathBuf::from(&data_dir).join("identity.key");
            let mut mgr = PeerIdentityManager::load_or_generate(&key_path)
                .map_err(|e| anyhow::anyhow!("Failed to load identity: {}", e))?;

            let old_peer_id = mgr.peer_id();
            let new_peer_id = mgr
                .rotate()
                .map_err(|e| anyhow::anyhow!("Failed to rotate identity: {}", e))?;

            crate::output::success("Peer identity key rotated successfully");
            crate::output::print_kv("Old PeerId", &old_peer_id.to_string());
            crate::output::print_kv("New PeerId", &new_peer_id.to_string());
            crate::output::print_kv("Rotation count", &mgr.rotation_count().to_string());
        }

        super::IdentityCommands::ExportPem { data_dir } => {
            let key_path = std::path::PathBuf::from(&data_dir).join("identity.key");
            let mgr = PeerIdentityManager::load_or_generate(&key_path)
                .map_err(|e| anyhow::anyhow!("Failed to load identity: {}", e))?;

            let pem = mgr.export_public_key_pem();
            print!("{}", pem);
        }
    }

    Ok(())
}

// ============================================================================
// Plugin Management
// ============================================================================

pub async fn handle_plugin_command(command: super::PluginCommands) -> Result<()> {
    use ipfrs_cli::config::Config;
    use ipfrs_cli::output::{error, info, success, TablePrinter};
    use ipfrs_cli::plugin::PluginManager;

    let mut manager = PluginManager::new();
    manager.discover_plugins();

    match command {
        super::PluginCommands::List => {
            let plugins = manager.list_plugins();

            if plugins.is_empty() {
                info("No plugins found.");
                println!("\nTo add plugins, place executables in:");
                println!("  ~/.ipfrs/plugins/");
                println!("\nPlugin naming convention:");
                println!("  ipfrs-plugin-<name>");
                println!("\nExample:");
                println!("  ipfrs-plugin-hello");
                return Ok(());
            }

            info(&format!("Found {} plugin(s):", plugins.len()));
            println!();

            let mut table = TablePrinter::new(vec!["Name", "Description"]);

            for name in plugins {
                let desc = manager
                    .get_plugin(name)
                    .and_then(|p| p.description())
                    .unwrap_or("No description");
                table.add_row(vec![name, desc]);
            }

            table.print();
        }
        super::PluginCommands::Info { name } => {
            if let Some(plugin) = manager.get_plugin(&name) {
                success(&format!("Plugin: {}", plugin.name()));
                println!("Path: {}", plugin.path().display());

                if let Some(desc) = plugin.description() {
                    println!("Description: {}", desc);
                } else {
                    println!("Description: No description available");
                }
            } else {
                error(&format!("Plugin '{}' not found", name));
                println!("\nAvailable plugins:");
                for plugin_name in manager.list_plugins() {
                    println!("  - {}", plugin_name);
                }
                std::process::exit(1);
            }
        }
        super::PluginCommands::Run { name, args } => {
            let config = Config::load()?;

            match manager.execute_plugin(&name, &args, &config) {
                Ok(code) => {
                    std::process::exit(code);
                }
                Err(e) => {
                    error(&format!("Failed to execute plugin '{}': {}", name, e));
                    std::process::exit(1);
                }
            }
        }
    }

    Ok(())
}

// ============================================================================
// Shell Completions
// ============================================================================

pub fn generate_completions(shell: super::CompletionShell) {
    use clap::CommandFactory;
    use clap_complete::{generate, Shell};

    let mut cmd = super::Cli::command();
    let bin_name = "ipfrs";

    let shell = match shell {
        super::CompletionShell::Bash => Shell::Bash,
        super::CompletionShell::Zsh => Shell::Zsh,
        super::CompletionShell::Fish => Shell::Fish,
        super::CompletionShell::PowerShell => Shell::PowerShell,
        super::CompletionShell::Elvish => Shell::Elvish,
    };

    generate(shell, &mut cmd, bin_name, &mut std::io::stdout());
}
