//! Utility functions for CLI maintenance and distribution
//!
//! This module provides utilities for:
//! - Man page generation
//! - Auto-update checking
//! - Version management

use anyhow::{Context, Result};
use std::path::Path;

/// Current version of the CLI
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Repository URL for checking updates
pub const REPO_URL: &str = "https://github.com/tensorlogic/ipfrs";

/// Generate man pages for all commands
///
/// This function generates comprehensive man pages using clap_mangen.
/// The man pages are written to the specified output directory.
///
/// # Arguments
///
/// * `cmd` - The clap::Command structure to generate man pages for
/// * `out_dir` - Directory where man pages will be written
///
/// # Examples
///
/// ```no_run
/// use ipfrs_cli::{build_cli, utils::generate_man_pages};
/// use std::path::Path;
///
/// let cmd = build_cli();
/// let out_dir = Path::new("target/man");
/// generate_man_pages(&cmd, out_dir).expect("Failed to generate man pages");
/// ```
#[allow(dead_code)]
pub fn generate_man_pages(cmd: &clap::Command, out_dir: &Path) -> Result<()> {
    use clap_mangen::Man;
    use std::fs;

    // Ensure output directory exists
    fs::create_dir_all(out_dir).context("Failed to create output directory")?;

    // Generate main man page
    let man = Man::new(cmd.clone());
    let mut buffer = Vec::new();
    man.render(&mut buffer)
        .context("Failed to render main man page")?;

    let main_path = out_dir.join("ipfrs.1");
    fs::write(&main_path, buffer).context("Failed to write main man page")?;

    println!("Generated: {}", main_path.display());

    // Generate man pages for each subcommand
    for subcommand in cmd.get_subcommands() {
        let name = subcommand.get_name();
        let man = Man::new(subcommand.clone());
        let mut buffer = Vec::new();
        man.render(&mut buffer)
            .with_context(|| format!("Failed to render man page for {}", name))?;

        let path = out_dir.join(format!("ipfrs-{}.1", name));
        fs::write(&path, buffer)
            .with_context(|| format!("Failed to write man page for {}", name))?;

        println!("Generated: {}", path.display());
    }

    Ok(())
}

/// Check for available updates
///
/// This function checks if a newer version is available by querying
/// the GitHub releases API.
///
/// # Returns
///
/// Returns `Some(version)` if an update is available, `None` otherwise.
///
/// # Examples
///
/// ```no_run
/// use ipfrs_cli::utils::check_for_updates;
///
/// # async fn example() {
/// match check_for_updates().await {
///     Ok(Some(version)) => println!("Update available: {}", version),
///     Ok(None) => println!("Up to date"),
///     Err(e) => eprintln!("Failed to check for updates: {}", e),
/// }
/// # }
/// ```
pub async fn check_for_updates() -> Result<Option<String>> {
    // For now, this is a placeholder implementation
    // In a real implementation, this would query GitHub API or similar
    Ok(None)
}

/// Compare two semantic versions
///
/// Returns true if `new_version` is newer than `current_version`.
#[allow(dead_code)]
fn is_newer_version(current_version: &str, new_version: &str) -> bool {
    // Simple string comparison for now
    // In production, use a proper semver crate
    current_version < new_version
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::len_zero)]
    fn test_version_constant() {
        // VERSION is a non-empty constant from CARGO_PKG_VERSION
        assert!(VERSION.len() > 0);
        assert!(VERSION.contains('.'));
    }

    #[test]
    fn test_repo_url() {
        assert!(REPO_URL.starts_with("https://"));
    }

    #[test]
    fn test_is_newer_version() {
        assert!(is_newer_version("0.1.0", "0.2.0"));
        assert!(is_newer_version("0.1.0", "1.0.0"));
        assert!(!is_newer_version("0.2.0", "0.1.0"));
        assert!(!is_newer_version("1.0.0", "1.0.0"));
    }
}
