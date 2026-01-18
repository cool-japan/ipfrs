//! Common utilities shared across command modules
//!
//! This module provides validation helpers and utility functions
//! used by multiple command handlers.

use anyhow::Result;

use crate::output::{self, format_bytes};

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
        output::warning(
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
    const LARGE_FILE_THRESHOLD: u64 = 100 * 1024 * 1024; // 100 MB
    const HUGE_FILE_THRESHOLD: u64 = 1024 * 1024 * 1024; // 1 GB

    if size > HUGE_FILE_THRESHOLD {
        output::warning(&format!(
            "Very large file detected: {}. This operation may take significant time and memory.",
            format_bytes(size)
        ));
    } else if size > LARGE_FILE_THRESHOLD {
        output::warning(&format!(
            "Large file detected: {}. This may take a while.",
            format_bytes(size)
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_cid_format_empty() {
        assert!(validate_cid_format("").is_err());
    }

    #[test]
    fn test_validate_cid_format_valid_qm() {
        assert!(validate_cid_format("QmXvBJfNuZ9A8Y3EqJkT").is_ok());
    }

    #[test]
    fn test_validate_cid_format_valid_bafy() {
        assert!(validate_cid_format("bafybeigdyrzt5sfp7udm7").is_ok());
    }

    #[test]
    fn test_validate_path_readable_nonexistent() {
        assert!(validate_path_readable("/nonexistent/path/to/file").is_err());
    }
}
