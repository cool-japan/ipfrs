//! Prometheus metrics commands
//!
//! Provides the `ipfrs metrics` subcommand family:
//!
//! - `ipfrs metrics show`  — Print all node metrics in Prometheus text format.
//! - `ipfrs metrics reset` — Reset all per-node counters to zero (no-op for the
//!   global registry; instructs the user to restart the daemon for a full reset).

use anyhow::Result;

use crate::commands::query::OutputFormat;
use crate::output::{self, print_header};

/// Handle `ipfrs metrics show`.
///
/// Fetches the Prometheus text output from the `ipfrs-interface` global
/// registry and prints it to stdout.
///
/// When `format` is [`OutputFormat::Json`], a minimal JSON envelope is emitted
/// instead of raw Prometheus text so the output can be consumed by tools that
/// expect JSON.
pub async fn handle_metrics_show(format: &OutputFormat) -> Result<()> {
    let text = ipfrs_interface::metrics::encode_metrics()
        .map_err(|e| anyhow::anyhow!("Failed to encode metrics: {}", e))?;

    if format.is_json() {
        // Wrap in a simple JSON object so callers can parse it uniformly.
        println!("{{");
        println!("  \"format\": \"prometheus_text\",");
        // Escape the inner text for JSON embedding.
        let escaped = text
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n");
        println!("  \"metrics\": \"{}\"", escaped);
        println!("}}");
    } else {
        print_header("IPFRS Prometheus Metrics");
        print!("{}", text);
    }

    Ok(())
}

/// Handle `ipfrs metrics reset`.
///
/// Prometheus counters are monotonically increasing and cannot be reset at
/// runtime without a process restart.  This command prints an informative
/// message and exits successfully so automation pipelines are not broken.
pub async fn handle_metrics_reset() -> Result<()> {
    output::info(
        "Prometheus counters are monotonically increasing and cannot be reset at runtime.\n\
         To reset all metrics, restart the IPFRS daemon process.",
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_show_text() {
        // Should not return an error even when no metrics have been recorded.
        let result = handle_metrics_show(&OutputFormat::Text).await;
        assert!(result.is_ok(), "metrics show (text) failed: {:?}", result);
    }

    #[tokio::test]
    async fn test_metrics_show_json() {
        let result = handle_metrics_show(&OutputFormat::Json).await;
        assert!(result.is_ok(), "metrics show (json) failed: {:?}", result);
    }

    #[tokio::test]
    async fn test_metrics_reset() {
        let result = handle_metrics_reset().await;
        assert!(result.is_ok(), "metrics reset failed: {:?}", result);
    }
}
