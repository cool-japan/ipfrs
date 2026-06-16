//! Output formatting and colored output utilities for IPFRS CLI
//!
//! This module provides utilities for formatting CLI output with colors,
//! tables, and different output modes (text, JSON, compact).
//!
//! # Examples
//!
//! ```rust
//! use ipfrs_cli::output::{OutputStyle, format_bytes, format_bytes_detailed};
//!
//! // Create output style
//! let style = OutputStyle::new(true, "text");
//!
//! // Format file sizes
//! let size = format_bytes(1048576);
//! assert_eq!(size, "1.00 MB");
//!
//! let detailed = format_bytes_detailed(1234567);
//! assert_eq!(detailed, "1.18 MB (1234567 bytes)");
//! ```

#![allow(dead_code)]

use colored::Colorize;
use std::io::{self, Write};

/// Check if stdout is a TTY (terminal)
///
/// # Examples
///
/// ```rust
/// use ipfrs_cli::output::is_tty;
///
/// // In tests, this typically returns false
/// let tty = is_tty();
/// assert!(tty == true || tty == false); // Platform dependent
/// ```
pub fn is_tty() -> bool {
    use std::io::IsTerminal;
    std::io::stdout().is_terminal()
}

/// Output style configuration for controlling formatting and colors
///
/// # Examples
///
/// ```rust
/// use ipfrs_cli::output::OutputStyle;
///
/// // Create style with colors enabled
/// let style = OutputStyle::new(true, "text");
/// assert_eq!(style.format, "text");
///
/// // Create compact style
/// let compact = OutputStyle::new(false, "compact");
/// assert!(compact.is_compact());
///
/// // JSON format disables colors
/// let json = OutputStyle::new(true, "json");
/// assert_eq!(json.format, "json");
/// ```
pub struct OutputStyle {
    /// Enable colored output
    pub color: bool,
    /// Output format (text, json, compact)
    pub format: String,
    /// Compact mode (minimal output)
    pub compact: bool,
    /// Quiet mode (suppress non-essential output)
    pub quiet: bool,
}

impl Default for OutputStyle {
    fn default() -> Self {
        Self {
            color: is_tty(),
            format: "text".to_string(),
            compact: false,
            quiet: false,
        }
    }
}

impl OutputStyle {
    /// Create new output style with color control
    pub fn new(color: bool, format: &str) -> Self {
        // Disable colors if not TTY or if format is JSON
        let effective_color = color && is_tty() && format != "json" && format != "compact";
        let compact = format == "compact";
        Self {
            color: effective_color,
            format: format.to_string(),
            compact,
            quiet: false,
        }
    }

    /// Create new output style with quiet mode
    pub fn with_quiet(color: bool, format: &str, quiet: bool) -> Self {
        let mut style = Self::new(color, format);
        style.quiet = quiet;
        style
    }

    /// Check if compact mode is enabled
    pub fn is_compact(&self) -> bool {
        self.compact || self.format == "compact"
    }

    /// Check if quiet mode is enabled
    pub fn is_quiet(&self) -> bool {
        self.quiet
    }
}

/// Print a success message
pub fn success(msg: &str) {
    if is_tty() {
        println!("{} {}", "✓".green().bold(), msg.green());
    } else {
        println!("{}", msg);
    }
}

/// Print an error message
pub fn error(msg: &str) {
    if is_tty() {
        eprintln!("{} {}", "✗".red().bold(), msg.red());
    } else {
        eprintln!("error: {}", msg);
    }
}

/// Print a warning message
pub fn warning(msg: &str) {
    if is_tty() {
        eprintln!("{} {}", "!".yellow().bold(), msg.yellow());
    } else {
        eprintln!("warning: {}", msg);
    }
}

/// Print an info message
pub fn info(msg: &str) {
    if is_tty() {
        println!("{} {}", "ℹ".blue().bold(), msg);
    } else {
        println!("{}", msg);
    }
}

/// Print a CID (content identifier) with highlighting
pub fn print_cid(label: &str, cid: &str) {
    if is_tty() {
        println!("{}: {}", label, cid.cyan().bold());
    } else {
        println!("{}: {}", label, cid);
    }
}

/// Print a key-value pair
pub fn print_kv(key: &str, value: &str) {
    if is_tty() {
        println!("  {}: {}", key.dimmed(), value);
    } else {
        println!("  {}: {}", key, value);
    }
}

/// Print a header/title
pub fn print_header(title: &str) {
    if is_tty() {
        println!("{}", title.bold().underline());
    } else {
        println!("{}", title);
        println!("{}", "=".repeat(title.len()));
    }
}

/// Print a section header
pub fn print_section(title: &str) {
    if is_tty() {
        println!("\n{}", title.bold());
    } else {
        println!("\n{}", title);
    }
}

/// Format bytes as human-readable size
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Format bytes with both human-readable and exact value
pub fn format_bytes_detailed(bytes: u64) -> String {
    if bytes >= 1024 {
        format!("{} ({} bytes)", format_bytes(bytes), bytes)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Print a list item
pub fn print_list_item(item: &str) {
    if is_tty() {
        println!("  {} {}", "•".dimmed(), item);
    } else {
        println!("  - {}", item);
    }
}

/// Print a numbered list item
pub fn print_numbered_item(num: usize, item: &str) {
    if is_tty() {
        println!("  {}. {}", num.to_string().dimmed(), item);
    } else {
        println!("  {}. {}", num, item);
    }
}

/// Table printer for formatted output
pub struct TablePrinter {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    column_widths: Vec<usize>,
}

impl TablePrinter {
    /// Create a new table with headers
    pub fn new(headers: Vec<&str>) -> Self {
        let headers: Vec<String> = headers.iter().map(|s| s.to_string()).collect();
        let column_widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
        Self {
            headers,
            rows: Vec::new(),
            column_widths,
        }
    }

    /// Add a row to the table
    pub fn add_row(&mut self, row: Vec<&str>) {
        let row: Vec<String> = row.iter().map(|s| s.to_string()).collect();
        for (i, cell) in row.iter().enumerate() {
            if i < self.column_widths.len() {
                self.column_widths[i] = self.column_widths[i].max(cell.len());
            }
        }
        self.rows.push(row);
    }

    /// Print the table to stdout
    pub fn print(&self) {
        let color = is_tty();

        // Print header
        let header_line: String = self
            .headers
            .iter()
            .enumerate()
            .map(|(i, h)| format!("{:width$}", h, width = self.column_widths[i]))
            .collect::<Vec<_>>()
            .join("  ");

        if color {
            println!("{}", header_line.bold());
        } else {
            println!("{}", header_line);
        }

        // Print separator
        let separator: String = self
            .column_widths
            .iter()
            .map(|&w| "-".repeat(w))
            .collect::<Vec<_>>()
            .join("  ");

        if color {
            println!("{}", separator.dimmed());
        } else {
            println!("{}", separator);
        }

        // Print rows
        for row in &self.rows {
            let row_line: String = row
                .iter()
                .enumerate()
                .map(|(i, cell)| {
                    let width = self.column_widths.get(i).copied().unwrap_or(cell.len());
                    format!("{:width$}", cell, width = width)
                })
                .collect::<Vec<_>>()
                .join("  ");
            println!("{}", row_line);
        }
    }
}

/// Write raw bytes to stdout (for binary output)
pub fn write_raw(data: &[u8]) -> io::Result<()> {
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    handle.write_all(data)?;
    handle.flush()
}

/// Print in compact mode (minimal output)
pub fn compact_print(key: &str, value: &str) {
    println!("{}:{}", key, value);
}

/// Print CID in compact mode
pub fn compact_cid(cid: &str) {
    println!("{}", cid);
}

/// Print list in compact mode (one item per line)
pub fn compact_list(items: &[String]) {
    for item in items {
        println!("{}", item);
    }
}

/// Print key-value pairs in compact mode
pub fn compact_kv_pairs(pairs: &[(&str, &str)]) {
    for (key, value) in pairs {
        println!("{}:{}", key, value);
    }
}

/// Type of query result
#[derive(Debug, Clone)]
pub enum QueryResultType {
    SemanticMatch,
    LogicBinding,
    HybridMatch,
}

/// Unified query result for semantic, logic, and hybrid queries
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub result_type: QueryResultType,
    pub cid: Option<String>,
    pub score: Option<f32>,
    pub bindings: std::collections::HashMap<String, String>,
    pub metadata: std::collections::HashMap<String, String>,
}

impl QueryResult {
    /// Format as a JSON object string
    pub fn to_json(&self) -> String {
        let mut parts = Vec::new();
        if let Some(cid) = &self.cid {
            parts.push(format!("\"cid\": \"{}\"", cid));
        }
        if let Some(score) = self.score {
            parts.push(format!("\"score\": {:.4}", score));
        }
        if !self.bindings.is_empty() {
            let bindings_str = self
                .bindings
                .iter()
                .map(|(k, v)| format!("\"{}\": \"{}\"", k, v))
                .collect::<Vec<_>>()
                .join(", ");
            parts.push(format!("\"bindings\": {{{}}}", bindings_str));
        }
        let result_type = match self.result_type {
            QueryResultType::SemanticMatch => "semantic",
            QueryResultType::LogicBinding => "logic",
            QueryResultType::HybridMatch => "hybrid",
        };
        parts.push(format!("\"type\": \"{}\"", result_type));
        format!("{{{}}}", parts.join(", "))
    }
}

/// Print troubleshooting hint for common errors
///
/// This function provides helpful troubleshooting messages for common error scenarios
pub fn troubleshooting_hint(error_type: &str) {
    let hint = match error_type {
        "daemon_not_running" => {
            "The IPFRS daemon is not running.\n\
             To start the daemon, run: ipfrs daemon start\n\
             Or run in foreground: ipfrs daemon"
        }
        "daemon_already_running" => {
            "The IPFRS daemon is already running.\n\
             To stop it, run: ipfrs daemon stop\n\
             To check status: ipfrs daemon status"
        }
        "repo_not_initialized" => {
            "IPFRS repository not initialized.\n\
             To initialize a repository, run: ipfrs init\n\
             Or specify a custom directory: ipfrs init -d /path/to/repo"
        }
        "connection_failed" => {
            "Failed to connect to peer.\n\
             Troubleshooting steps:\n\
             1. Check if the peer is online\n\
             2. Verify the multiaddr format is correct\n\
             3. Check your network connection\n\
             4. Ensure firewall allows IPFRS connections"
        }
        "cid_not_found" => {
            "Content not found.\n\
             This could mean:\n\
             1. The CID is incorrect or malformed\n\
             2. The content is not available on the network\n\
             3. You need to connect to more peers\n\
             Try: ipfrs swarm peers (to check connections)"
        }
        "permission_denied" => {
            "Permission denied.\n\
             Troubleshooting steps:\n\
             1. Check file/directory permissions\n\
             2. Ensure you have write access to the data directory\n\
             3. Try running with appropriate permissions"
        }
        "config_error" => {
            "Configuration error.\n\
             Troubleshooting steps:\n\
             1. Check config file syntax (TOML format)\n\
             2. Verify config file location: ~/.config/ipfrs/config.toml\n\
             3. Reset to defaults: rm ~/.config/ipfrs/config.toml && ipfrs init"
        }
        "network_timeout" => {
            "Network operation timed out.\n\
             Troubleshooting steps:\n\
             1. Check your internet connection\n\
             2. Try connecting to bootstrap peers\n\
             3. Increase timeout in config file\n\
             4. Check if peers are reachable: ipfrs ping <peer-id>"
        }
        _ => "For more help, visit: https://github.com/tensorlogic/ipfrs/issues",
    };

    if is_tty() {
        println!("\n{} {}", "Hint:".yellow().bold(), hint);
    } else {
        println!("\nHint: {}", hint);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
    }

    #[test]
    fn test_format_bytes_detailed() {
        assert_eq!(format_bytes_detailed(512), "512 bytes");
        assert_eq!(format_bytes_detailed(1024), "1.00 KB (1024 bytes)");
    }

    #[test]
    fn test_table_printer() {
        let mut table = TablePrinter::new(vec!["Name", "Size", "CID"]);
        table.add_row(vec!["file.txt", "1024", "Qm..."]);
        table.add_row(vec!["data.bin", "2048", "Qm..."]);
        // Just verify it doesn't panic
        table.print();
    }

    #[test]
    fn test_output_style_quiet_mode() {
        let style = OutputStyle::with_quiet(true, "text", true);
        assert!(style.is_quiet());
        assert!(!style.is_compact());
    }

    #[test]
    fn test_output_style_quiet_mode_disabled() {
        let style = OutputStyle::with_quiet(true, "text", false);
        assert!(!style.is_quiet());
    }

    #[test]
    fn test_output_style_quiet_with_json() {
        let style = OutputStyle::with_quiet(true, "json", true);
        assert!(style.is_quiet());
        assert_eq!(style.format, "json");
    }

    #[test]
    fn test_output_style_quiet_default() {
        let style = OutputStyle::default();
        assert!(!style.is_quiet());
    }
}
