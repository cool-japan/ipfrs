//! Progress bar and spinner utilities for IPFRS CLI

#![allow(dead_code)]

use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

/// Minimum file size (10 MB) for which a progress bar is shown.
///
/// Files smaller than this threshold use a hidden progress bar so that
/// the caller code can call `pb.inc()` without any visible output.
pub const LARGE_FILE_THRESHOLD: u64 = 10 * 1024 * 1024; // 10 MB

/// Create a progress bar for a file operation that is conditionally visible.
///
/// The progress bar is **hidden** (no terminal output) when:
/// - `total_bytes` is below [`LARGE_FILE_THRESHOLD`] (10 MB), or
/// - stdout is not a TTY (e.g. when piped to another process or script).
///
/// For large files on a TTY the bar shows bytes transferred, rate, and ETA,
/// matching the style used across all other IPFRS file operations.
///
/// # Arguments
///
/// * `total_bytes` – Total expected size of the transfer in bytes.
/// * `operation`   – Short verb shown at the start of the bar line (e.g. `"Adding"`, `"Downloading"`).
pub fn file_progress_bar(total_bytes: u64, operation: &str) -> ProgressBar {
    // Hide for small files or non-interactive output.
    use std::io::IsTerminal;
    let is_tty = std::io::stdout().is_terminal();
    if total_bytes < LARGE_FILE_THRESHOLD || !is_tty {
        return ProgressBar::hidden();
    }

    let pb = ProgressBar::new(total_bytes);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(&format!(
                "{{spinner:.green}} {} [{{elapsed_precise}}] [{{bar:40.cyan/blue}}] {{bytes}}/{{total_bytes}} ({{bytes_per_sec}}, {{eta}})",
                operation
            ))
            .expect("valid progress template")
            .progress_chars("=>-"),
    );
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

/// Create a progress bar for file operations
pub fn file_progress(total: u64, message: &str) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} {msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec})")
            .expect("valid template")
            .progress_chars("=>-"),
    );
    pb.set_message(message.to_string());
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

/// Create a progress bar for block operations
pub fn block_progress(total: u64, message: &str) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} {msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} blocks")
            .expect("valid template")
            .progress_chars("=>-"),
    );
    pb.set_message(message.to_string());
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

/// Create a spinner for operations with unknown duration
pub fn spinner(message: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg} [{elapsed_precise}]")
            .expect("valid template"),
    );
    pb.set_message(message.to_string());
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

/// Create a download progress bar
pub fn download_progress(total: u64, filename: &str) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} Downloading {msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .expect("valid template")
            .progress_chars("=>-"),
    );
    pb.set_message(filename.to_string());
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

/// Create an upload progress bar
pub fn upload_progress(total: u64, filename: &str) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} Uploading {msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .expect("valid template")
            .progress_chars("=>-"),
    );
    pb.set_message(filename.to_string());
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

/// Create a multi-progress indicator for batch operations
pub fn batch_progress(total: u64, message: &str) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} {msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%)")
            .expect("valid template")
            .progress_chars("=>-"),
    );
    pb.set_message(message.to_string());
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

/// Finish progress bar with success message
pub fn finish_success(pb: &ProgressBar, message: &str) {
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg}")
            .expect("valid template"),
    );
    pb.finish_with_message(format!("\x1b[32m✓\x1b[0m {}", message));
}

/// Finish progress bar with error message
pub fn finish_error(pb: &ProgressBar, message: &str) {
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg}")
            .expect("valid template"),
    );
    pb.finish_with_message(format!("\x1b[31m✗\x1b[0m {}", message));
}

/// Finish spinner with success
pub fn finish_spinner_success(pb: &ProgressBar, message: &str) {
    pb.finish_with_message(format!("\x1b[32m✓\x1b[0m {}", message));
}

/// Finish spinner with error
pub fn finish_spinner_error(pb: &ProgressBar, message: &str) {
    pb.finish_with_message(format!("\x1b[31m✗\x1b[0m {}", message));
}

/// Progress tracker for streaming operations
pub struct StreamProgress {
    pb: ProgressBar,
    total: u64,
    current: u64,
}

impl StreamProgress {
    /// Create a new stream progress tracker
    pub fn new(total: u64, message: &str) -> Self {
        let pb = file_progress(total, message);
        Self {
            pb,
            total,
            current: 0,
        }
    }

    /// Update progress with bytes written/read
    pub fn update(&mut self, bytes: u64) {
        self.current += bytes;
        self.pb.set_position(self.current);
    }

    /// Set absolute position
    pub fn set_position(&mut self, pos: u64) {
        self.current = pos;
        self.pb.set_position(pos);
    }

    /// Get current progress percentage
    pub fn percentage(&self) -> f64 {
        if self.total == 0 {
            100.0
        } else {
            (self.current as f64 / self.total as f64) * 100.0
        }
    }

    /// Finish with success
    pub fn finish(self, message: &str) {
        finish_success(&self.pb, message);
    }

    /// Finish with error
    pub fn finish_error(self, message: &str) {
        finish_error(&self.pb, message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spinner_creation() {
        let pb = spinner("Testing...");
        pb.finish_with_message("Done");
    }

    #[test]
    fn test_progress_creation() {
        let pb = file_progress(1000, "Processing");
        pb.set_position(500);
        pb.finish_with_message("Complete");
    }

    #[test]
    fn test_stream_progress() {
        let mut sp = StreamProgress::new(100, "Streaming");
        sp.update(25);
        assert_eq!(sp.percentage(), 25.0);
        sp.update(25);
        assert_eq!(sp.percentage(), 50.0);
        sp.finish("Done");
    }
}
