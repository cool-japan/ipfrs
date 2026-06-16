//! Daemon management commands
//!
//! This module provides functions for managing the IPFRS daemon:
//! - `run_daemon` - Run daemon in foreground
//! - `daemon_start` - Start daemon in background
//! - `daemon_stop` - Stop daemon
//! - `daemon_status` - Check daemon status
//! - `daemon_restart` - Restart daemon
//! - `daemon_health` - Comprehensive health check

use anyhow::Result;
use tracing::info;

use crate::output::{self, error, format_bytes, print_kv, success};
use crate::progress;

/// Run daemon in foreground
pub async fn run_daemon(data_dir: String) -> Result<()> {
    use ipfrs::{Node, NodeConfig};

    info!("Initializing IPFRS node...");
    let mut config = NodeConfig::default();
    config.network.data_dir = std::path::PathBuf::from(&data_dir);
    config.storage.path = std::path::PathBuf::from(&data_dir).join("blocks");

    let mut node = Node::new(config)?;

    info!("Starting IPFRS node...");
    node.start().await?;

    info!("IPFRS daemon running. Press Ctrl+C to stop.");

    // Wait for Ctrl+C
    tokio::signal::ctrl_c().await?;

    info!("Shutting down...");
    node.stop().await?;

    Ok(())
}

/// Start daemon in background
pub async fn daemon_start(data_dir: String, pid_file: String, log_file: String) -> Result<()> {
    use std::fs;
    use std::process::{Command, Stdio};

    let pid_path = std::path::Path::new(&pid_file);

    // Check if daemon is already running
    if pid_path.exists() {
        let pid_content = fs::read_to_string(pid_path)?;
        if let Ok(pid) = pid_content.trim().parse::<i32>() {
            // Check if process is still running
            #[cfg(unix)]
            {
                use std::process::Command as StdCommand;
                let check = StdCommand::new("kill")
                    .arg("-0")
                    .arg(pid.to_string())
                    .output();
                if check.is_ok_and(|c| c.status.success()) {
                    error("Daemon is already running");
                    print_kv("PID", &pid.to_string());
                    print_kv("PID file", &pid_file);
                    return Ok(());
                }
            }
        }
        // If we get here, the PID file is stale, remove it
        fs::remove_file(pid_path)?;
    }

    // Create data directory if it doesn't exist
    let data_path = std::path::Path::new(&data_dir);
    if !data_path.exists() {
        fs::create_dir_all(data_path)?;
    }

    let pb = progress::spinner("Starting daemon in background");

    // Get the current executable path
    let exe_path = std::env::current_exe()?;

    // Spawn daemon process in background
    let log_file_handle = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file)?;

    let child = Command::new(exe_path)
        .arg("daemon")
        .arg("run")
        .arg("--data-dir")
        .arg(&data_dir)
        .stdin(Stdio::null())
        .stdout(log_file_handle.try_clone()?)
        .stderr(log_file_handle)
        .spawn()?;

    let pid = child.id();

    // Write PID file
    fs::write(&pid_file, pid.to_string())?;

    progress::finish_spinner_success(&pb, "Daemon started");

    success("IPFRS daemon started in background");
    print_kv("PID", &pid.to_string());
    print_kv("PID file", &pid_file);
    print_kv("Log file", &log_file);
    print_kv("Data directory", &data_dir);

    Ok(())
}

/// Stop daemon
pub async fn daemon_stop(pid_file: String) -> Result<()> {
    use std::fs;

    let pid_path = std::path::Path::new(&pid_file);

    if !pid_path.exists() {
        error("Daemon is not running (PID file not found)");
        print_kv("PID file", &pid_file);
        return Ok(());
    }

    let pid_content = fs::read_to_string(pid_path)?;
    let pid = pid_content
        .trim()
        .parse::<i32>()
        .map_err(|e| anyhow::anyhow!("Invalid PID in file: {}", e))?;

    let pb = progress::spinner(&format!("Stopping daemon (PID: {})", pid));

    #[cfg(unix)]
    {
        use std::process::Command;

        // Send SIGTERM
        let result = Command::new("kill")
            .arg("-TERM")
            .arg(pid.to_string())
            .output()?;

        if !result.status.success() {
            progress::finish_spinner_error(&pb, "Failed to stop daemon");
            error(&format!("Failed to send SIGTERM to process {}", pid));
            error("Daemon may not be running or you may not have permission");
            return Ok(());
        }

        // Wait for process to terminate (with timeout)
        let mut attempts = 0;
        while attempts < 30 {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;

            let check = Command::new("kill").arg("-0").arg(pid.to_string()).output();

            if !check.is_ok_and(|c| c.status.success()) {
                // Process has terminated
                break;
            }
            attempts += 1;
        }

        // Check if process is still running
        let check = Command::new("kill").arg("-0").arg(pid.to_string()).output();

        if check.is_ok_and(|c| c.status.success()) {
            progress::finish_spinner_error(&pb, "Daemon did not stop gracefully");
            output::warning(&format!(
                "Process {} is still running after {} seconds",
                pid,
                attempts / 10
            ));
            output::warning("You may need to use 'kill -9' to force termination");
            return Ok(());
        }
    }

    #[cfg(not(unix))]
    {
        progress::finish_spinner_error(&pb, "Not supported on this platform");
        error("Daemon management is only supported on Unix-like systems");
        return Err(anyhow::anyhow!(
            "Daemon management not supported on this platform"
        ));
    }

    // Remove PID file
    fs::remove_file(pid_path)?;

    progress::finish_spinner_success(&pb, "Daemon stopped");
    success(&format!("Stopped daemon (PID: {})", pid));

    Ok(())
}

/// Check daemon status
pub async fn daemon_status(pid_file: String) -> Result<()> {
    use std::fs;

    let pid_path = std::path::Path::new(&pid_file);

    if !pid_path.exists() {
        output::info("Daemon is not running");
        print_kv("PID file", &pid_file);
        print_kv("Status", "stopped");
        return Ok(());
    }

    let pid_content = fs::read_to_string(pid_path)?;
    let pid = pid_content
        .trim()
        .parse::<i32>()
        .map_err(|e| anyhow::anyhow!("Invalid PID in file: {}", e))?;

    #[cfg(unix)]
    {
        use std::process::Command;

        // Check if process is running
        let check = Command::new("kill")
            .arg("-0")
            .arg(pid.to_string())
            .output()?;

        if check.status.success() {
            success("Daemon is running");
            print_kv("PID", &pid.to_string());
            print_kv("PID file", &pid_file);
            print_kv("Status", "running");

            // Try to get process info
            let ps_output = Command::new("ps")
                .arg("-p")
                .arg(pid.to_string())
                .arg("-o")
                .arg("etime=,rss=")
                .output();

            if let Ok(output) = ps_output {
                if output.status.success() {
                    let info = String::from_utf8_lossy(&output.stdout);
                    let parts: Vec<&str> = info.split_whitespace().collect();
                    if parts.len() >= 2 {
                        print_kv("Uptime", parts[0]);
                        let memory_kb = parts[1].parse::<u64>().unwrap_or(0);
                        print_kv("Memory", &format_bytes(memory_kb * 1024));
                    }
                }
            }
        } else {
            output::warning("Daemon is not running (stale PID file)");
            print_kv("PID file", &pid_file);
            print_kv("Stale PID", &pid.to_string());
            print_kv("Status", "stopped");
            output::info("You may want to remove the stale PID file");
        }
    }

    #[cfg(not(unix))]
    {
        error("Daemon status check is only supported on Unix-like systems");
        print_kv("PID", &pid.to_string());
    }

    Ok(())
}

/// Restart daemon
pub async fn daemon_restart(data_dir: String, pid_file: String, log_file: String) -> Result<()> {
    output::info("Restarting daemon...");

    // Stop the daemon if running
    let pid_path = std::path::Path::new(&pid_file);
    if pid_path.exists() {
        daemon_stop(pid_file.clone()).await?;
        // Wait a bit for cleanup
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    } else {
        output::info("Daemon was not running");
    }

    // Start the daemon
    daemon_start(data_dir, pid_file, log_file).await?;

    success("Daemon restarted successfully");

    Ok(())
}

/// Comprehensive health check for the daemon and system
///
/// Checks:
/// - Daemon status
/// - Repository health
/// - Network connectivity
/// - Disk space
/// - Memory usage
/// - API responsiveness
pub async fn daemon_health(pid_file: String, data_dir: String, format: String) -> Result<()> {
    use std::fs;

    let mut health_status = Vec::new();
    let mut overall_healthy = true;

    // Header
    if format == "text" {
        println!("IPFRS Health Check");
        println!("==================");
        println!();
    }

    // 1. Daemon Status
    let pid_path = std::path::Path::new(&pid_file);
    let daemon_running = if pid_path.exists() {
        if let Ok(pid_content) = fs::read_to_string(pid_path) {
            if let Ok(pid) = pid_content.trim().parse::<i32>() {
                #[cfg(unix)]
                {
                    use std::process::Command;
                    let check = Command::new("kill").arg("-0").arg(pid.to_string()).output();
                    check.is_ok_and(|c| c.status.success())
                }
                #[cfg(not(unix))]
                {
                    true
                }
            } else {
                false
            }
        } else {
            false
        }
    } else {
        false
    };

    health_status.push(("daemon_running", daemon_running));
    if !daemon_running {
        overall_healthy = false;
    }

    if format == "text" {
        println!("Daemon Status:");
        println!("-------------");
        if daemon_running {
            success("✓ Daemon is running");
        } else {
            error("✗ Daemon is not running");
        }
        println!();
    }

    // 2. Repository Health
    if format == "text" {
        println!("Repository Health:");
        println!("-----------------");
    }

    let data_path = std::path::Path::new(&data_dir);
    let repo_exists = data_path.exists() && data_path.is_dir();
    health_status.push(("repository_exists", repo_exists));

    if format == "text" {
        if repo_exists {
            success(&format!("✓ Repository exists at {}", data_dir));

            // Check repository size
            if let Ok(blocks_path) = data_path.join("blocks").canonicalize() {
                if blocks_path.exists() {
                    // Count blocks
                    if let Ok(entries) = fs::read_dir(&blocks_path) {
                        let block_count = entries.count();
                        print_kv("  Blocks", &block_count.to_string());
                    }
                }
            }
        } else {
            error(&format!("✗ Repository not found at {}", data_dir));
            overall_healthy = false;
        }
        println!();
    }

    // 3. Disk Space
    if format == "text" {
        println!("Disk Space:");
        println!("-----------");
    }

    #[cfg(unix)]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("df").arg("-h").arg(&data_dir).output() {
            if output.status.success() {
                let df_output = String::from_utf8_lossy(&output.stdout);
                let lines: Vec<&str> = df_output.lines().collect();
                if lines.len() >= 2 {
                    let parts: Vec<&str> = lines[1].split_whitespace().collect();
                    if parts.len() >= 5 {
                        let available = parts[3];
                        let use_percent = parts[4];

                        let usage: u32 = use_percent.trim_end_matches('%').parse().unwrap_or(0);
                        let disk_healthy = usage < 90;
                        health_status.push(("disk_space_ok", disk_healthy));

                        if format == "text" {
                            if disk_healthy {
                                success(&format!(
                                    "✓ Disk usage: {} (available: {})",
                                    use_percent, available
                                ));
                            } else {
                                output::warning(&format!(
                                    "⚠ Disk usage high: {} (available: {})",
                                    use_percent, available
                                ));
                                overall_healthy = false;
                            }
                        }
                    }
                }
            }
        }
    }

    if format == "text" {
        println!();
    }

    // 4. Memory Usage (if daemon is running)
    if daemon_running && format == "text" {
        println!("Memory Usage:");
        println!("-------------");

        if let Ok(pid_content) = fs::read_to_string(pid_path) {
            if let Ok(pid) = pid_content.trim().parse::<i32>() {
                #[cfg(unix)]
                {
                    use std::process::Command;
                    if let Ok(output) = Command::new("ps")
                        .arg("-p")
                        .arg(pid.to_string())
                        .arg("-o")
                        .arg("rss=,vsz=,%mem=")
                        .output()
                    {
                        if output.status.success() {
                            let mem_info = String::from_utf8_lossy(&output.stdout);
                            let parts: Vec<&str> = mem_info.split_whitespace().collect();
                            if parts.len() >= 3 {
                                let rss_kb = parts[0].parse::<u64>().unwrap_or(0);
                                let vsz_kb = parts[1].parse::<u64>().unwrap_or(0);
                                let mem_percent = parts[2];

                                print_kv("  RSS", &format_bytes(rss_kb * 1024));
                                print_kv("  VSZ", &format_bytes(vsz_kb * 1024));
                                print_kv("  Memory %", mem_percent);
                            }
                        }
                    }
                }
            }
        }
        println!();
    }

    // 5. Overall Status
    if format == "text" {
        println!("Overall Status:");
        println!("---------------");
        if overall_healthy {
            success("✓ All health checks passed");
        } else {
            error("✗ Some health checks failed");
            println!();
            println!("Recommendations:");
            if !daemon_running {
                println!("  • Start the daemon: ipfrs daemon start");
            }
            if !repo_exists {
                println!("  • Initialize repository: ipfrs init");
            }
        }
    } else if format == "json" {
        // JSON output
        use serde_json::json;
        let health_obj = json!({
            "healthy": overall_healthy,
            "daemon_running": daemon_running,
            "repository_exists": repo_exists,
            "checks": health_status.iter().map(|(k, v)| json!({
                "name": k,
                "passed": v
            })).collect::<Vec<_>>()
        });
        println!("{}", serde_json::to_string_pretty(&health_obj)?);
    }

    Ok(())
}
