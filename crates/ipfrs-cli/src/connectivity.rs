//! Fast offline detection for IPFRS CLI
//!
//! Determines whether the IPFRS daemon is running and reachable within a
//! bounded timeout (≤ 2 s).  The checks are deliberately cheap:
//!
//! 1. Look for the PID file on disk — if it is missing the daemon is
//!    definitely not running.
//! 2. Send signal 0 to the recorded PID to verify the process still exists
//!    (POSIX `kill(pid, 0)` semantic, implemented via `std::process::Command`).
//!
//! A separate helper [`with_network_timeout`] wraps any `Future` with a
//! configurable deadline so network-bound commands never block indefinitely.

use std::path::Path;
use std::time::Duration;

/// Maximum time budget for all daemon-reachability checks.
pub const DAEMON_CHECK_TIMEOUT: Duration = Duration::from_secs(2);

/// Check whether the IPFRS daemon is reachable.
///
/// Returns `true` if a running process with the recorded PID can be found,
/// `false` in every other case (missing PID file, stale PID, parse error …).
///
/// The whole operation is designed to complete in well under
/// [`DAEMON_CHECK_TIMEOUT`].
///
/// # Arguments
///
/// * `data_dir` – Path to the IPFRS data directory that contains
///   `daemon.pid` (e.g. `.ipfrs`).
pub async fn check_daemon_reachable(data_dir: &str) -> bool {
    let pid_file = Path::new(data_dir).join("daemon.pid");

    if !pid_file.exists() {
        return false;
    }

    // Read PID – keep errors silent; we just report "not reachable".
    let raw = match tokio::fs::read_to_string(&pid_file).await {
        Ok(s) => s,
        Err(_) => return false,
    };

    match raw.trim().parse::<u32>() {
        Ok(pid) => is_process_alive(pid),
        Err(_) => false,
    }
}

/// Test whether a process with the given PID is alive.
///
/// On POSIX systems this sends signal 0 (`kill -0 <pid>`).  On Windows the
/// approach is approximated by checking whether the process exists via
/// `tasklist`; on non-POSIX Unix platforms we fall back to `false`.
fn is_process_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        // SAFETY: kill(2) with signal 0 is purely a permission/existence check;
        // it does not deliver any signal.
        let output = std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .output();

        match output {
            Ok(out) => out.status.success(),
            Err(_) => false,
        }
    }

    #[cfg(windows)]
    {
        // On Windows approximate with tasklist.
        let output = std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {}", pid), "/NH"])
            .output();

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                stdout.contains(&pid.to_string())
            }
            Err(_) => false,
        }
    }

    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
        false
    }
}

/// Build a user-friendly error message for the case where the IPFRS daemon
/// is not running or not reachable.
///
/// # Arguments
///
/// * `data_dir` – Path to the IPFRS data directory (e.g. `.ipfrs`).
pub fn offline_error_message(data_dir: &str) -> String {
    format!(
        "IPFRS daemon is not running.\n\
         Start it with:       ipfrs daemon start --data-dir {data_dir}\n\
         Or run foreground:   ipfrs daemon run   --data-dir {data_dir}\n\
         Check status with:   ipfrs daemon status"
    )
}

/// Wrap a future with a wall-clock timeout.
///
/// Returns `Some(value)` if the future completes before `timeout` elapses,
/// or `None` if it times out.
///
/// # Examples
///
/// ```rust,no_run
/// use ipfrs_cli::connectivity::{with_network_timeout, DAEMON_CHECK_TIMEOUT};
///
/// async fn example() {
///     let result = with_network_timeout(
///         async { 42_u32 },
///         DAEMON_CHECK_TIMEOUT,
///     ).await;
///     assert_eq!(result, Some(42));
/// }
/// ```
pub async fn with_network_timeout<F, T>(fut: F, timeout: Duration) -> Option<T>
where
    F: std::future::Future<Output = T>,
{
    tokio::time::timeout(timeout, fut).await.ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    #[tokio::test]
    async fn test_missing_data_dir_returns_false() {
        let result = check_daemon_reachable("/nonexistent/ipfrs/data/dir").await;
        assert!(!result, "should be false when data dir does not exist");
    }

    #[tokio::test]
    async fn test_missing_pid_file_returns_false() {
        // Create a temp dir without a pid file inside.
        let dir = temp_dir().join("ipfrs_test_connectivity_no_pid");
        let _ = std::fs::create_dir_all(&dir);
        let result = check_daemon_reachable(&dir.to_string_lossy()).await;
        assert!(!result, "should be false when pid file is absent");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_stale_pid_returns_false() {
        // PID 99999999 is virtually guaranteed not to exist.
        let dir = temp_dir().join("ipfrs_test_connectivity_stale");
        let _ = std::fs::create_dir_all(&dir);
        let pid_path = dir.join("daemon.pid");
        std::fs::write(&pid_path, "99999999").expect("write pid file");
        let result = check_daemon_reachable(&dir.to_string_lossy()).await;
        assert!(!result, "should be false for a stale PID");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_with_network_timeout_completes() {
        let result = with_network_timeout(async { 7_u32 }, Duration::from_secs(1)).await;
        assert_eq!(result, Some(7));
    }

    #[tokio::test]
    async fn test_with_network_timeout_expires() {
        let result = with_network_timeout(
            async {
                tokio::time::sleep(Duration::from_secs(10)).await;
                42_u32
            },
            Duration::from_millis(50),
        )
        .await;
        assert!(result.is_none(), "should time out");
    }

    #[test]
    fn test_offline_error_message_contains_data_dir() {
        let msg = offline_error_message("/tmp/test-ipfrs");
        assert!(
            msg.contains("daemon is not running"),
            "should mention daemon not running"
        );
        assert!(
            msg.contains("/tmp/test-ipfrs"),
            "should include the data_dir path"
        );
    }

    #[test]
    fn test_offline_error_message_contains_start_command() {
        let msg = offline_error_message(".ipfrs");
        assert!(
            msg.contains("daemon start"),
            "should suggest 'daemon start'"
        );
        assert!(msg.contains("daemon run"), "should suggest 'daemon run'");
    }
}
