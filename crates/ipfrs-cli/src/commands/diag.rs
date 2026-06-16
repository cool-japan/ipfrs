//! `ipfrs diag` — Node diagnostics command.
//!
//! Collects and prints a snapshot of the running node's health, resource
//! usage, and subsystem statistics.  If the daemon is offline a minimal
//! offline report is shown instead.

use anyhow::Result;

// ---------------------------------------------------------------------------
// DiagReport
// ---------------------------------------------------------------------------

/// Diagnostic snapshot collected from a running (or offline) node.
#[derive(Debug, serde::Serialize)]
pub struct DiagReport {
    /// Whether the daemon process was reachable when the report was generated.
    pub daemon_running: bool,
    /// Libp2p peer ID string (e.g. `"12D3KooW…"`).
    pub peer_id: Option<String>,
    /// Number of peers currently connected.
    pub connected_peers: Option<usize>,
    /// Total number of blocks in storage.
    pub storage_blocks: Option<u64>,
    /// Total bytes occupying storage.
    pub storage_bytes: Option<u64>,
    /// Approximate in-process memory usage in bytes.
    pub memory_bytes: Option<u64>,
    /// Average inference latency across recent queries (milliseconds).
    pub avg_inference_ms: Option<f64>,
    /// Semantic cache hit rate in [0, 1].
    pub cache_hit_rate: Option<f64>,
    /// Number of TensorLogic rules loaded.
    pub tensorlogic_rules: Option<usize>,
    /// Number of TensorLogic facts loaded.
    pub tensorlogic_facts: Option<usize>,
    /// Number of HNSW vectors indexed.
    pub hnsw_vectors: Option<u64>,
    /// Node uptime in seconds.
    pub uptime_secs: Option<u64>,
}

impl DiagReport {
    /// Build a minimal report for when the daemon is not running.
    ///
    /// `daemon_running` is set to `false`; every optional field is `None`.
    pub fn offline() -> Self {
        Self {
            daemon_running: false,
            peer_id: None,
            connected_peers: None,
            storage_blocks: None,
            storage_bytes: None,
            memory_bytes: None,
            avg_inference_ms: None,
            cache_hit_rate: None,
            tensorlogic_rules: None,
            tensorlogic_facts: None,
            hnsw_vectors: None,
            uptime_secs: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/// Format a byte count into a human-readable string (B / KB / MB / GB).
fn fmt_bytes(bytes: u64) -> String {
    const KB: u64 = 1_024;
    const MB: u64 = KB * 1_024;
    const GB: u64 = MB * 1_024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Format an uptime given in seconds into `Xh Ym Zs`.
fn fmt_uptime(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 {
        format!("{h}h {m}m")
    } else if m > 0 {
        format!("{m}m {s}s")
    } else {
        format!("{s}s")
    }
}

/// Format a large integer with thousands separators.
fn fmt_count(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}

/// Print the human-readable text report to stdout.
fn print_text_report(report: &DiagReport) {
    println!("=== IPFRS Node Diagnostics ===");

    let daemon_str = if report.daemon_running {
        "running"
    } else {
        "not running"
    };
    println!("{:<20} {}", "Daemon:", daemon_str);

    if let Some(ref pid) = report.peer_id {
        println!("{:<20} {}", "Peer ID:", pid);
    }

    if let Some(peers) = report.connected_peers {
        println!("{:<20} {}", "Connected peers:", peers);
    }

    if let Some(blocks) = report.storage_blocks {
        println!("{:<20} {}", "Storage blocks:", fmt_count(blocks));
    }

    if let Some(bytes) = report.storage_bytes {
        println!("{:<20} {}", "Storage bytes:", fmt_bytes(bytes));
    }

    if let Some(mem) = report.memory_bytes {
        println!("{:<20} {}", "Memory:", fmt_bytes(mem));
    }

    if let Some(ms) = report.avg_inference_ms {
        println!("{:<20} {:.1} ms", "Avg inference:", ms);
    }

    if let Some(rate) = report.cache_hit_rate {
        println!("{:<20} {:.1}%", "Cache hit rate:", rate * 100.0);
    }

    if let (Some(rules), Some(facts)) = (report.tensorlogic_rules, report.tensorlogic_facts) {
        println!("{:<20} {} rules, {} facts", "TensorLogic:", rules, facts);
    }

    if let Some(vecs) = report.hnsw_vectors {
        println!("{:<20} {}", "HNSW vectors:", fmt_count(vecs));
    }

    if let Some(secs) = report.uptime_secs {
        println!("{:<20} {}", "Uptime:", fmt_uptime(secs));
    }
}

// ---------------------------------------------------------------------------
// handle_diag
// ---------------------------------------------------------------------------

/// Handle the `ipfrs diag` subcommand.
///
/// Attempts to start a node locally to collect live diagnostics.  If the
/// daemon is not running (or the node cannot be initialised) an offline report
/// is printed instead.
///
/// # Arguments
///
/// * `json_output` — when `true` the report is serialised as JSON; otherwise
///   a human-readable table is printed.
pub async fn handle_diag(json_output: bool) -> Result<()> {
    // Attempt to bring up a transient node to collect diagnostics.
    let report = match try_collect_diagnostics().await {
        Ok(r) => r,
        Err(_) => {
            if !json_output {
                eprintln!("Daemon not running");
            }
            DiagReport::offline()
        }
    };

    if json_output {
        let json = serde_json::to_string_pretty(&report)?;
        println!("{json}");
    } else {
        print_text_report(&report);
    }

    Ok(())
}

/// Inner helper: start a node, query it, stop it, return the [`DiagReport`].
async fn try_collect_diagnostics() -> Result<DiagReport> {
    use ipfrs::{Node, NodeConfig};

    let mut node = Node::new(NodeConfig::default())?;
    node.start().await?;

    let diag = node.diagnostics().await?;
    let peer_id = node.peer_id().ok();

    node.stop().await?;

    let storage_blocks = Some(diag.storage.total_blocks);
    let storage_bytes = Some(diag.storage.total_bytes);
    let memory_bytes = Some(diag.resources.memory_bytes);
    let uptime_secs = Some(diag.uptime.as_secs());

    let (avg_inference_ms, tensorlogic_rules, tensorlogic_facts) =
        if let Some(ref tl) = diag.tensorlogic {
            (tl.avg_inference_ms, Some(tl.num_rules), Some(tl.num_facts))
        } else {
            (None, None, None)
        };

    let (cache_hit_rate, hnsw_vectors) = if let Some(ref sem) = diag.semantic {
        (sem.cache_hit_rate, Some(sem.num_vectors as u64))
    } else {
        (None, None)
    };

    let connected_peers = diag.network.as_ref().map(|n| n.connected_peers);

    Ok(DiagReport {
        daemon_running: true,
        peer_id,
        connected_peers,
        storage_blocks,
        storage_bytes,
        memory_bytes,
        avg_inference_ms,
        cache_hit_rate,
        tensorlogic_rules,
        tensorlogic_facts,
        hnsw_vectors,
        uptime_secs,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offline_report_all_none() {
        let r = DiagReport::offline();
        assert!(!r.daemon_running);
        assert!(r.peer_id.is_none());
        assert!(r.connected_peers.is_none());
        assert!(r.storage_blocks.is_none());
        assert!(r.storage_bytes.is_none());
        assert!(r.memory_bytes.is_none());
        assert!(r.avg_inference_ms.is_none());
        assert!(r.cache_hit_rate.is_none());
        assert!(r.tensorlogic_rules.is_none());
        assert!(r.tensorlogic_facts.is_none());
        assert!(r.hnsw_vectors.is_none());
        assert!(r.uptime_secs.is_none());
    }

    #[test]
    fn test_diag_report_json_serialization() {
        let r = DiagReport {
            daemon_running: true,
            peer_id: Some("12D3KooWTest".to_owned()),
            connected_peers: Some(3),
            storage_blocks: Some(1_247),
            storage_bytes: Some(44_369_510),
            memory_bytes: Some(134_700_032),
            avg_inference_ms: Some(12.3),
            cache_hit_rate: Some(0.874),
            tensorlogic_rules: Some(42),
            tensorlogic_facts: Some(156),
            hnsw_vectors: Some(10_000),
            uptime_secs: Some(12_240),
        };

        let json = serde_json::to_string(&r).expect("serialization failed");
        let back: serde_json::Value = serde_json::from_str(&json).expect("deserialization failed");

        assert_eq!(back["daemon_running"], true);
        assert_eq!(back["peer_id"], "12D3KooWTest");
        assert_eq!(back["connected_peers"], 3);
        assert_eq!(back["storage_blocks"], 1_247);
        assert_eq!(back["hnsw_vectors"], 10_000);
        assert_eq!(back["tensorlogic_rules"], 42);
        assert_eq!(back["tensorlogic_facts"], 156);
    }

    #[test]
    fn test_diag_report_debug() {
        let r = DiagReport::offline();
        let dbg = format!("{r:?}");
        assert!(dbg.contains("daemon_running: false"));
    }

    #[test]
    fn test_fmt_bytes() {
        assert_eq!(fmt_bytes(512), "512 B");
        assert_eq!(fmt_bytes(1_536), "1.5 KB");
        assert_eq!(fmt_bytes(44_369_510), "42.3 MB");
        assert_eq!(fmt_bytes(2_147_483_648), "2.0 GB");
    }

    #[test]
    fn test_fmt_uptime() {
        assert_eq!(fmt_uptime(30), "30s");
        assert_eq!(fmt_uptime(90), "1m 30s");
        assert_eq!(fmt_uptime(12_240), "3h 24m");
    }

    #[test]
    fn test_fmt_count() {
        assert_eq!(fmt_count(0), "0");
        assert_eq!(fmt_count(1_247), "1,247");
        assert_eq!(fmt_count(10_000_000), "10,000,000");
    }
}
