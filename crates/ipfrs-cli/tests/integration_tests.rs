//! Integration tests for IPFRS CLI
//!
//! These tests verify the complete command pipeline and interactions

use std::process::Command;

/// Helper function to run the ipfrs binary with arguments
fn run_ipfrs(args: &[&str]) -> std::process::Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_ipfrs"));
    for arg in args {
        cmd.arg(arg);
    }
    cmd.output().expect("Failed to execute command")
}

/// Helper to check if output contains expected string
fn output_contains(output: &std::process::Output, expected: &str) -> bool {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    stdout.contains(expected) || stderr.contains(expected)
}

#[test]
fn test_version_command() {
    let output = run_ipfrs(&["version"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "ipfrs"));
}

#[test]
fn test_help_command() {
    let output = run_ipfrs(&["--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "IPFRS"));
    assert!(output_contains(&output, "Usage:") || output_contains(&output, "Commands:"));
}

#[test]
fn test_help_short() {
    let output = run_ipfrs(&["-h"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "IPFRS"));
}

#[test]
fn test_version_short() {
    let output = run_ipfrs(&["-V"]);
    assert!(output.status.success());
}

#[test]
fn test_invalid_command() {
    let output = run_ipfrs(&["nonexistent_command"]);
    assert!(!output.status.success());
}

#[test]
fn test_add_missing_argument() {
    let output = run_ipfrs(&["add"]);
    assert!(!output.status.success());
}

#[test]
fn test_get_missing_argument() {
    let output = run_ipfrs(&["get"]);
    assert!(!output.status.success());
}

#[test]
fn test_cat_missing_argument() {
    let output = run_ipfrs(&["cat"]);
    assert!(!output.status.success());
}

#[test]
fn test_block_help() {
    let output = run_ipfrs(&["block", "--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "Block operations") || output_contains(&output, "block"));
}

#[test]
fn test_daemon_help() {
    let output = run_ipfrs(&["daemon", "--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "daemon") || output_contains(&output, "Daemon"));
}

#[test]
fn test_swarm_help() {
    let output = run_ipfrs(&["swarm", "--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "swarm") || output_contains(&output, "Swarm"));
}

#[test]
fn test_dht_help() {
    let output = run_ipfrs(&["dht", "--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "DHT") || output_contains(&output, "dht"));
}

#[test]
fn test_pin_help() {
    let output = run_ipfrs(&["pin", "--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "Pin") || output_contains(&output, "pin"));
}

#[test]
fn test_repo_help() {
    let output = run_ipfrs(&["repo", "--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "repo") || output_contains(&output, "Repository"));
}

#[test]
fn test_tensor_help() {
    let output = run_ipfrs(&["tensor", "--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "tensor") || output_contains(&output, "Tensor"));
}

#[test]
fn test_logic_help() {
    let output = run_ipfrs(&["logic", "--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "logic") || output_contains(&output, "Logic"));
}

#[test]
fn test_semantic_help() {
    let output = run_ipfrs(&["semantic", "--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "semantic") || output_contains(&output, "Semantic"));
}

#[test]
fn test_model_help() {
    let output = run_ipfrs(&["model", "--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "model") || output_contains(&output, "Model"));
}

#[test]
fn test_gradient_help() {
    let output = run_ipfrs(&["gradient", "--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "gradient") || output_contains(&output, "Gradient"));
}

#[test]
fn test_dag_help() {
    let output = run_ipfrs(&["dag", "--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "DAG") || output_contains(&output, "dag"));
}

#[test]
fn test_bootstrap_help() {
    let output = run_ipfrs(&["bootstrap", "--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "bootstrap") || output_contains(&output, "Bootstrap"));
}

#[test]
fn test_stats_help() {
    let output = run_ipfrs(&["stats", "--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "stats") || output_contains(&output, "statistics"));
}

#[test]
fn test_completions_bash() {
    let output = run_ipfrs(&["completions", "bash"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "ipfrs") || output_contains(&output, "_ipfrs"));
}

#[test]
fn test_completions_zsh() {
    let output = run_ipfrs(&["completions", "zsh"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "ipfrs") || output_contains(&output, "_ipfrs"));
}

#[test]
fn test_completions_fish() {
    let output = run_ipfrs(&["completions", "fish"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "ipfrs"));
}

#[test]
fn test_verbose_flag() {
    let output = run_ipfrs(&["--verbose", "version"]);
    assert!(output.status.success());
}

#[test]
fn test_no_color_flag() {
    let output = run_ipfrs(&["--no-color", "version"]);
    assert!(output.status.success());
}

#[test]
fn test_init_help() {
    let output = run_ipfrs(&["init", "--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "Initialize") || output_contains(&output, "init"));
}

#[test]
fn test_add_help() {
    let output = run_ipfrs(&["add", "--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "Add") || output_contains(&output, "file"));
}

#[test]
fn test_get_help() {
    let output = run_ipfrs(&["get", "--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "Get") || output_contains(&output, "file"));
}

#[test]
fn test_cat_help() {
    let output = run_ipfrs(&["cat", "--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "Output") || output_contains(&output, "contents"));
}

#[test]
fn test_ls_help() {
    let output = run_ipfrs(&["ls", "--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "List") || output_contains(&output, "directory"));
}

#[test]
fn test_id_help() {
    let output = run_ipfrs(&["id", "--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "peer") || output_contains(&output, "ID"));
}

#[test]
fn test_ping_help() {
    let output = run_ipfrs(&["ping", "--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "Ping") || output_contains(&output, "peer"));
}

#[test]
fn test_shell_help() {
    let output = run_ipfrs(&["shell", "--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "shell") || output_contains(&output, "interactive"));
}

#[test]
fn test_tui_help() {
    let output = run_ipfrs(&["tui", "--help"]);
    assert!(output.status.success());
    assert!(
        output_contains(&output, "Terminal User Interface")
            || output_contains(&output, "dashboard")
    );
}

// Error handling tests

#[test]
fn test_block_get_missing_cid() {
    let output = run_ipfrs(&["block", "get"]);
    assert!(!output.status.success());
}

#[test]
fn test_block_put_missing_path() {
    let output = run_ipfrs(&["block", "put"]);
    assert!(!output.status.success());
}

#[test]
fn test_block_stat_missing_cid() {
    let output = run_ipfrs(&["block", "stat"]);
    assert!(!output.status.success());
}

#[test]
fn test_block_rm_missing_cid() {
    let output = run_ipfrs(&["block", "rm"]);
    assert!(!output.status.success());
}

#[test]
fn test_ping_missing_peer() {
    let output = run_ipfrs(&["ping"]);
    assert!(!output.status.success());
}

#[test]
fn test_ls_missing_cid() {
    let output = run_ipfrs(&["ls"]);
    assert!(!output.status.success());
}

// Regression tests - ensure backward compatibility

#[test]
fn test_version_backward_compat() {
    // Version command should always work
    let output = run_ipfrs(&["version"]);
    assert!(output.status.success());
}

#[test]
fn test_help_backward_compat() {
    // Help should work with both --help and -h
    let output1 = run_ipfrs(&["--help"]);
    let output2 = run_ipfrs(&["-h"]);
    assert!(output1.status.success());
    assert!(output2.status.success());
}

#[test]
fn test_init_default_dir_backward_compat() {
    // Init should use .ipfrs as default
    let output = run_ipfrs(&["init", "--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, ".ipfrs"));
}

#[test]
fn test_format_flag_backward_compat() {
    // Format flag should be available on relevant commands
    let output = run_ipfrs(&["add", "--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "format") || output_contains(&output, "json"));
}

// Command structure validation tests

#[test]
fn test_all_subcommands_have_help() {
    let commands = vec![
        "init",
        "add",
        "get",
        "cat",
        "ls",
        "block",
        "list",
        "stats",
        "version",
        "id",
        "ping",
        "swarm",
        "dht",
        "bootstrap",
        "logic",
        "semantic",
        "dag",
        "pin",
        "repo",
        "tensor",
        "model",
        "gradient",
        "daemon",
        "shell",
        "tui",
        "plugin",
        "completions",
    ];

    for cmd in commands {
        let output = run_ipfrs(&[cmd, "--help"]);
        assert!(
            output.status.success(),
            "Command {} should have --help",
            cmd
        );
    }
}

#[test]
fn test_plugin_help() {
    let output = run_ipfrs(&["plugin", "--help"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "Plugin"));
}

#[test]
fn test_plugin_list_command() {
    let output = run_ipfrs(&["plugin", "list"]);
    // Should succeed even if no plugins are found
    assert!(output.status.success());
}

#[test]
fn test_plugin_info_missing_argument() {
    let output = run_ipfrs(&["plugin", "info"]);
    // Should fail when plugin name is missing
    assert!(!output.status.success());
}

#[test]
fn test_plugin_run_missing_argument() {
    let output = run_ipfrs(&["plugin", "run"]);
    // Should fail when plugin name is missing
    assert!(!output.status.success());
}
