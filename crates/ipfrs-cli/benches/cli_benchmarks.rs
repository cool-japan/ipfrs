//! Benchmarks for IPFRS CLI performance
//!
//! Measures command parsing, argument validation, and overall CLI responsiveness

use criterion::{criterion_group, criterion_main, Criterion};
use ipfrs_cli::config::Config;
use std::hint::black_box;
use std::process::Command;

/// Benchmark command parsing performance
fn bench_command_parsing(c: &mut Criterion) {
    c.bench_function("parse_version_command", |b| {
        b.iter(|| {
            let _ = Command::new(env!("CARGO_BIN_EXE_ipfrs"))
                .arg("version")
                .output();
        });
    });

    c.bench_function("parse_help_command", |b| {
        b.iter(|| {
            let _ = Command::new(env!("CARGO_BIN_EXE_ipfrs"))
                .arg("--help")
                .output();
        });
    });

    c.bench_function("parse_add_command", |b| {
        b.iter(|| {
            let _ = Command::new(env!("CARGO_BIN_EXE_ipfrs"))
                .arg("add")
                .arg("--help")
                .output();
        });
    });
}

/// Benchmark help text generation
fn bench_help_generation(c: &mut Criterion) {
    c.bench_function("generate_main_help", |b| {
        b.iter(|| {
            let _ = Command::new(env!("CARGO_BIN_EXE_ipfrs"))
                .arg("--help")
                .output()
                .expect("Failed to generate help");
        });
    });

    c.bench_function("generate_subcommand_help", |b| {
        b.iter(|| {
            let _ = Command::new(env!("CARGO_BIN_EXE_ipfrs"))
                .arg("block")
                .arg("--help")
                .output()
                .expect("Failed to generate help");
        });
    });
}

/// Benchmark completion generation
fn bench_completion_generation(c: &mut Criterion) {
    c.bench_function("generate_bash_completion", |b| {
        b.iter(|| {
            let _ = Command::new(env!("CARGO_BIN_EXE_ipfrs"))
                .arg("completions")
                .arg("bash")
                .output();
        });
    });

    c.bench_function("generate_zsh_completion", |b| {
        b.iter(|| {
            let _ = Command::new(env!("CARGO_BIN_EXE_ipfrs"))
                .arg("completions")
                .arg("zsh")
                .output();
        });
    });
}

/// Benchmark argument validation
fn bench_argument_validation(c: &mut Criterion) {
    c.bench_function("validate_missing_argument", |b| {
        b.iter(|| {
            let _ = Command::new(env!("CARGO_BIN_EXE_ipfrs"))
                .arg("add")
                .output();
        });
    });

    c.bench_function("validate_invalid_command", |b| {
        b.iter(|| {
            let _ = Command::new(env!("CARGO_BIN_EXE_ipfrs"))
                .arg("nonexistent_command")
                .output();
        });
    });
}

/// Benchmark overall CLI startup time
fn bench_cli_startup(c: &mut Criterion) {
    c.bench_function("cli_startup_version", |b| {
        b.iter(|| {
            let output = Command::new(env!("CARGO_BIN_EXE_ipfrs"))
                .arg("version")
                .output()
                .expect("Failed to execute");
            black_box(output);
        });
    });

    c.bench_function("cli_startup_with_verbose", |b| {
        b.iter(|| {
            let output = Command::new(env!("CARGO_BIN_EXE_ipfrs"))
                .arg("--verbose")
                .arg("version")
                .output()
                .expect("Failed to execute");
            black_box(output);
        });
    });
}

/// Benchmark config loading performance
fn bench_config_loading(c: &mut Criterion) {
    c.bench_function("config_load_cached", |b| {
        b.iter(|| {
            let config = Config::load().expect("Failed to load config");
            black_box(config);
        });
    });

    c.bench_function("config_load_uncached", |b| {
        b.iter(|| {
            let config = Config::load_uncached().expect("Failed to load config");
            black_box(config);
        });
    });
}

/// Benchmark TUI operations
fn bench_tui_operations(c: &mut Criterion) {
    c.bench_function("tui_help_generation", |b| {
        b.iter(|| {
            let output = Command::new(env!("CARGO_BIN_EXE_ipfrs"))
                .arg("tui")
                .arg("--help")
                .output()
                .expect("Failed to generate TUI help");
            black_box(output);
        });
    });
}

criterion_group!(
    benches,
    bench_command_parsing,
    bench_help_generation,
    bench_completion_generation,
    bench_argument_validation,
    bench_cli_startup,
    bench_config_loading,
    bench_tui_operations
);
criterion_main!(benches);
