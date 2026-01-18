//! Man page generation utility for IPFRS CLI
//!
//! This binary generates man pages for all IPFRS commands using clap_mangen.
//! Run this during the build process to generate documentation for installation.
//!
//! Usage:
//!   cargo run --bin ipfrs-genman -- target/man

use anyhow::Result;
use ipfrs_cli::{build_cli, utils::generate_man_pages};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    // Get output directory from command line or use default
    let out_dir = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/man"));

    println!("Generating man pages to: {}", out_dir.display());

    // Build the CLI command structure
    let cmd = build_cli();

    generate_man_pages(&cmd, &out_dir)?;

    println!("\nMan pages generated successfully!");
    println!("To install system-wide, run:");
    println!("  sudo cp {}/*.1 /usr/share/man/man1/", out_dir.display());
    println!("  sudo mandb");

    Ok(())
}
