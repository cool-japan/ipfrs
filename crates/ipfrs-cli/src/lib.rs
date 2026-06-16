//! IPFRS CLI Library
//!
//! This library provides the core functionality for the IPFRS command-line interface.
//! While primarily used by the binary, it also exposes utilities for output formatting,
//! configuration management, and interactive shell support.
//!
//! # Modules
//!
//! - [`commands`] - Command handler implementations (modular refactoring)
//! - [`config`] - Configuration file management and settings (with caching)
//! - [`connectivity`] - Fast offline/daemon detection (< 2 s)
//! - [`output`] - Output formatting with colors and tables
//! - [`plugin`] - Plugin system for extending CLI with custom commands
//! - [`progress`] - Progress indicators for long-running operations
//! - [`shell`] - Interactive REPL shell implementation
//! - [`tui`] - Terminal User Interface dashboard
//! - [`utils`] - Utility functions for maintenance and updates
//!
//! # Performance Optimizations
//!
//! The CLI has been optimized for fast startup and low latency:
//!
//! - **Config Caching**: Configuration files are loaded once and cached globally
//!   using [`std::sync::OnceLock`] to avoid repeated disk I/O.
//! - **Lazy Initialization**: Heavy modules are only loaded when needed.
//! - **Minimal Dependencies**: Core functionality uses lightweight dependencies.
//!
//! # Examples
//!
//! ```rust
//! use ipfrs_cli::output::{format_bytes, OutputStyle};
//!
//! // Format file sizes
//! let size = format_bytes(1048576);
//! assert_eq!(size, "1.00 MB");
//!
//! // Create output style
//! let style = OutputStyle::new(true, "text");
//! assert_eq!(style.format, "text");
//! ```
//!
//! ## Configuration Management
//!
//! ```rust
//! use ipfrs_cli::config::Config;
//!
//! // Load config with caching (fast on subsequent calls)
//! let config = Config::load().expect("config should load successfully");
//! assert_eq!(config.general.log_level, "info");
//!
//! // Force fresh load without cache
//! let fresh_config = Config::load_uncached().expect("config should load successfully");
//! ```

pub mod commands;
pub mod config;
pub mod connectivity;
pub mod output;
pub mod plugin;
pub mod progress;
pub mod shell;
pub mod tui;
pub mod utils;

/// Build the CLI command structure
///
/// This function creates the complete clap command structure for IPFRS CLI.
/// It's used by both the main binary and utilities like man page generation.
///
/// # Returns
///
/// Returns a fully configured `clap::Command` with all subcommands and options.
///
/// # Examples
///
/// ```
/// use ipfrs_cli::build_cli;
///
/// let cli = build_cli();
/// assert_eq!(cli.get_name(), "ipfrs");
/// ```
pub fn build_cli() -> clap::Command {
    use clap::{Arg, Command};

    Command::new("ipfrs")
        .version(env!("CARGO_PKG_VERSION"))
        .about("IPFRS - Inter-Planet File RUST System")
        .long_about(
            "IPFRS - Inter-Planet File RUST System\n\n\
            A next-generation content-addressed storage system with built-in support for\n\
            tensors, logic programming, and semantic search. IPFRS extends IPFS concepts\n\
            with advanced features for AI/ML workloads and distributed data management.\n\n\
            Examples:\n  \
            ipfrs init                    Initialize a new repository\n  \
            ipfrs add file.txt            Add a file to IPFRS\n  \
            ipfrs get <cid>               Retrieve content by CID\n  \
            ipfrs daemon                  Start the IPFRS daemon\n  \
            ipfrs shell                   Start interactive shell",
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Enable verbose logging (shows debug information)")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("no-color")
                .long("no-color")
                .help("Disable colored output (useful for scripts and non-TTY environments)")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("quiet")
                .short('q')
                .long("quiet")
                .help(
                    "Quiet mode - suppress non-essential output (useful for scripts and pipelines)",
                )
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("FILE")
                .help("Path to configuration file (overrides default config location)")
                .action(clap::ArgAction::Set),
        )
        // Add basic subcommands for man page generation
        // (The main binary has the full implementation)
        .subcommand(Command::new("init").about("Initialize an IPFRS repository"))
        .subcommand(Command::new("add").about("Add file to IPFRS"))
        .subcommand(Command::new("cat").about("Output file contents by CID"))
        .subcommand(Command::new("get").about("Download file to filesystem"))
        .subcommand(Command::new("ls").about("List directory contents"))
        .subcommand(
            Command::new("block")
                .about("Manage raw blocks")
                .subcommand(Command::new("get").about("Get raw block"))
                .subcommand(Command::new("put").about("Put raw block"))
                .subcommand(Command::new("stat").about("Block statistics"))
                .subcommand(Command::new("rm").about("Remove block")),
        )
        .subcommand(
            Command::new("daemon")
                .about("Manage IPFRS daemon")
                .subcommand(Command::new("start").about("Start daemon in background"))
                .subcommand(Command::new("stop").about("Stop background daemon"))
                .subcommand(Command::new("status").about("Check daemon status"))
                .subcommand(Command::new("restart").about("Restart daemon")),
        )
        .subcommand(
            Command::new("dag")
                .about("Manage DAG nodes")
                .subcommand(Command::new("get").about("Get DAG node"))
                .subcommand(Command::new("put").about("Put DAG node"))
                .subcommand(Command::new("resolve").about("Resolve IPLD path"))
                .subcommand(Command::new("export").about("Export DAG to CAR"))
                .subcommand(Command::new("import").about("Import DAG from CAR")),
        )
        .subcommand(
            Command::new("swarm")
                .about("Manage swarm connections")
                .subcommand(Command::new("peers").about("List connected peers"))
                .subcommand(Command::new("connect").about("Connect to peer"))
                .subcommand(Command::new("disconnect").about("Disconnect from peer"))
                .subcommand(Command::new("addrs").about("List listening addresses")),
        )
        .subcommand(
            Command::new("dht")
                .about("DHT operations")
                .subcommand(Command::new("findprovs").about("Find providers"))
                .subcommand(Command::new("findpeer").about("Find peer address"))
                .subcommand(Command::new("provide").about("Announce provider")),
        )
        .subcommand(Command::new("id").about("Show node identity"))
        .subcommand(Command::new("version").about("Show version info"))
        .subcommand(
            Command::new("stats")
                .about("Show statistics")
                .subcommand(Command::new("repo").about("Repository statistics"))
                .subcommand(Command::new("bw").about("Bandwidth statistics"))
                .subcommand(Command::new("bitswap").about("Bitswap statistics")),
        )
        .subcommand(
            Command::new("pin")
                .about("Manage pinned content")
                .subcommand(Command::new("add").about("Pin content"))
                .subcommand(Command::new("rm").about("Unpin content"))
                .subcommand(Command::new("ls").about("List pins"))
                .subcommand(Command::new("verify").about("Verify pin integrity")),
        )
        .subcommand(
            Command::new("repo")
                .about("Manage repository")
                .subcommand(Command::new("gc").about("Run garbage collection"))
                .subcommand(Command::new("stat").about("Repository statistics"))
                .subcommand(Command::new("fsck").about("Verify repository"))
                .subcommand(Command::new("version").about("Repository version")),
        )
        .subcommand(
            Command::new("tensor")
                .about("Manage tensors")
                .subcommand(Command::new("add").about("Add tensor"))
                .subcommand(Command::new("get").about("Get tensor"))
                .subcommand(Command::new("info").about("Tensor metadata"))
                .subcommand(Command::new("export").about("Export tensor format")),
        )
        .subcommand(
            Command::new("logic")
                .about("Logic programming")
                .subcommand(Command::new("infer").about("Run inference query"))
                .subcommand(Command::new("prove").about("Show proof tree"))
                .subcommand(Command::new("kb-stats").about("Knowledge base statistics"))
                .subcommand(Command::new("kb-save").about("Save knowledge base"))
                .subcommand(Command::new("kb-load").about("Load knowledge base")),
        )
        .subcommand(
            Command::new("semantic")
                .about("Semantic search")
                .subcommand(Command::new("search").about("Vector search"))
                .subcommand(Command::new("index").about("Manual indexing"))
                .subcommand(Command::new("similar").about("Find similar"))
                .subcommand(Command::new("stats").about("Index statistics"))
                .subcommand(Command::new("save").about("Save semantic index"))
                .subcommand(Command::new("load").about("Load semantic index")),
        )
        .subcommand(
            Command::new("model")
                .about("Model management")
                .subcommand(Command::new("add").about("Add model directory"))
                .subcommand(Command::new("checkpoint").about("Create snapshot"))
                .subcommand(Command::new("diff").about("Compare models"))
                .subcommand(Command::new("rollback").about("Restore version")),
        )
        .subcommand(
            Command::new("gradient")
                .about("Gradient operations")
                .subcommand(Command::new("push").about("Publish gradient"))
                .subcommand(Command::new("pull").about("Fetch gradient"))
                .subcommand(Command::new("aggregate").about("Federated learning"))
                .subcommand(Command::new("history").about("View updates")),
        )
        .subcommand(
            Command::new("bootstrap")
                .about("Manage bootstrap peers")
                .subcommand(Command::new("list").about("List bootstrap peers"))
                .subcommand(Command::new("add").about("Add bootstrap peer"))
                .subcommand(Command::new("rm").about("Remove bootstrap peer")),
        )
        .subcommand(Command::new("ping").about("Ping peer"))
        .subcommand(Command::new("shell").about("Start interactive shell"))
        .subcommand(Command::new("tui").about("Start Terminal User Interface dashboard"))
        .subcommand(
            Command::new("plugin")
                .about("Plugin management")
                .long_about(
                    "Manage and execute IPFRS CLI plugins.\n\n\
                    Plugins are executables that extend the CLI with custom commands.\n\
                    Place plugins in ~/.ipfrs/plugins/ with the naming convention:\n  \
                    ipfrs-plugin-<name>\n\n\
                    Examples:\n  \
                    ipfrs plugin list              List all available plugins\n  \
                    ipfrs plugin info <name>       Show plugin information\n  \
                    ipfrs plugin run <name> ...    Execute a plugin",
                )
                .subcommand(Command::new("list").about("List available plugins"))
                .subcommand(
                    Command::new("info")
                        .about("Show plugin information")
                        .arg(Arg::new("name").required(true).help("Plugin name")),
                )
                .subcommand(
                    Command::new("run")
                        .about("Execute a plugin")
                        .arg(Arg::new("name").required(true).help("Plugin name"))
                        .arg(
                            Arg::new("args")
                                .num_args(0..)
                                .help("Plugin arguments")
                                .trailing_var_arg(true),
                        ),
                ),
        )
}
