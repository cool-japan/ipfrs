//! IPFRS CLI - Command-line interface for IPFRS
//!
//! Version: 0.3.0 "The Fast & The Wise"

// Import library modules
use ipfrs_cli::commands::*;
use ipfrs_cli::{output, shell, tui, utils};

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::info;

use output::{error, format_bytes};

/// Exit codes for shell script integration
///
/// These codes allow scripts to detect and handle different error conditions
#[allow(dead_code)]
mod exit_codes {
    /// Success
    pub const SUCCESS: i32 = 0;
    /// General error
    pub const ERROR: i32 = 1;
    /// Invalid arguments or command-line usage
    pub const USAGE_ERROR: i32 = 2;
    /// File or content not found
    pub const NOT_FOUND: i32 = 3;
    /// Permission denied or authentication failed
    pub const PERMISSION_DENIED: i32 = 4;
    /// Network or connection error
    pub const NETWORK_ERROR: i32 = 5;
    /// I/O error (file system operations)
    pub const IO_ERROR: i32 = 6;
    /// Timeout error
    pub const TIMEOUT: i32 = 7;
    /// Configuration error
    pub const CONFIG_ERROR: i32 = 8;
}

#[derive(Parser)]
#[command(name = "ipfrs")]
#[command(version = "0.3.0")]
#[command(about = "IPFRS - Inter-Planet File RUST System")]
#[command(long_about = "IPFRS - Inter-Planet File RUST System\n\n\
A next-generation content-addressed storage system with built-in support for\n\
tensors, logic programming, and semantic search. IPFRS extends IPFS concepts\n\
with advanced features for AI/ML workloads and distributed data management.\n\n\
Examples:\n  \
ipfrs init                    Initialize a new repository\n  \
ipfrs add file.txt            Add a file to IPFRS\n  \
ipfrs get <cid>               Retrieve content by CID\n  \
ipfrs daemon                  Start the IPFRS daemon\n  \
ipfrs shell                   Start interactive shell")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging (shows debug information)
    #[arg(short, long)]
    verbose: bool,

    /// Disable colored output (useful for scripts and non-TTY environments)
    #[arg(long)]
    no_color: bool,

    /// Quiet mode - suppress non-essential output (useful for scripts and pipelines)
    #[arg(short, long)]
    quiet: bool,

    /// Path to configuration file (overrides default config location)
    #[arg(short, long, value_name = "FILE")]
    config: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize an IPFRS repository in the specified directory
    #[command(
        long_about = "Initialize a new IPFRS repository with default configuration.\n\n\
        This creates the repository directory structure, generates configuration files,\n\
        and initializes the storage backend.\n\n\
        Example:\n  \
        ipfrs init                    # Initialize in .ipfrs\n  \
        ipfrs init -d /path/to/repo   # Initialize in custom directory"
    )]
    Init {
        /// Data directory where the repository will be created
        #[arg(short, long, default_value = ".ipfrs", value_name = "DIR")]
        data_dir: String,
    },

    /// Manage the IPFRS daemon process
    #[command(long_about = "Start, stop, and manage the IPFRS daemon.\n\n\
        The daemon runs in the background and handles network operations,\n\
        peer connections, and content routing.\n\n\
        Examples:\n  \
        ipfrs daemon                  # Run daemon in foreground\n  \
        ipfrs daemon start            # Start daemon in background\n  \
        ipfrs daemon stop             # Stop background daemon\n  \
        ipfrs daemon status           # Check daemon status")]
    Daemon {
        #[command(subcommand)]
        command: Option<DaemonCommands>,
    },

    /// Start HTTP Gateway server for web access
    #[command(
        long_about = "Start an HTTP gateway server for accessing IPFRS content via HTTP.\n\n\
        The gateway provides a REST API for interacting with IPFRS content\n\
        and can serve files directly over HTTP.\n\n\
        Examples:\n  \
        ipfrs gateway -l 0.0.0.0:8080\n  \
        ipfrs gateway -l 0.0.0.0:8443 --tls-cert cert.pem --tls-key key.pem"
    )]
    Gateway {
        /// Address and port to listen on
        #[arg(short, long, default_value = "127.0.0.1:8080", value_name = "ADDR")]
        listen: String,
        /// Data directory containing the IPFRS repository
        #[arg(short, long, default_value = ".ipfrs", value_name = "DIR")]
        data_dir: String,
        /// TLS certificate file path (PEM format)
        #[arg(long, value_name = "FILE")]
        tls_cert: Option<String>,
        /// TLS private key file path (PEM format)
        #[arg(long, value_name = "FILE")]
        tls_key: Option<String>,
    },

    /// Add a file or directory to IPFRS
    #[command(
        long_about = "Add a file or directory to IPFRS and return its CID.\n\n\
        The content is chunked, hashed, and stored in the local repository.\n\
        The returned CID can be used to retrieve the content later.\n\n\
        Examples:\n  \
        ipfrs add myfile.txt          # Add a file\n  \
        ipfrs add --format json dir/  # Add directory with JSON output"
    )]
    Add {
        /// Path to the file or directory to add
        #[arg(value_name = "PATH")]
        path: String,
        /// Output format: text or json
        #[arg(long, default_value = "text", value_name = "FORMAT")]
        format: String,
    },

    /// Retrieve content from IPFRS and save it to disk
    #[command(long_about = "Download and save content from IPFRS by its CID.\n\n\
        Retrieves the content from local storage or the network and saves it\n\
        to the specified output path.\n\n\
        Examples:\n  \
        ipfrs get QmHash123           # Save to file named by CID\n  \
        ipfrs get QmHash123 -o file   # Save to specific file")]
    Get {
        /// Content Identifier (CID) of the content to retrieve
        #[arg(value_name = "CID")]
        cid: String,
        /// Output file path (defaults to CID if not specified)
        #[arg(short, long, value_name = "FILE")]
        output: Option<String>,
    },

    /// Output the contents of a file to stdout
    #[command(long_about = "Retrieve and output content directly to stdout.\n\n\
        This is useful for piping content to other commands or viewing\n\
        file contents without saving to disk.\n\n\
        Examples:\n  \
        ipfrs cat QmHash123           # Output to stdout\n  \
        ipfrs cat QmHash123 | less    # Pipe to pager")]
    Cat {
        /// Content Identifier (CID) to retrieve and output
        #[arg(value_name = "CID")]
        cid: String,
    },

    /// List the contents of an IPFRS directory
    #[command(
        long_about = "List files and subdirectories in an IPFRS directory.\n\n\
        Shows file names, sizes, and types for all entries in the directory.\n\n\
        Examples:\n  \
        ipfrs ls QmDirHash            # List directory contents\n  \
        ipfrs ls QmDirHash --format json"
    )]
    Ls {
        /// Content Identifier (CID) of the directory
        #[arg(value_name = "CID")]
        cid: String,
        /// Output format: text or json
        #[arg(long, default_value = "text", value_name = "FORMAT")]
        format: String,
    },

    /// Block operations
    Block {
        #[command(subcommand)]
        command: BlockCommands,
    },

    /// List all stored blocks
    List {
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Show node statistics
    Stats {
        #[command(subcommand)]
        command: StatsCommands,
    },

    /// Show node information
    Info,

    /// Show version information
    Version,

    /// Show peer ID and addresses
    Id {
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Ping a peer
    Ping {
        /// Peer ID to ping
        peer_id: String,
        /// Number of pings to send
        #[arg(short, long, default_value = "5")]
        count: u32,
    },

    /// Swarm network operations
    Swarm {
        #[command(subcommand)]
        command: SwarmCommands,
    },

    /// DHT operations
    Dht {
        #[command(subcommand)]
        command: DhtCommands,
    },

    /// Bootstrap operations
    Bootstrap {
        #[command(subcommand)]
        command: BootstrapCommands,
    },

    /// Logic operations (TensorLogic)
    Logic {
        #[command(subcommand)]
        command: LogicCommands,
    },

    /// Semantic search operations
    Semantic {
        #[command(subcommand)]
        command: SemanticCommands,
    },

    /// DAG (Directed Acyclic Graph) operations
    Dag {
        #[command(subcommand)]
        command: DagCommands,
    },

    /// Pin management operations
    Pin {
        #[command(subcommand)]
        command: PinCommands,
    },

    /// Repository management
    Repo {
        #[command(subcommand)]
        command: RepoCommands,
    },

    /// Tensor operations
    Tensor {
        #[command(subcommand)]
        command: TensorCommands,
    },

    /// Model management operations
    Model {
        #[command(subcommand)]
        command: ModelCommands,
    },

    /// Gradient operations for federated learning
    Gradient {
        #[command(subcommand)]
        command: GradientCommands,
    },

    /// Start interactive shell (REPL)
    Shell {
        /// Data directory
        #[arg(short, long, default_value = ".ipfrs")]
        data_dir: String,
    },

    /// Start Terminal User Interface dashboard
    #[command(
        long_about = "Start an interactive terminal dashboard for monitoring IPFRS node.\n\n\
        The TUI provides real-time statistics about:\n  \
        - Connected peers and network activity\n  \
        - Storage usage and block statistics\n  \
        - Bandwidth monitoring with sparkline graphs\n  \
        - Node uptime and system health\n\n\
        Navigation:\n  \
        - Tab/Arrow keys to switch between tabs\n  \
        - 1-4 to jump to specific tabs\n  \
        - q or Ctrl+C to exit\n\n\
        Example:\n  \
        ipfrs tui                     # Launch the TUI dashboard"
    )]
    Tui,

    /// Plugin management and execution
    #[command(long_about = "Manage and execute IPFRS CLI plugins.\n\n\
        Plugins are executables that extend the CLI with custom commands.\n\
        Place plugins in ~/.ipfrs/plugins/ with naming: ipfrs-plugin-<name>\n\n\
        Examples:\n  \
        ipfrs plugin list              # List all available plugins\n  \
        ipfrs plugin info <name>       # Show plugin information\n  \
        ipfrs plugin run <name> ...    # Execute a plugin")]
    Plugin {
        #[command(subcommand)]
        command: PluginCommands,
    },

    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: CompletionShell,
    },

    /// Check for available updates (hidden command)
    #[command(hide = true)]
    Update {
        /// Check without prompting
        #[arg(long)]
        check: bool,
    },
}

/// Supported shells for completion generation
#[derive(Debug, Clone, clap::ValueEnum)]
enum CompletionShell {
    Bash,
    Zsh,
    Fish,
    PowerShell,
    Elvish,
}

#[derive(Subcommand)]
enum PluginCommands {
    /// List all available plugins
    List,

    /// Show plugin information
    Info {
        /// Plugin name
        name: String,
    },

    /// Execute a plugin
    Run {
        /// Plugin name
        name: String,
        /// Plugin arguments (everything after the plugin name)
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
}

#[derive(Subcommand)]
enum TensorCommands {
    /// Add a tensor file
    Add {
        /// Path to tensor file
        path: String,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Get a tensor by CID
    Get {
        /// Content ID (CID) of the tensor
        cid: String,
        /// Output path
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Show tensor metadata
    Info {
        /// Content ID (CID) of the tensor
        cid: String,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Export tensor to different format
    Export {
        /// Content ID (CID) of the tensor
        cid: String,
        /// Output path
        #[arg(short, long)]
        output: String,
        /// Target format (safetensors, numpy, pytorch)
        #[arg(long, default_value = "safetensors")]
        target_format: String,
    },
}

#[derive(Subcommand)]
enum DaemonCommands {
    /// Run daemon in foreground (default if no subcommand)
    Run {
        /// Data directory
        #[arg(short, long, default_value = ".ipfrs")]
        data_dir: String,
    },

    /// Start daemon in background
    Start {
        /// Data directory
        #[arg(short, long, default_value = ".ipfrs")]
        data_dir: String,
        /// PID file path
        #[arg(long, default_value = ".ipfrs/daemon.pid")]
        pid_file: String,
        /// Log file path
        #[arg(long, default_value = ".ipfrs/daemon.log")]
        log_file: String,
    },

    /// Stop daemon
    Stop {
        /// PID file path
        #[arg(long, default_value = ".ipfrs/daemon.pid")]
        pid_file: String,
    },

    /// Show daemon status
    Status {
        /// PID file path
        #[arg(long, default_value = ".ipfrs/daemon.pid")]
        pid_file: String,
    },

    /// Restart daemon
    Restart {
        /// Data directory
        #[arg(short, long, default_value = ".ipfrs")]
        data_dir: String,
        /// PID file path
        #[arg(long, default_value = ".ipfrs/daemon.pid")]
        pid_file: String,
        /// Log file path
        #[arg(long, default_value = ".ipfrs/daemon.log")]
        log_file: String,
    },

    /// Comprehensive health check
    Health {
        /// PID file path
        #[arg(long, default_value = ".ipfrs/daemon.pid")]
        pid_file: String,
        /// Data directory
        #[arg(short, long, default_value = ".ipfrs")]
        data_dir: String,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },
}

#[derive(Subcommand)]
enum SwarmCommands {
    /// List connected peers
    Peers {
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Connect to a peer
    Connect {
        /// Multiaddress to connect to
        addr: String,
    },

    /// Disconnect from a peer
    Disconnect {
        /// Peer ID to disconnect from
        peer_id: String,
    },

    /// List listening addresses
    Addrs {
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },
}

#[derive(Subcommand)]
enum DhtCommands {
    /// Find providers for a CID
    Findprovs {
        /// Content ID (CID)
        cid: String,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Announce content to DHT
    Provide {
        /// Content ID (CID)
        cid: String,
    },

    /// Find peer addresses in DHT
    Findpeer {
        /// Peer ID to find
        peer_id: String,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },
}

#[derive(Subcommand)]
enum StatsCommands {
    /// Show storage statistics
    Repo {
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Show bandwidth statistics
    Bw {
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Show bitswap statistics
    Bitswap {
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },
}

#[derive(Subcommand)]
enum BootstrapCommands {
    /// List bootstrap peers
    List {
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Add a bootstrap peer
    Add {
        /// Multiaddress of peer to add
        addr: String,
    },

    /// Remove a bootstrap peer
    Rm {
        /// Multiaddress of peer to remove
        addr: String,
    },
}

#[derive(Subcommand)]
enum BlockCommands {
    /// Get raw block data
    Get {
        /// Content ID (CID)
        cid: String,
    },

    /// Put raw block data
    Put {
        /// Path to file containing block data
        path: String,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Show block statistics
    Stat {
        /// Content ID (CID)
        cid: String,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Remove a block
    Rm {
        /// Content ID (CID)
        cid: String,
        /// Force removal without confirmation
        #[arg(short, long)]
        force: bool,
    },
}

#[derive(Subcommand)]
enum LogicCommands {
    /// Run an inference query
    Infer {
        /// Predicate name
        predicate: String,
        /// Terms (as JSON strings)
        #[arg(short, long)]
        terms: Vec<String>,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Generate a proof for a goal
    Prove {
        /// Predicate name
        predicate: String,
        /// Terms (as JSON strings)
        #[arg(short, long)]
        terms: Vec<String>,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Show knowledge base statistics
    KbStats {
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Save knowledge base to file
    KbSave {
        /// Path to save the knowledge base
        path: String,
    },

    /// Load knowledge base from file
    KbLoad {
        /// Path to load the knowledge base from
        path: String,
    },
}

#[derive(Subcommand)]
enum SemanticCommands {
    /// Search for content by query
    Search {
        /// Search query
        query: String,
        /// Number of results to return
        #[arg(short = 'k', long, default_value = "10")]
        top_k: usize,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Index content for semantic search
    Index {
        /// Content ID (CID) to index
        cid: String,
        /// Optional metadata
        #[arg(short, long)]
        metadata: Option<String>,
    },

    /// Find similar content
    Similar {
        /// Content ID (CID) to find similar content for
        cid: String,
        /// Number of results to return
        #[arg(short = 'k', long, default_value = "10")]
        top_k: usize,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Show semantic index statistics
    Stats {
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Save semantic index to file
    Save {
        /// Path to save the semantic index
        path: String,
    },

    /// Load semantic index from file
    Load {
        /// Path to load the semantic index from
        path: String,
    },
}

#[derive(Subcommand)]
enum DagCommands {
    /// Get a DAG node
    Get {
        /// Content ID (CID) of the DAG node
        cid: String,
        /// Output format (text, json)
        #[arg(long, default_value = "json")]
        format: String,
    },

    /// Put a DAG node
    Put {
        /// JSON data to store as DAG node
        data: String,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Resolve an IPLD path
    Resolve {
        /// IPLD path (e.g., /ipfs/QmHash/path/to/data)
        path: String,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Export a DAG to CAR file
    Export {
        /// Root CID of the DAG to export
        cid: String,
        /// Output CAR file path
        #[arg(short, long)]
        output: String,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Import blocks from CAR file
    Import {
        /// CAR file path to import from
        path: String,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },
}

#[derive(Subcommand)]
enum PinCommands {
    /// Pin content to prevent garbage collection
    Add {
        /// Content ID (CID) to pin
        cid: String,
        /// Recursively pin all linked blocks
        #[arg(short, long)]
        recursive: bool,
        /// Optional name for the pin
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Unpin content
    Rm {
        /// Content ID (CID) to unpin
        cid: String,
        /// Recursively unpin all linked blocks
        #[arg(short, long)]
        recursive: bool,
    },

    /// List pinned content
    Ls {
        /// Filter by pin type (all, direct, recursive, indirect)
        #[arg(long, default_value = "all")]
        pin_type: String,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Verify that pinned content is available
    Verify {
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },
}

#[derive(Subcommand)]
enum RepoCommands {
    /// Run garbage collection
    Gc {
        /// Perform a dry run (don't actually delete)
        #[arg(long)]
        dry_run: bool,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Show repository statistics
    Stat {
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Verify repository integrity
    Fsck {
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Show repository version
    Version {
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },
}

#[derive(Subcommand)]
enum ModelCommands {
    /// Add a model directory
    Add {
        /// Path to model directory
        path: String,
        /// Model name
        #[arg(short, long)]
        name: Option<String>,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Create a model checkpoint/snapshot
    Checkpoint {
        /// Model CID
        cid: String,
        /// Checkpoint message
        #[arg(short, long)]
        message: Option<String>,
        /// Metadata (JSON string)
        #[arg(short = 'M', long)]
        metadata: Option<String>,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Compare two model versions
    Diff {
        /// First model CID
        cid1: String,
        /// Second model CID
        cid2: String,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Restore a model version
    Rollback {
        /// Checkpoint CID to restore
        cid: String,
        /// Output path
        #[arg(short, long)]
        output: Option<String>,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },
}

#[derive(Subcommand)]
enum GradientCommands {
    /// Publish a gradient to the network
    Push {
        /// Path to gradient file
        path: String,
        /// Model CID this gradient applies to
        #[arg(short, long)]
        model_cid: Option<String>,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Fetch a gradient from the network
    Pull {
        /// Gradient CID
        cid: String,
        /// Output path
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Aggregate multiple gradients (federated learning)
    Aggregate {
        /// Gradient CIDs to aggregate
        #[arg(short, long)]
        cids: Vec<String>,
        /// Output path for aggregated gradient
        #[arg(short, long)]
        output: String,
        /// Aggregation method (mean, sum, weighted)
        #[arg(long, default_value = "mean")]
        method: String,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// View gradient update history for a model
    History {
        /// Model CID
        cid: String,
        /// Maximum number of history entries
        #[arg(short, long, default_value = "10")]
        limit: usize,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt().with_env_filter(log_level).init();

    match cli.command {
        Commands::Init { data_dir } => {
            init_repo(data_dir).await?;
        }
        Commands::Daemon { command } => match command {
            Some(DaemonCommands::Run { data_dir }) => {
                info!("Starting IPFRS daemon in {}", data_dir);
                run_daemon(data_dir).await?;
            }
            None => {
                // Default to Run with default data directory
                let data_dir = ".ipfrs".to_string();
                info!("Starting IPFRS daemon in {}", data_dir);
                run_daemon(data_dir).await?;
            }
            Some(DaemonCommands::Start {
                data_dir,
                pid_file,
                log_file,
            }) => {
                daemon_start(data_dir, pid_file, log_file).await?;
            }
            Some(DaemonCommands::Stop { pid_file }) => {
                daemon_stop(pid_file).await?;
            }
            Some(DaemonCommands::Status { pid_file }) => {
                daemon_status(pid_file).await?;
            }
            Some(DaemonCommands::Restart {
                data_dir,
                pid_file,
                log_file,
            }) => {
                daemon_restart(data_dir, pid_file, log_file).await?;
            }
            Some(DaemonCommands::Health {
                pid_file,
                data_dir,
                format,
            }) => {
                daemon_health(pid_file, data_dir, format).await?;
            }
        },
        Commands::Gateway {
            listen,
            data_dir,
            tls_cert,
            tls_key,
        } => {
            info!("Starting HTTP Gateway on {}", listen);
            run_gateway(listen, data_dir, tls_cert, tls_key).await?;
        }
        Commands::Add { path, format } => {
            info!("Adding file: {}", path);
            add_file(path, &format).await?;
        }
        Commands::Get { cid, output } => {
            get_file(cid, output).await?;
        }
        Commands::Cat { cid } => {
            info!("Retrieving content: {}", cid);
            cat_file(cid).await?;
        }
        Commands::Ls { cid, format } => {
            ls_directory(cid, &format).await?;
        }
        Commands::Block { command } => match command {
            BlockCommands::Get { cid } => {
                block_get(cid).await?;
            }
            BlockCommands::Put { path, format } => {
                block_put(path, &format).await?;
            }
            BlockCommands::Stat { cid, format } => {
                block_stat(cid, &format).await?;
            }
            BlockCommands::Rm { cid, force } => {
                block_rm(cid, force).await?;
            }
        },
        Commands::List { format } => {
            list_blocks(&format).await?;
        }
        Commands::Stats { command } => match command {
            StatsCommands::Repo { format } => {
                stats_repo(&format).await?;
            }
            StatsCommands::Bw { format } => {
                stats_bw(&format).await?;
            }
            StatsCommands::Bitswap { format } => {
                stats_bitswap(&format).await?;
            }
        },
        Commands::Info => {
            print_info();
        }
        Commands::Version => {
            print_version();
        }
        Commands::Id { format } => {
            show_id(&format).await?;
        }
        Commands::Ping { peer_id, count } => {
            ping_peer(&peer_id, count).await?;
        }
        Commands::Swarm { command } => match command {
            SwarmCommands::Peers { format } => {
                show_peers(&format).await?;
            }
            SwarmCommands::Connect { addr } => {
                swarm_connect(&addr).await?;
            }
            SwarmCommands::Disconnect { peer_id } => {
                swarm_disconnect(&peer_id).await?;
            }
            SwarmCommands::Addrs { format } => {
                swarm_addrs(&format).await?;
            }
        },
        Commands::Dht { command } => match command {
            DhtCommands::Findprovs { cid, format } => {
                dht_findprovs(&cid, &format).await?;
            }
            DhtCommands::Provide { cid } => {
                dht_provide(&cid).await?;
            }
            DhtCommands::Findpeer { peer_id, format } => {
                dht_findpeer(&peer_id, &format).await?;
            }
        },
        Commands::Bootstrap { command } => match command {
            BootstrapCommands::List { format } => {
                bootstrap_list(&format).await?;
            }
            BootstrapCommands::Add { addr } => {
                bootstrap_add(&addr).await?;
            }
            BootstrapCommands::Rm { addr } => {
                bootstrap_rm(&addr).await?;
            }
        },
        Commands::Logic { command } => match command {
            LogicCommands::Infer {
                predicate,
                terms,
                format,
            } => {
                logic_infer(&predicate, &terms, &format).await?;
            }
            LogicCommands::Prove {
                predicate,
                terms,
                format,
            } => {
                logic_prove(&predicate, &terms, &format).await?;
            }
            LogicCommands::KbStats { format } => {
                logic_kb_stats(&format).await?;
            }
            LogicCommands::KbSave { path } => {
                logic_kb_save(&path).await?;
            }
            LogicCommands::KbLoad { path } => {
                logic_kb_load(&path).await?;
            }
        },
        Commands::Semantic { command } => match command {
            SemanticCommands::Search {
                query,
                top_k,
                format,
            } => {
                semantic_search(&query, top_k, &format).await?;
            }
            SemanticCommands::Index { cid, metadata } => {
                semantic_index(&cid, metadata.as_deref()).await?;
            }
            SemanticCommands::Similar { cid, top_k, format } => {
                semantic_similar(&cid, top_k, &format).await?;
            }
            SemanticCommands::Stats { format } => {
                semantic_stats(&format).await?;
            }
            SemanticCommands::Save { path } => {
                semantic_save(&path).await?;
            }
            SemanticCommands::Load { path } => {
                semantic_load(&path).await?;
            }
        },
        Commands::Dag { command } => match command {
            DagCommands::Get { cid, format } => {
                dag_get(&cid, &format).await?;
            }
            DagCommands::Put { data, format } => {
                dag_put(&data, &format).await?;
            }
            DagCommands::Resolve { path, format } => {
                dag_resolve(&path, &format).await?;
            }
            DagCommands::Export {
                cid,
                output,
                format,
            } => {
                dag_export(&cid, &output, &format).await?;
            }
            DagCommands::Import { path, format } => {
                dag_import(&path, &format).await?;
            }
        },
        Commands::Pin { command } => match command {
            PinCommands::Add {
                cid,
                recursive,
                name,
            } => {
                pin_add(&cid, recursive, name.as_deref()).await?;
            }
            PinCommands::Rm { cid, recursive } => {
                pin_rm(&cid, recursive).await?;
            }
            PinCommands::Ls { pin_type, format } => {
                pin_ls(&pin_type, &format).await?;
            }
            PinCommands::Verify { format } => {
                pin_verify(&format).await?;
            }
        },
        Commands::Repo { command } => match command {
            RepoCommands::Gc { dry_run, format } => {
                repo_gc(dry_run, &format).await?;
            }
            RepoCommands::Stat { format } => {
                repo_stat(&format).await?;
            }
            RepoCommands::Fsck { format } => {
                repo_fsck(&format).await?;
            }
            RepoCommands::Version { format } => {
                repo_version(&format).await?;
            }
        },
        Commands::Tensor { command } => match command {
            TensorCommands::Add { path, format } => {
                tensor_add(&path, &format).await?;
            }
            TensorCommands::Get { cid, output } => {
                tensor_get(&cid, output.as_deref()).await?;
            }
            TensorCommands::Info { cid, format } => {
                tensor_info(&cid, &format).await?;
            }
            TensorCommands::Export {
                cid,
                output,
                target_format,
            } => {
                tensor_export(&cid, &output, &target_format).await?;
            }
        },
        Commands::Model { command } => match command {
            ModelCommands::Add { path, name, format } => {
                model_add(&path, name.as_deref(), &format).await?;
            }
            ModelCommands::Checkpoint {
                cid,
                message,
                metadata,
                format,
            } => {
                model_checkpoint(&cid, message.as_deref(), metadata.as_deref(), &format).await?;
            }
            ModelCommands::Diff { cid1, cid2, format } => {
                model_diff(&cid1, &cid2, &format).await?;
            }
            ModelCommands::Rollback {
                cid,
                output,
                format,
            } => {
                model_rollback(&cid, output.as_deref(), &format).await?;
            }
        },
        Commands::Gradient { command } => match command {
            GradientCommands::Push {
                path,
                model_cid,
                format,
            } => {
                gradient_push(&path, model_cid.as_deref(), &format).await?;
            }
            GradientCommands::Pull { cid, output } => {
                gradient_pull(&cid, output.as_deref()).await?;
            }
            GradientCommands::Aggregate {
                cids,
                output,
                method,
                format,
            } => {
                gradient_aggregate(&cids, &output, &method, &format).await?;
            }
            GradientCommands::History { cid, limit, format } => {
                gradient_history(&cid, limit, &format).await?;
            }
        },
        Commands::Shell { data_dir } => {
            use std::path::PathBuf;
            let config = shell::ShellConfig {
                data_dir: PathBuf::from(data_dir),
                ..Default::default()
            };
            let mut shell = shell::Shell::new(config)?;
            shell.run().await?;
        }
        Commands::Tui => {
            // tui already imported at top
            tui::run_tui().await?;
        }
        Commands::Plugin { command } => {
            handle_plugin_command(command).await?;
        }
        Commands::Completions { shell } => {
            generate_completions(shell);
        }
        Commands::Update { check } => {
            use output::{info, success};

            info("Checking for updates...");

            match utils::check_for_updates().await {
                Ok(Some(version)) => {
                    success(&format!(
                        "Update available: {} (current: {})",
                        version,
                        utils::VERSION
                    ));
                    if !check {
                        println!("\nTo update IPFRS, visit: {}", utils::REPO_URL);
                    }
                }
                Ok(None) => {
                    success(&format!(
                        "You are running the latest version: {}",
                        utils::VERSION
                    ));
                }
                Err(e) => {
                    error(&format!("Failed to check for updates: {}", e));
                }
            }
        }
    }

    Ok(())
}

// ============================================================================
// Validation Helpers
// ============================================================================

/// Validate that a CID string has valid format
#[allow(dead_code)]
fn validate_cid_format(cid_str: &str) -> Result<()> {
    if cid_str.is_empty() {
        return Err(anyhow::anyhow!("CID cannot be empty"));
    }

    // Check common CID prefixes
    if !cid_str.starts_with("Qm") && !cid_str.starts_with("bafy") && !cid_str.starts_with("bafk") {
        output::warning(
            "CID may have invalid format. Expected to start with 'Qm', 'bafy', or 'bafk'",
        );
    }

    Ok(())
}

/// Check if a path is readable
#[allow(dead_code)]
fn validate_path_readable(path: &str) -> Result<()> {
    let path_obj = std::path::Path::new(path);

    if !path_obj.exists() {
        return Err(anyhow::anyhow!(
            "Path does not exist: {}\nPlease check the path and try again.",
            path
        ));
    }

    // Try to open the file to check read permissions
    if path_obj.is_file() {
        std::fs::File::open(path_obj).map_err(|e| {
            anyhow::anyhow!(
                "Cannot read file: {}\nError: {}\n\nCheck file permissions.",
                path,
                e
            )
        })?;
    }

    Ok(())
}

/// Warn about potentially large operations
#[allow(dead_code)]
fn check_file_size_warning(size: u64) {
    const LARGE_FILE_THRESHOLD: u64 = 100 * 1024 * 1024; // 100 MB
    const HUGE_FILE_THRESHOLD: u64 = 1024 * 1024 * 1024; // 1 GB

    if size > HUGE_FILE_THRESHOLD {
        output::warning(&format!(
            "Very large file detected: {}. This operation may take significant time and memory.",
            format_bytes(size)
        ));
    } else if size > LARGE_FILE_THRESHOLD {
        output::warning(&format!(
            "Large file detected: {}. This may take a while.",
            format_bytes(size)
        ));
    }
}

// ============================================================================
// Daemon Management
// ============================================================================

async fn handle_plugin_command(command: PluginCommands) -> Result<()> {
    use ipfrs_cli::config::Config;
    use ipfrs_cli::output::{error, info, success, TablePrinter};
    use ipfrs_cli::plugin::PluginManager;

    let mut manager = PluginManager::new();
    manager.discover_plugins();

    match command {
        PluginCommands::List => {
            let plugins = manager.list_plugins();

            if plugins.is_empty() {
                info("No plugins found.");
                println!("\nTo add plugins, place executables in:");
                println!("  ~/.ipfrs/plugins/");
                println!("\nPlugin naming convention:");
                println!("  ipfrs-plugin-<name>");
                println!("\nExample:");
                println!("  ipfrs-plugin-hello");
                return Ok(());
            }

            info(&format!("Found {} plugin(s):", plugins.len()));
            println!();

            let mut table = TablePrinter::new(vec!["Name", "Description"]);

            for name in plugins {
                let desc = manager
                    .get_plugin(name)
                    .and_then(|p| p.description())
                    .unwrap_or("No description");
                table.add_row(vec![name, desc]);
            }

            table.print();
        }
        PluginCommands::Info { name } => {
            if let Some(plugin) = manager.get_plugin(&name) {
                success(&format!("Plugin: {}", plugin.name()));
                println!("Path: {}", plugin.path().display());

                if let Some(desc) = plugin.description() {
                    println!("Description: {}", desc);
                } else {
                    println!("Description: No description available");
                }
            } else {
                error(&format!("Plugin '{}' not found", name));
                println!("\nAvailable plugins:");
                for plugin_name in manager.list_plugins() {
                    println!("  - {}", plugin_name);
                }
                std::process::exit(1);
            }
        }
        PluginCommands::Run { name, args } => {
            let config = Config::load()?;

            match manager.execute_plugin(&name, &args, &config) {
                Ok(code) => {
                    std::process::exit(code);
                }
                Err(e) => {
                    error(&format!("Failed to execute plugin '{}': {}", name, e));
                    std::process::exit(1);
                }
            }
        }
    }

    Ok(())
}

fn generate_completions(shell: CompletionShell) {
    use clap::CommandFactory;
    use clap_complete::{generate, Shell};

    let mut cmd = Cli::command();
    let bin_name = "ipfrs";

    let shell = match shell {
        CompletionShell::Bash => Shell::Bash,
        CompletionShell::Zsh => Shell::Zsh,
        CompletionShell::Fish => Shell::Fish,
        CompletionShell::PowerShell => Shell::PowerShell,
        CompletionShell::Elvish => Shell::Elvish,
    };

    generate(shell, &mut cmd, bin_name, &mut std::io::stdout());
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn test_cli_parsing() {
        // Test basic command parsing
        let cli = Cli::try_parse_from(["ipfrs", "version"]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_init_command() {
        let cli = Cli::try_parse_from(["ipfrs", "init"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Init { data_dir } => {
                    assert_eq!(data_dir, ".ipfrs");
                }
                _ => panic!("Expected Init command"),
            }
        }
    }

    #[test]
    fn test_init_command_custom_dir() {
        let cli = Cli::try_parse_from(["ipfrs", "init", "--data-dir", "/tmp/ipfrs"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Init { data_dir } => {
                    assert_eq!(data_dir, "/tmp/ipfrs");
                }
                _ => panic!("Expected Init command"),
            }
        }
    }

    #[test]
    fn test_add_command() {
        let cli = Cli::try_parse_from(["ipfrs", "add", "test.txt"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Add { path, format } => {
                    assert_eq!(path, "test.txt");
                    assert_eq!(format, "text");
                }
                _ => panic!("Expected Add command"),
            }
        }
    }

    #[test]
    fn test_add_command_json_format() {
        let cli = Cli::try_parse_from(["ipfrs", "add", "test.txt", "--format", "json"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Add { path, format } => {
                    assert_eq!(path, "test.txt");
                    assert_eq!(format, "json");
                }
                _ => panic!("Expected Add command"),
            }
        }
    }

    #[test]
    fn test_get_command() {
        let cli = Cli::try_parse_from(["ipfrs", "get", "QmTest123"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Get { cid, output } => {
                    assert_eq!(cid, "QmTest123");
                    assert_eq!(output, None);
                }
                _ => panic!("Expected Get command"),
            }
        }
    }

    #[test]
    fn test_get_command_with_output() {
        let cli = Cli::try_parse_from(["ipfrs", "get", "QmTest123", "-o", "output.txt"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Get { cid, output } => {
                    assert_eq!(cid, "QmTest123");
                    assert_eq!(output, Some("output.txt".to_string()));
                }
                _ => panic!("Expected Get command"),
            }
        }
    }

    #[test]
    fn test_cat_command() {
        let cli = Cli::try_parse_from(["ipfrs", "cat", "QmTest123"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Cat { cid } => {
                    assert_eq!(cid, "QmTest123");
                }
                _ => panic!("Expected Cat command"),
            }
        }
    }

    #[test]
    fn test_ls_command() {
        let cli = Cli::try_parse_from(["ipfrs", "ls", "QmDir123"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Ls { cid, format } => {
                    assert_eq!(cid, "QmDir123");
                    assert_eq!(format, "text");
                }
                _ => panic!("Expected Ls command"),
            }
        }
    }

    #[test]
    fn test_block_get_command() {
        let cli = Cli::try_parse_from(["ipfrs", "block", "get", "QmBlock123"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Block { command } => match command {
                    BlockCommands::Get { cid } => {
                        assert_eq!(cid, "QmBlock123");
                    }
                    _ => panic!("Expected Block Get command"),
                },
                _ => panic!("Expected Block command"),
            }
        }
    }

    #[test]
    fn test_block_put_command() {
        let cli = Cli::try_parse_from(["ipfrs", "block", "put", "data.bin"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Block { command } => match command {
                    BlockCommands::Put { path, format } => {
                        assert_eq!(path, "data.bin");
                        assert_eq!(format, "text");
                    }
                    _ => panic!("Expected Block Put command"),
                },
                _ => panic!("Expected Block command"),
            }
        }
    }

    #[test]
    fn test_block_stat_command() {
        let cli = Cli::try_parse_from(["ipfrs", "block", "stat", "QmBlock123"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Block { command } => match command {
                    BlockCommands::Stat { cid, format } => {
                        assert_eq!(cid, "QmBlock123");
                        assert_eq!(format, "text");
                    }
                    _ => panic!("Expected Block Stat command"),
                },
                _ => panic!("Expected Block command"),
            }
        }
    }

    #[test]
    fn test_block_rm_command() {
        let cli = Cli::try_parse_from(["ipfrs", "block", "rm", "QmBlock123"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Block { command } => match command {
                    BlockCommands::Rm { cid, force } => {
                        assert_eq!(cid, "QmBlock123");
                        assert!(!force);
                    }
                    _ => panic!("Expected Block Rm command"),
                },
                _ => panic!("Expected Block command"),
            }
        }
    }

    #[test]
    fn test_block_rm_command_force() {
        let cli = Cli::try_parse_from(["ipfrs", "block", "rm", "QmBlock123", "--force"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Block { command } => match command {
                    BlockCommands::Rm { cid, force } => {
                        assert_eq!(cid, "QmBlock123");
                        assert!(force);
                    }
                    _ => panic!("Expected Block Rm command"),
                },
                _ => panic!("Expected Block command"),
            }
        }
    }

    #[test]
    fn test_ping_command() {
        let cli = Cli::try_parse_from(["ipfrs", "ping", "12D3KooWTest"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Ping { peer_id, count } => {
                    assert_eq!(peer_id, "12D3KooWTest");
                    assert_eq!(count, 5);
                }
                _ => panic!("Expected Ping command"),
            }
        }
    }

    #[test]
    fn test_ping_command_custom_count() {
        let cli = Cli::try_parse_from(["ipfrs", "ping", "12D3KooWTest", "-c", "10"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Ping { peer_id, count } => {
                    assert_eq!(peer_id, "12D3KooWTest");
                    assert_eq!(count, 10);
                }
                _ => panic!("Expected Ping command"),
            }
        }
    }

    #[test]
    fn test_id_command() {
        let cli = Cli::try_parse_from(["ipfrs", "id"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Id { format } => {
                    assert_eq!(format, "text");
                }
                _ => panic!("Expected Id command"),
            }
        }
    }

    #[test]
    fn test_id_command_json_format() {
        let cli = Cli::try_parse_from(["ipfrs", "id", "--format", "json"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Id { format } => {
                    assert_eq!(format, "json");
                }
                _ => panic!("Expected Id command"),
            }
        }
    }

    #[test]
    fn test_version_command() {
        let cli = Cli::try_parse_from(["ipfrs", "version"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Version => {}
                _ => panic!("Expected Version command"),
            }
        }
    }

    #[test]
    fn test_shell_command() {
        let cli = Cli::try_parse_from(["ipfrs", "shell"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Shell { data_dir } => {
                    assert_eq!(data_dir, ".ipfrs");
                }
                _ => panic!("Expected Shell command"),
            }
        }
    }

    #[test]
    fn test_verbose_flag() {
        let cli = Cli::try_parse_from(["ipfrs", "--verbose", "version"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            assert!(cli.verbose);
        }
    }

    #[test]
    fn test_no_color_flag() {
        let cli = Cli::try_parse_from(["ipfrs", "--no-color", "version"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            assert!(cli.no_color);
        }
    }

    #[test]
    fn test_config_flag() {
        let cli = Cli::try_parse_from(["ipfrs", "--config", "/tmp/config.toml", "version"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            assert_eq!(cli.config, Some("/tmp/config.toml".to_string()));
        }
    }

    #[test]
    fn test_daemon_run_command() {
        let cli = Cli::try_parse_from(["ipfrs", "daemon", "run"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Daemon { command } => match command {
                    Some(DaemonCommands::Run { data_dir }) => {
                        assert_eq!(data_dir, ".ipfrs");
                    }
                    _ => panic!("Expected Daemon Run command"),
                },
                _ => panic!("Expected Daemon command"),
            }
        }
    }

    #[test]
    fn test_daemon_start_command() {
        let cli = Cli::try_parse_from(["ipfrs", "daemon", "start"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Daemon { command } => match command {
                    Some(DaemonCommands::Start {
                        data_dir,
                        pid_file,
                        log_file,
                    }) => {
                        assert_eq!(data_dir, ".ipfrs");
                        assert_eq!(pid_file, ".ipfrs/daemon.pid");
                        assert_eq!(log_file, ".ipfrs/daemon.log");
                    }
                    _ => panic!("Expected Daemon Start command"),
                },
                _ => panic!("Expected Daemon command"),
            }
        }
    }

    #[test]
    fn test_daemon_stop_command() {
        let cli = Cli::try_parse_from(["ipfrs", "daemon", "stop"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Daemon { command } => match command {
                    Some(DaemonCommands::Stop { pid_file }) => {
                        assert_eq!(pid_file, ".ipfrs/daemon.pid");
                    }
                    _ => panic!("Expected Daemon Stop command"),
                },
                _ => panic!("Expected Daemon command"),
            }
        }
    }

    #[test]
    fn test_completions_bash() {
        let cli = Cli::try_parse_from(["ipfrs", "completions", "bash"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Completions { shell } => {
                    assert!(matches!(shell, CompletionShell::Bash));
                }
                _ => panic!("Expected Completions command"),
            }
        }
    }

    #[test]
    fn test_completions_zsh() {
        let cli = Cli::try_parse_from(["ipfrs", "completions", "zsh"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Completions { shell } => {
                    assert!(matches!(shell, CompletionShell::Zsh));
                }
                _ => panic!("Expected Completions command"),
            }
        }
    }

    #[test]
    fn test_tensor_add_command() {
        let cli = Cli::try_parse_from(["ipfrs", "tensor", "add", "model.safetensors"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Tensor { command } => match command {
                    TensorCommands::Add { path, format } => {
                        assert_eq!(path, "model.safetensors");
                        assert_eq!(format, "text");
                    }
                    _ => panic!("Expected Tensor Add command"),
                },
                _ => panic!("Expected Tensor command"),
            }
        }
    }

    #[test]
    fn test_tensor_get_command() {
        let cli = Cli::try_parse_from(["ipfrs", "tensor", "get", "QmTensor123"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Tensor { command } => match command {
                    TensorCommands::Get { cid, output } => {
                        assert_eq!(cid, "QmTensor123");
                        assert_eq!(output, None);
                    }
                    _ => panic!("Expected Tensor Get command"),
                },
                _ => panic!("Expected Tensor command"),
            }
        }
    }

    #[test]
    fn test_tensor_info_command() {
        let cli = Cli::try_parse_from(["ipfrs", "tensor", "info", "QmTensor123"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Tensor { command } => match command {
                    TensorCommands::Info { cid, format } => {
                        assert_eq!(cid, "QmTensor123");
                        assert_eq!(format, "text");
                    }
                    _ => panic!("Expected Tensor Info command"),
                },
                _ => panic!("Expected Tensor command"),
            }
        }
    }

    #[test]
    fn test_invalid_command() {
        let cli = Cli::try_parse_from(["ipfrs", "invalid_command"]);
        assert!(cli.is_err());
    }

    #[test]
    fn test_missing_required_argument() {
        let cli = Cli::try_parse_from(["ipfrs", "add"]);
        assert!(cli.is_err());
    }

    #[test]
    fn test_cli_help_generation() {
        let mut cmd = Cli::command();
        let help = cmd.render_help();
        let help_str = help.to_string();
        assert!(help_str.contains("ipfrs"));
        assert!(help_str.contains("IPFRS"));
    }

    #[test]
    fn test_cli_has_all_major_commands() {
        let cmd = Cli::command();
        let subcommands: Vec<_> = cmd.get_subcommands().map(|c| c.get_name()).collect();

        // Check for essential commands
        assert!(subcommands.contains(&"init"));
        assert!(subcommands.contains(&"add"));
        assert!(subcommands.contains(&"get"));
        assert!(subcommands.contains(&"cat"));
        assert!(subcommands.contains(&"ls"));
        assert!(subcommands.contains(&"block"));
        assert!(subcommands.contains(&"daemon"));
        assert!(subcommands.contains(&"version"));
        assert!(subcommands.contains(&"shell"));
        assert!(subcommands.contains(&"plugin"));
    }

    #[test]
    fn test_quiet_flag() {
        let cli = Cli::try_parse_from(["ipfrs", "--quiet", "version"]).unwrap();
        assert!(cli.quiet);
    }

    #[test]
    fn test_quiet_flag_short() {
        let cli = Cli::try_parse_from(["ipfrs", "-q", "version"]).unwrap();
        assert!(cli.quiet);
    }

    #[test]
    fn test_quiet_and_verbose_together() {
        // These flags are independent - both can be set
        let cli = Cli::try_parse_from(["ipfrs", "-q", "-v", "version"]).unwrap();
        assert!(cli.quiet);
        assert!(cli.verbose);
    }

    #[test]
    fn test_exit_codes_defined() {
        // Ensure exit codes are accessible
        assert_eq!(exit_codes::SUCCESS, 0);
        assert_eq!(exit_codes::ERROR, 1);
        assert_eq!(exit_codes::USAGE_ERROR, 2);
        assert_eq!(exit_codes::NOT_FOUND, 3);
        assert_eq!(exit_codes::PERMISSION_DENIED, 4);
        assert_eq!(exit_codes::NETWORK_ERROR, 5);
        assert_eq!(exit_codes::IO_ERROR, 6);
        assert_eq!(exit_codes::TIMEOUT, 7);
        assert_eq!(exit_codes::CONFIG_ERROR, 8);
    }

    #[test]
    fn test_plugin_list_command() {
        let cli = Cli::try_parse_from(["ipfrs", "plugin", "list"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Plugin { command } => match command {
                    PluginCommands::List => {}
                    _ => panic!("Expected PluginCommands::List"),
                },
                _ => panic!("Expected Plugin command"),
            }
        }
    }

    #[test]
    fn test_plugin_info_command() {
        let cli = Cli::try_parse_from(["ipfrs", "plugin", "info", "test-plugin"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Plugin { command } => match command {
                    PluginCommands::Info { name } => {
                        assert_eq!(name, "test-plugin");
                    }
                    _ => panic!("Expected PluginCommands::Info"),
                },
                _ => panic!("Expected Plugin command"),
            }
        }
    }

    #[test]
    fn test_plugin_run_command() {
        let cli = Cli::try_parse_from(["ipfrs", "plugin", "run", "my-plugin", "--arg1", "value"]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Plugin { command } => match command {
                    PluginCommands::Run { name, args } => {
                        assert_eq!(name, "my-plugin");
                        assert_eq!(args, vec!["--arg1", "value"]);
                    }
                    _ => panic!("Expected PluginCommands::Run"),
                },
                _ => panic!("Expected Plugin command"),
            }
        }
    }

    #[test]
    fn test_plugin_run_with_multiple_args() {
        let cli = Cli::try_parse_from([
            "ipfrs",
            "plugin",
            "run",
            "my-plugin",
            "arg1",
            "arg2",
            "--flag",
            "-v",
        ]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Commands::Plugin { command } => match command {
                    PluginCommands::Run { name, args } => {
                        assert_eq!(name, "my-plugin");
                        assert_eq!(args, vec!["arg1", "arg2", "--flag", "-v"]);
                    }
                    _ => panic!("Expected PluginCommands::Run"),
                },
                _ => panic!("Expected Plugin command"),
            }
        }
    }
}
