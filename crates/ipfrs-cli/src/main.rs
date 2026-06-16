//! IPFRS CLI - Command-line interface for IPFRS
//!
//! Version: 0.3.0 "The Fast & The Wise"

// Import library modules
use ipfrs_cli::commands::ipld as ipld_cmds;
use ipfrs_cli::commands::query::OutputFormat;
use ipfrs_cli::commands::*;
use ipfrs_cli::connectivity::{check_daemon_reachable, offline_error_message};
use ipfrs_cli::{output, shell, tui, utils};

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::info;

use output::error;

mod dispatch;

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
        ipfrs get QmHash123 -o file   # Save to specific file\n  \
        ipfrs get QmHash123 --timeout 60  # Allow up to 60 s for network fetch")]
    Get {
        /// Content Identifier (CID) of the content to retrieve
        #[arg(value_name = "CID")]
        cid: String,
        /// Output file path (defaults to CID if not specified)
        #[arg(short, long, value_name = "FILE")]
        output: Option<String>,
        /// Timeout in seconds for the block fetch (0 = no timeout, default: 30)
        #[arg(long, default_value = "30", value_name = "SECS")]
        timeout: u64,
    },

    /// Output the contents of a file to stdout
    #[command(long_about = "Retrieve and output content directly to stdout.\n\n\
        This is useful for piping content to other commands or viewing\n\
        file contents without saving to disk.\n\n\
        Examples:\n  \
        ipfrs cat QmHash123           # Output to stdout\n  \
        ipfrs cat QmHash123 | less    # Pipe to pager\n  \
        ipfrs cat QmHash123 --timeout 60  # Allow up to 60 s for network fetch")]
    Cat {
        /// Content Identifier (CID) to retrieve and output
        #[arg(value_name = "CID")]
        cid: String,
        /// Timeout in seconds for the block fetch (0 = no timeout, default: 30)
        #[arg(long, default_value = "30", value_name = "SECS")]
        timeout: u64,
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

    /// Run a query — semantic similarity, logic inference, or both
    #[command(
        long_about = "Run a query using semantic similarity, logic inference, or both.\n\n\
            Auto-detects query type:\n  \
            - Logic predicates: ancestor(X, bob)\n  \
            - Natural language: tensor operations\n\n\
            Examples:\n  \
            ipfrs query \"tensor operations\"\n  \
            ipfrs query \"ancestor(X, bob)\"\n  \
            ipfrs query --hybrid \"machine learning\" --top-k 5\n  \
            ipfrs query --hybrid \"machine learning\" --logic \"indexed(X)\"\n  \
            ipfrs query \"parent(X, bob)\" --format json"
    )]
    Query {
        /// Query string (natural language or logic predicate)
        #[arg(value_name = "QUERY")]
        query: String,

        /// Enable hybrid mode (run both semantic and logic search)
        #[arg(long, default_value = "false")]
        hybrid: bool,

        /// Pipeline mode: read CIDs from stdin and apply query as a logic predicate filter
        #[arg(long)]
        pipeline: bool,

        /// Number of results to return
        #[arg(long, short = 'k', default_value = "10")]
        top_k: usize,

        /// Logic predicate filter applied to semantic results in hybrid mode (use X as CID placeholder)
        #[arg(long, value_name = "PREDICATE")]
        logic: Option<String>,

        /// Output format: text (human-readable) or json (newline-delimited JSON)
        #[arg(long, value_enum, default_value = "text")]
        format: OutputFormat,

        /// Output as JSON (deprecated: use --format json instead)
        #[arg(long, hide = true)]
        json: bool,
    },

    /// DAG (Directed Acyclic Graph) operations
    Dag {
        #[command(subcommand)]
        command: DagCommands,
    },

    /// IPLD path resolution and block inspection
    #[command(long_about = "Resolve IPLD paths and inspect DAG blocks.\n\n\
            Subcommands:\n  \
            ipfrs ipld resolve /ipld/<cid>/field/0   Resolve a path and print the value\n  \
            ipfrs ipld stat <cid>                    Print codec, size, and link count\n  \
            ipfrs ipld links <cid>                   List all CIDs linked from a block")]
    Ipld {
        #[command(subcommand)]
        subcommand: IpldCommands,
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

    /// Peer identity management
    #[command(long_about = "Manage the node's Ed25519 peer identity key.\n\n\
            The identity key determines the node's PeerId on the network.\n\
            Rotating the key changes the PeerId and severs existing connections.\n\n\
            Examples:\n  \
            ipfrs identity show            # Show current PeerId and key info\n  \
            ipfrs identity rotate          # Generate a new keypair\n  \
            ipfrs identity export-pem      # Export public key as PEM")]
    Identity {
        #[command(subcommand)]
        subcommand: IdentityCommands,
    },

    /// Prometheus metrics operations
    #[command(
        long_about = "View and manage Prometheus metrics collected by the IPFRS node.\n\n\
            Examples:\n  \
            ipfrs metrics show             # Print all metrics in Prometheus text format\n  \
            ipfrs metrics show --format json  # Print metrics wrapped in JSON\n  \
            ipfrs metrics reset            # Reset metric counters (requires daemon restart)"
    )]
    Metrics {
        #[command(subcommand)]
        command: MetricsCommands,
    },

    /// Print a diagnostic snapshot of the running node
    #[command(
        long_about = "Collect and display a diagnostic snapshot of the IPFRS node.\n\n\
            Reports daemon status, storage usage, network peers, inference latency,\n\
            HNSW index size, TensorLogic knowledge-base stats, and uptime.\n\n\
            Examples:\n  \
            ipfrs diag               # Human-readable report\n  \
            ipfrs diag --json        # Machine-readable JSON"
    )]
    Diag {
        /// Emit diagnostics as a JSON object instead of a human-readable table
        #[arg(long)]
        json: bool,
    },
}

/// Subcommands for `ipfrs identity`
#[derive(Subcommand)]
enum IdentityCommands {
    /// Show the current PeerId and public key fingerprint
    Show {
        /// Data directory containing the identity key file
        #[arg(short, long, default_value = ".ipfrs")]
        data_dir: String,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Rotate the peer identity key (generates a new Ed25519 keypair)
    Rotate {
        /// Data directory containing the identity key file
        #[arg(short, long, default_value = ".ipfrs")]
        data_dir: String,
    },

    /// Export the current public key as a PEM block
    ExportPem {
        /// Data directory containing the identity key file
        #[arg(short, long, default_value = ".ipfrs")]
        data_dir: String,
    },
}

/// Subcommands for `ipfrs metrics`
#[derive(Subcommand)]
enum MetricsCommands {
    /// Print all metrics in Prometheus text format (or JSON with --format json)
    Show {
        /// Output format: `text` (default) or `json`
        #[arg(long, default_value = "text", value_name = "FORMAT")]
        format: OutputFormat,
    },
    /// Reset metric counters (informational — requires daemon restart for a full reset)
    Reset,
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

    /// Run a Datalog-style goal query (e.g., "ancestor(X, bob)")
    Query {
        /// Goal predicate to solve (e.g., "ancestor(X, bob)")
        #[arg(value_name = "GOAL")]
        goal: String,

        /// Maximum inference depth
        #[arg(long, default_value = "10")]
        max_depth: usize,

        /// Timeout in milliseconds (overrides --timeout when set)
        #[arg(long, value_name = "MS")]
        timeout_ms: Option<u64>,

        /// Timeout in seconds (default: 30; use --timeout-ms for millisecond precision)
        #[arg(long, default_value = "30")]
        timeout: u64,

        /// Output format: text (human-readable) or json (newline-delimited JSON)
        #[arg(long, value_enum, default_value = "text")]
        format: OutputFormat,

        /// Output as JSON (deprecated: use --format json instead)
        #[arg(long, hide = true)]
        json: bool,
    },

    /// Filter CIDs from stdin by applying a logic predicate (use X as CID placeholder)
    #[command(
        long_about = "Read CIDs from stdin (one per line) and output only those for which \
            the predicate holds.\n\n\
            X in the predicate template is replaced by each CID before inference.\n\n\
            Examples:\n  \
            echo 'bafkrei...' | ipfrs logic filter 'indexed(X)'\n  \
            ipfrs semantic query \"tensors\" --json | jq -r '.[].cid' | ipfrs logic filter 'valid(X)'"
    )]
    Filter {
        /// Logic predicate template (use X as placeholder for each CID)
        #[arg(value_name = "PREDICATE")]
        predicate: String,

        /// Output matching CIDs as a JSON array
        #[arg(long)]
        json: bool,

        /// Data directory (default: .ipfrs)
        #[arg(long, default_value = ".ipfrs")]
        data_dir: String,
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

    /// Semantic similarity search by text query
    Query {
        /// Query text to search for similar content
        #[arg(value_name = "TEXT")]
        text: String,

        /// Number of top results to return
        #[arg(long, short = 'k', default_value = "10")]
        top_k: usize,

        /// Minimum similarity threshold (0.0 to 1.0)
        #[arg(long, default_value = "0.0")]
        threshold: f32,

        /// Output format: text (human-readable) or json (newline-delimited JSON)
        #[arg(long, value_enum, default_value = "text")]
        format: OutputFormat,

        /// Output as JSON (deprecated: use --format json instead)
        #[arg(long, hide = true)]
        json: bool,
    },
}

/// Subcommands for `ipfrs ipld`
#[derive(Subcommand)]
enum IpldCommands {
    /// Resolve an IPLD path and print the value
    ///
    /// Path format: /ipld/\<cid\>/field/subfield/0
    Resolve {
        /// Full IPLD path (e.g., /ipld/bafk.../head/args/0)
        #[arg(value_name = "PATH")]
        path: String,
        /// Output format: text (default) or json
        #[arg(long, value_enum, default_value = "text")]
        format: OutputFormat,
    },

    /// Print metadata about a block: codec, size, links count
    Stat {
        /// Content Identifier (CID) of the block
        #[arg(value_name = "CID")]
        cid: String,
        /// Output format: text (default) or json
        #[arg(long, value_enum, default_value = "text")]
        format: OutputFormat,
    },

    /// List all CIDs linked from a given block
    Links {
        /// Content Identifier (CID) of the block
        #[arg(value_name = "CID")]
        cid: String,
        /// Output format: text (default) or json
        #[arg(long, value_enum, default_value = "text")]
        format: OutputFormat,
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
        /// Only collect blocks older than this many seconds (default: 3600 = 1 h)
        #[arg(long, default_value_t = 3600u64)]
        min_age: u64,
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

    /// Flush the Sled WAL / trigger compaction
    ///
    /// Without `--force` the compaction scheduler decides whether enough time
    /// has elapsed and the store is sufficiently idle.  Pass `--force` to
    /// flush unconditionally regardless of schedule.
    #[command(
        long_about = "Flush the Sled write-ahead log and trigger storage compaction.\n\n\
        By default the compaction scheduler determines whether enough time has\n\
        elapsed since the last compaction and whether the store is idle.\n\
        Use --force to flush immediately regardless of schedule.\n\n\
        Also reports current deduplication statistics.\n\n\
        Examples:\n  \
        ipfrs repo compact            # Compact if schedule allows\n  \
        ipfrs repo compact --force    # Flush unconditionally\n  \
        ipfrs repo compact --format json"
    )]
    Compact {
        /// Force compaction even when the scheduler considers it not yet due
        #[arg(long)]
        force: bool,
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

    // Fast offline check: if the command requires the daemon and it is not
    // running, print a helpful message and exit early rather than letting the
    // error surface as a cryptic connection-refused deep in the call stack.
    if requires_daemon(&cli.command) {
        // Use a default data directory for the PID-file check.  Most network
        // commands do not carry a `--data-dir` argument, so we use the
        // conventional default.
        let data_dir = ".ipfrs";
        if !check_daemon_reachable(data_dir).await {
            eprintln!("{}", offline_error_message(data_dir));
            std::process::exit(exit_codes::NETWORK_ERROR);
        }
    }

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
        Commands::Get {
            cid,
            output,
            timeout,
        } => {
            get_file(cid, output, timeout).await?;
        }
        Commands::Cat { cid, timeout } => {
            info!("Retrieving content: {}", cid);
            cat_file(cid, timeout).await?;
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
        Commands::Query {
            query,
            hybrid,
            pipeline,
            top_k,
            logic,
            format,
            json,
        } => {
            // --json is a legacy alias for --format json.
            let effective_format = if json { OutputFormat::Json } else { format };
            handle_query(
                &query,
                hybrid,
                pipeline,
                top_k,
                logic.as_deref(),
                &effective_format,
            )
            .await?;
        }
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
            LogicCommands::Query {
                goal,
                max_depth,
                timeout_ms,
                timeout,
                format,
                json,
            } => {
                // --timeout-ms takes precedence; fall back to --timeout (seconds).
                let effective_timeout_secs = if let Some(ms) = timeout_ms {
                    // Round up to nearest second for the existing API.
                    ms.div_ceil(1000)
                } else {
                    timeout
                };
                let json_output = json || format.is_json();
                logic_query_streaming(&goal, max_depth, json_output, effective_timeout_secs)
                    .await?;
            }
            LogicCommands::Filter {
                predicate,
                json,
                data_dir,
            } => {
                logic_filter(&predicate, json, &data_dir).await?;
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
            SemanticCommands::Query {
                text,
                top_k,
                threshold,
                format,
                json,
            } => {
                let json_output = json || format.is_json();
                semantic_query(&text, top_k, threshold, json_output).await?;
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
        Commands::Ipld { subcommand } => match subcommand {
            IpldCommands::Resolve { path, format } => {
                ipld_cmds::ipld_resolve(&path, &format).await?;
            }
            IpldCommands::Stat { cid, format } => {
                ipld_cmds::ipld_stat(&cid, &format).await?;
            }
            IpldCommands::Links { cid, format } => {
                ipld_cmds::ipld_links(&cid, &format).await?;
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
            RepoCommands::Gc {
                dry_run,
                min_age,
                format,
            } => {
                repo_gc(dry_run, min_age, &format).await?;
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
            RepoCommands::Compact { force, format } => {
                storage_compact(force, &format).await?;
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
            dispatch::handle_plugin_command(command).await?;
        }
        Commands::Completions { shell } => {
            dispatch::generate_completions(shell);
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
        Commands::Identity { subcommand } => {
            dispatch::handle_identity_command(subcommand).await?;
        }
        Commands::Metrics { command } => match command {
            MetricsCommands::Show { format } => {
                handle_metrics_show(&format).await?;
            }
            MetricsCommands::Reset => {
                handle_metrics_reset().await?;
            }
        },
        Commands::Diag { json } => {
            handle_diag(json).await?;
        }
    }

    Ok(())
}

// ============================================================================
// Daemon-connectivity helpers
// ============================================================================

/// Returns `true` for commands that require the IPFRS network daemon to be
/// running.  Used to produce a fast, actionable error message instead of a
/// confusing connection-refused error deep in the call stack.
fn requires_daemon(cmd: &Commands) -> bool {
    matches!(
        cmd,
        Commands::Swarm { .. } | Commands::Dht { .. } | Commands::Bootstrap { .. }
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "cli_tests.rs"]
mod tests {}
