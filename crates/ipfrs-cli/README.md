# ipfrs-cli

Command-line interface for IPFRS.

## Overview

`ipfrs-cli` provides a user-friendly CLI for interacting with IPFRS:

- **Kubo-Compatible Commands**: Drop-in replacement for `ipfs` CLI
- **Enhanced Features**: TensorLogic-specific operations
- **Interactive Mode**: REPL for exploration
- **Daemon Management**: Start/stop IPFRS node

## Key Features

### Kubo-Compatible Commands
Standard IPFS commands work out of the box:

```bash
ipfrs add <file>              # Add file to IPFRS
ipfrs cat <cid>               # Retrieve file by CID
ipfrs get <cid>               # Download file/directory
ipfrs ls <cid>                # List directory contents
ipfrs dag get <cid>           # Get DAG node
ipfrs block get <cid>         # Get raw block
ipfrs swarm peers             # List connected peers
ipfrs dht findprovs <cid>     # Find providers
```

### TensorLogic Extensions
Advanced operations for AI workloads:

```bash
ipfrs tensor add <file>       # Add tensor with metadata
ipfrs tensor get <cid>        # Retrieve tensor (zero-copy)
ipfrs logic put <term>        # Store logic term
ipfrs logic query <goal>      # Run distributed inference
ipfrs gradient push <cid>     # Publish gradient update
ipfrs model checkpoint        # Create versioned snapshot
```

### Daemon Management
Control IPFRS node lifecycle:

```bash
ipfrs daemon                  # Run node in foreground
ipfrs daemon start            # Start background daemon
ipfrs daemon stop             # Stop background daemon
ipfrs daemon status           # Check daemon status
ipfrs init                    # Initialize repository
```

### Interactive Mode
REPL for experimentation:

```bash
ipfrs shell
ipfrs> add my_file.txt
added QmHash... my_file.txt
ipfrs> cat QmHash...
[file contents]
ipfrs> exit
```

### Terminal UI Dashboard
Interactive dashboard for real-time monitoring:

```bash
ipfrs tui
```

Features:
- **Overview Tab**: Peer count, storage usage, bandwidth gauges, node info
- **Network Tab**: Network activity sparkline, connected peers list
- **Storage Tab**: Block statistics, recent blocks, cache metrics
- **Help Tab**: Keyboard shortcuts and navigation guide

Navigation:
- `Tab` or `←/→` - Switch between tabs
- `1-4` - Jump directly to a tab
- `q` or `Ctrl+C` - Exit the dashboard

## Architecture

The CLI is organized into modular components for maintainability and reusability:

```
ipfrs-cli/
├── src/
│   ├── commands/          # Modular command implementations (15+ modules)
│   │   ├── mod.rs        # Module organization and exports
│   │   ├── common.rs     # Shared validation utilities
│   │   ├── file.rs       # File operations (init, add, get, cat, ls)
│   │   ├── block.rs      # Block operations (get, put, stat, rm)
│   │   ├── dag.rs        # DAG operations (get, put, resolve, export/import)
│   │   ├── daemon.rs     # Daemon management (start, stop, status, restart)
│   │   ├── pin.rs        # Pin management (add, rm, ls, verify)
│   │   ├── repo.rs       # Repository management (gc, stat, fsck)
│   │   ├── network.rs    # Network operations (swarm, dht, bootstrap)
│   │   ├── stats.rs      # Statistics (repo, bw, bitswap)
│   │   ├── tensor.rs     # Tensor operations (add, get, info, export)
│   │   ├── logic.rs      # Logic programming (infer, prove, kb-*)
│   │   ├── semantic.rs   # Semantic search (search, index, similar)
│   │   ├── model.rs      # Model management (add, checkpoint, diff)
│   │   ├── gradient.rs   # Gradient operations (push, pull, aggregate)
│   │   └── gateway.rs    # HTTP gateway server
│   ├── main.rs           # CLI entry point and dispatch (2,079 lines)
│   ├── lib.rs            # Library interface for reusability
│   ├── config.rs         # Configuration management with caching
│   ├── output.rs         # Output formatting (colors, tables, JSON)
│   ├── progress.rs       # Progress indicators (spinners, bars)
│   ├── shell.rs          # Interactive REPL shell
│   ├── tui.rs            # Terminal UI dashboard (ratatui)
│   ├── plugin.rs         # Plugin system for extensibility
│   └── utils.rs          # Utility functions (version, man pages)
└── tests/
    └── integration_tests.rs  # End-to-end CLI testing
        ↓
    ipfrs (core library)
```

**Recent Refactoring (2026-01-09):**
- Reduced main.rs from 4,825 to 2,079 lines (57% code reduction)
- Extracted all command implementations to modular `commands/` files
- Total codebase: 8,652 lines across 15+ command modules
- Single source of truth for all command logic

## Design Principles

- **User-Friendly**: Clear error messages, helpful hints
- **Fast**: Optimized for low startup time with config caching
- **Compatible**: Familiar to IPFS users
- **Powerful**: Expose advanced IPFRS features

## Performance Optimizations

The CLI is optimized for fast startup and low latency:

- **Config Caching**: Configuration files are loaded once and cached globally using `OnceLock` to avoid repeated disk I/O
- **Lazy Initialization**: Heavy modules are only loaded when needed
- **Minimal Dependencies**: Core functionality uses lightweight dependencies
- **Benchmarking**: Comprehensive benchmark suite to track performance metrics

Run benchmarks with:
```bash
cargo bench -p ipfrs-cli
```

Key metrics:
- CLI startup time: < 100ms (measured via benchmarks)
- Config load (cached): < 1μs
- Config load (uncached): < 500μs
- Command parsing: < 10ms

## Man Page Generation

Generate comprehensive man pages for all IPFRS commands:

```bash
# Generate man pages to target/man directory
cargo run --bin ipfrs-genman

# Generate to custom directory
cargo run --bin ipfrs-genman -- /path/to/output

# Install system-wide (Linux/macOS)
cargo run --bin ipfrs-genman
sudo cp target/man/*.1 /usr/share/man/man1/
sudo mandb

# View man pages
man ipfrs
man ipfrs-add
man ipfrs-daemon
```

The man page generator creates:
- Main man page: `ipfrs.1`
- Subcommand pages: `ipfrs-<command>.1` for each command

## Update Checking

Check for available updates using the hidden update command:

```bash
# Check for updates
ipfrs update --check

# The command will notify you if a newer version is available
```

## Troubleshooting

The CLI provides helpful troubleshooting hints for common errors. Use the `troubleshooting_hint()` function in your code or check error messages for diagnostic steps covering:

- Daemon not running
- Repository not initialized
- Connection failures
- Content not found
- Permission issues
- Configuration errors
- Network timeouts

## Shell Script Integration

The CLI is designed to be script-friendly with proper exit codes and quiet mode:

### Exit Codes

Standard exit codes for reliable error handling in scripts:

- `0` - Success
- `1` - General error
- `2` - Invalid arguments or command-line usage
- `3` - File or content not found
- `4` - Permission denied or authentication failed
- `5` - Network or connection error
- `6` - I/O error (file system operations)
- `7` - Timeout error
- `8` - Configuration error

Example script:
```bash
#!/bin/bash
if ipfrs add myfile.txt; then
    echo "File added successfully"
else
    case $? in
        2) echo "Usage error - check your arguments" ;;
        3) echo "File not found" ;;
        6) echo "I/O error - check permissions" ;;
        *) echo "Error occurred" ;;
    esac
fi
```

### Quiet Mode

Use `--quiet` or `-q` to suppress non-essential output for pipelines:

```bash
# Get just the CID without progress messages
CID=$(ipfrs add --quiet myfile.txt)

# Pipe content directly
ipfrs cat --quiet $CID | grep "pattern"

# Combine with JSON output for parsing
ipfrs ls --quiet --format json $CID | jq '.[] | .name'
```

### Other Script-Friendly Features

- `--no-color` - Disable colored output (useful for logs)
- `--format json` - Machine-readable JSON output
- Consistent output to stdout (data) and stderr (diagnostics)

## Migration from IPFS (Kubo)

IPFRS is designed to be compatible with IPFS workflows. If you're familiar with IPFS/Kubo, here's what you need to know:

### Command Compatibility

Most IPFS commands work identically in IPFRS:

| IPFS Command | IPFRS Equivalent | Notes |
|--------------|------------------|-------|
| `ipfs init` | `ipfrs init` | ✅ Same behavior |
| `ipfs add <file>` | `ipfrs add <file>` | ✅ Compatible CIDs |
| `ipfs cat <cid>` | `ipfrs cat <cid>` | ✅ Same output |
| `ipfs get <cid>` | `ipfrs get <cid>` | ✅ Same functionality |
| `ipfs daemon` | `ipfrs daemon` | ✅ Same API endpoints |
| `ipfs swarm peers` | `ipfrs swarm peers` | ✅ Compatible |
| `ipfs id` | `ipfrs id` | ✅ Same format |
| `ipfs pin add` | `ipfrs pin add` | ✅ Compatible |
| `ipfs dag get` | `ipfrs dag get` | ✅ IPLD compatible |
| `ipfs block get` | `ipfrs block get` | ✅ Compatible |

### What's Different

**Enhanced Features** (IPFRS-specific):
```bash
# Tensor operations (new in IPFRS)
ipfrs tensor add model.safetensors
ipfrs tensor info <cid>

# Logic programming (new in IPFRS)
ipfrs logic infer --predicate "ancestor" --terms "Alice,Bob"
ipfrs logic prove --goal "path(A,B)"

# Semantic search (new in IPFRS)
ipfrs semantic search "query" -k 10
ipfrs semantic similar <cid>

# Model versioning (new in IPFRS)
ipfrs model add ./model_dir
ipfrs model checkpoint <cid> -m "v1.0"
ipfrs model diff <cid1> <cid2>

# Federated learning (new in IPFRS)
ipfrs gradient push ./gradients
ipfrs gradient aggregate --method mean
```

**Configuration Differences**:
- IPFRS config: `~/.ipfrs/config.toml` (TOML format)
- IPFS config: `~/.ipfs/config` (JSON format)
- Environment variable: `IPFRS_PATH` vs `IPFS_PATH`

**API Compatibility**:
- IPFRS exposes the same HTTP API on port 5001 (configurable)
- IPFRS gateway runs on port 8080 (configurable)
- Most IPFS HTTP API endpoints work without modification

### Migration Steps

1. **Install IPFRS**:
```bash
cargo install --path crates/ipfrs-cli
```

2. **Initialize Repository**:
```bash
# Create new IPFRS repository
ipfrs init

# Or import existing IPFS data
ipfrs repo import ~/.ipfs
```

3. **Test Compatibility**:
```bash
# Add content with IPFRS
ipfrs add myfile.txt

# Verify same CID as IPFS would generate
# (IPFRS uses the same CID format)
```

4. **Migrate Pins** (if needed):
```bash
# Export pins from IPFS
ipfs pin ls > pins.txt

# Import to IPFRS
cat pins.txt | while read cid type; do
  ipfrs pin add $cid
done
```

5. **Update Scripts**:
```bash
# Simply replace 'ipfs' with 'ipfrs' in most cases
sed -i 's/ipfs /ipfrs /g' myscript.sh
```

### Interoperability

IPFRS can communicate with IPFS nodes on the network:

```bash
# Connect to IPFS node
ipfrs swarm connect /ip4/x.x.x.x/tcp/4001/p2p/Qm...

# Retrieve content from IPFS network
ipfrs get <cid-from-ipfs>

# Provide content to IPFS network
ipfrs dht provide <cid>
```

### Key Differences Summary

**Similarities**:
- ✅ Same CID format (content addressing)
- ✅ Same DAG structure (IPLD)
- ✅ Same network protocol (libp2p)
- ✅ Compatible HTTP API
- ✅ Same pinning mechanism

**New in IPFRS**:
- 🆕 Native tensor support (Safetensors, NumPy, PyTorch)
- 🆕 Built-in logic programming and inference
- 🆕 Semantic search with vector embeddings
- 🆕 Model versioning and diffing
- 🆕 Federated learning support
- 🆕 Enhanced performance for ML workloads

### Troubleshooting Migration

**Issue: "Repository not found"**
```bash
# Solution: Initialize IPFRS repo
ipfrs init
```

**Issue: "Cannot connect to daemon"**
```bash
# Solution: Start IPFRS daemon
ipfrs daemon start
# Or run in foreground
ipfrs daemon
```

**Issue: "Different CID for same content"**
```bash
# Ensure same chunking parameters
ipfrs add --chunker rabin myfile.txt
```

**Issue: "IPFS peers not visible"**
```bash
# IPFRS is compatible - check bootstrap peers
ipfrs bootstrap list
ipfrs bootstrap add /dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN
```

For more help, see the [troubleshooting guide](#troubleshooting) or visit the [IPFRS documentation](https://github.com/tensorlogic/ipfrs).

## Installation

```bash
# Build from source
cargo install --path crates/ipfrs-cli

# Or use pre-built binary
curl -sSL https://ipfrs.io/install.sh | sh
```

## Usage Examples

### Basic File Operations
```bash
# Add a file
$ ipfrs add document.pdf
added QmZ4tDuvesekSs4qM5ZBKpXiZGun7S2CYtEZRB3DYXkjGx document.pdf

# Retrieve the file
$ ipfrs cat QmZ4tDuvesekSs4qM5ZBKpXiZGun7S2CYtEZRB3DYXkjGx > output.pdf

# Add directory recursively
$ ipfrs add -r ./my_directory/
```

### Tensor Operations
```bash
# Add model weights
$ ipfrs tensor add model.safetensors
added QmModel... model.safetensors (1.2 GB, shape: [1024, 768])

# Query semantic similarity
$ ipfrs semantic search "neural networks" -k 10
QmHash1... (similarity: 0.95)
QmHash2... (similarity: 0.89)
...
```

### Network Operations
```bash
# Show connected peers
$ ipfrs swarm peers
/ip4/192.168.1.100/tcp/4001/p2p/QmPeer1...
/ip4/10.0.0.50/udp/4001/quic-v1/p2p/QmPeer2...

# Find content providers
$ ipfrs dht findprovs QmHash...
QmPeer3...
QmPeer4...
```

## Shell Completions

Generate shell completion scripts for faster command-line interaction:

```bash
# Bash
ipfrs completions bash > /etc/bash_completion.d/ipfrs
# Or for user-only:
ipfrs completions bash > ~/.local/share/bash-completion/completions/ipfrs

# Zsh
ipfrs completions zsh > "${fpath[1]}/_ipfrs"

# Fish
ipfrs completions fish > ~/.config/fish/completions/ipfrs.fish

# PowerShell
ipfrs completions powershell > ipfrs.ps1

# Elvish
ipfrs completions elvish > ~/.config/elvish/lib/ipfrs.elv
```

After installing, restart your shell or source the completion file to enable tab completion for ipfrs commands and arguments.

## Remote Daemon Management

IPFRS CLI supports connecting to remote daemons, allowing you to manage multiple IPFRS nodes from a single machine.

### Configuration

Configure remote daemon in `~/.ipfrs/config.toml`:

```toml
[api]
# Remote API URL (for connecting to remote daemon)
remote_url = "http://192.168.1.100:5001"
# API token (for authenticated connections)
api_token = "your-secret-token"
# Connection timeout in seconds
timeout = 60
```

### Environment Variables

Override configuration using environment variables:

```bash
# Connect to remote daemon
export IPFRS_API_URL="http://remote-host:5001"
export IPFRS_API_TOKEN="your-token"

# Now all commands connect to the remote daemon
ipfrs swarm peers
ipfrs add myfile.txt
ipfrs id
```

### Multiple Daemon Management

Manage multiple daemons using shell aliases:

```bash
# Add to ~/.bashrc or ~/.zshrc
alias ipfrs-local='IPFRS_API_URL=http://localhost:5001 ipfrs'
alias ipfrs-prod='IPFRS_API_URL=https://prod.example.com:5001 IPFRS_API_TOKEN=xxx ipfrs'
alias ipfrs-dev='IPFRS_API_URL=http://dev.example.com:5001 ipfrs'

# Usage
ipfrs-local id          # Check local daemon
ipfrs-prod swarm peers  # Check production daemon
ipfrs-dev add file.txt  # Add to development daemon
```

### Secure Remote Connections

For production deployments, use HTTPS and authentication:

```bash
# Enable authentication on the daemon
ipfrs daemon start --auth-token "secure-random-token"

# Connect from client
export IPFRS_API_URL="https://daemon.example.com:5001"
export IPFRS_API_TOKEN="secure-random-token"
ipfrs id
```

### Remote Command Examples

```bash
# Check remote daemon status
IPFRS_API_URL=http://remote:5001 ipfrs daemon status

# Add file to remote daemon
IPFRS_API_URL=http://remote:5001 ipfrs add largefile.bin

# Query remote daemon peers
IPFRS_API_URL=http://remote:5001 ipfrs swarm peers

# Pin content on remote daemon
IPFRS_API_URL=http://remote:5001 ipfrs pin add QmHash...
```

## Configuration

Configuration file: `~/.ipfrs/config.toml`

```toml
[general]
data_dir = ".ipfrs"
log_level = "info"

[storage]
blocks_path = "blocks"
cache_size = 104857600  # 100MB

[network]
listen_addrs = ["/ip4/0.0.0.0/tcp/4001"]
max_connections = 256

[api]
# Local daemon
listen_addr = "127.0.0.1:5001"
# Remote daemon (optional)
# remote_url = "http://remote-host:5001"
auth_enabled = false
# api_token = "your-token"
timeout = 60

[gateway]
listen_addr = "127.0.0.1:8080"
```

### Environment Variables

Supported environment variables (override config):

- `IPFRS_PATH` - Data directory
- `IPFRS_LOG_LEVEL` - Log level (error, warn, info, debug, trace)
- `IPFRS_API_URL` - Remote API URL
- `IPFRS_API_TOKEN` - API authentication token
```

## Dependencies

- `clap` - Command-line argument parsing
- `tokio` - Async runtime
- `ipfrs` - Core library
- `colored` - Terminal colors

## References

- IPFRS v0.2.0 Whitepaper (CLI Design)
- IPFS CLI Documentation: https://docs.ipfs.tech/reference/kubo/cli/
