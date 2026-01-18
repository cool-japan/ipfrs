# IPFRS CLI User Guide

A comprehensive guide to using the IPFRS command-line interface.

## Table of Contents

- [Getting Started](#getting-started)
- [Basic File Operations](#basic-file-operations)
- [Network Operations](#network-operations)
- [Advanced Features](#advanced-features)
- [TensorLogic Extensions](#tensorlogic-extensions)
- [Daemon Management](#daemon-management)
- [Interactive Shell](#interactive-shell)
- [Terminal UI Dashboard](#terminal-ui-dashboard)
- [Plugin System](#plugin-system)
- [Configuration](#configuration)
- [Shell Scripting](#shell-scripting)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

```bash
# Build from source
cargo install --path crates/ipfrs-cli

# Verify installation
ipfrs version
```

### Initialize Repository

Before using IPFRS, you need to initialize a repository:

```bash
# Initialize in the default directory (~/.ipfrs)
ipfrs init

# Initialize in a custom directory
ipfrs init --dir /path/to/repo
```

This creates:
- Configuration file: `~/.ipfrs/config.toml`
- Data directory: `~/.ipfrs/blocks/`
- Keystore: `~/.ipfrs/keystore/`

### Start the Daemon

IPFRS operates as a daemon for network operations:

```bash
# Run in foreground (for testing)
ipfrs daemon

# Start in background
ipfrs daemon start

# Check status
ipfrs daemon status
```

## Basic File Operations

### Adding Files

Add files or directories to IPFRS:

```bash
# Add a single file
ipfrs add myfile.txt
# Output: added QmHash... myfile.txt

# Add a directory recursively
ipfrs add --recursive ./my_folder/

# Add with JSON output (for scripts)
ipfrs add --format json document.pdf
```

### Retrieving Files

Get content by CID:

```bash
# Output to stdout
ipfrs cat QmHash...

# Save to file
ipfrs cat QmHash... > output.txt

# Download to filesystem with original name
ipfrs get QmHash...

# Download to specific location
ipfrs get QmHash... --output /path/to/save/
```

### Listing Directories

```bash
# List directory contents
ipfrs ls QmDirectoryHash...

# With JSON output
ipfrs ls --format json QmDirectoryHash...
```

## Network Operations

### Peer Management

```bash
# List connected peers
ipfrs swarm peers

# Connect to a specific peer
ipfrs swarm connect /ip4/192.168.1.100/tcp/4001/p2p/QmPeerId...

# Disconnect from a peer
ipfrs swarm disconnect QmPeerId...

# Show listening addresses
ipfrs swarm addrs
```

### DHT Operations

```bash
# Find content providers
ipfrs dht findprovs QmHash...

# Find peer address
ipfrs dht findpeer QmPeerId...

# Announce yourself as a provider
ipfrs dht provide QmHash...
```

### Bootstrap Configuration

```bash
# List bootstrap peers
ipfrs bootstrap list

# Add bootstrap peer
ipfrs bootstrap add /dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN

# Remove bootstrap peer
ipfrs bootstrap rm /dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN
```

### Network Diagnostics

```bash
# Ping a peer
ipfrs ping QmPeerId...

# Ping multiple times
ipfrs ping -c 10 QmPeerId...

# Show node identity
ipfrs id

# Network statistics
ipfrs stats bw
ipfrs stats bitswap
```

## Advanced Features

### Block Operations

Work with raw blocks:

```bash
# Get raw block data
ipfrs block get QmHash...

# Store raw block
ipfrs block put rawdata.bin

# Show block statistics
ipfrs block stat QmHash...

# Remove block (if unpinned)
ipfrs block rm QmHash...
```

### DAG Operations

Manage IPLD DAG nodes:

```bash
# Get DAG node
ipfrs dag get QmHash...

# Put DAG node from JSON
echo '{"name": "Alice", "age": 30}' | ipfrs dag put

# Resolve IPLD path
ipfrs dag resolve /ipfs/QmHash.../path/to/field

# Export DAG to CAR format
ipfrs dag export QmHash... --output backup.car

# Import DAG from CAR
ipfrs dag import backup.car
```

### Pin Management

Prevent garbage collection:

```bash
# Pin content
ipfrs pin add QmHash...

# Pin with a name
ipfrs pin add --name "important-data" QmHash...

# Recursive pin (default)
ipfrs pin add --recursive QmHash...

# Unpin content
ipfrs pin rm QmHash...

# List all pins
ipfrs pin ls

# Filter by type
ipfrs pin ls --type recursive

# Verify pin integrity
ipfrs pin verify
```

### Repository Management

```bash
# Garbage collection
ipfrs repo gc

# Dry run (see what would be deleted)
ipfrs repo gc --dry-run

# Repository statistics
ipfrs repo stat

# Verify repository integrity
ipfrs repo fsck

# Show repository version
ipfrs repo version
```

### HTTP Gateway

Serve content over HTTP:

```bash
# Start gateway on default port (8080)
ipfrs gateway

# Custom port
ipfrs gateway -l 0.0.0.0:9090

# With TLS/HTTPS
ipfrs gateway -l 0.0.0.0:8443 --tls-cert cert.pem --tls-key key.pem

# Access content via browser
# http://localhost:8080/ipfs/QmHash...
```

## TensorLogic Extensions

### Tensor Operations

Work with tensor data (Safetensors, NumPy, PyTorch):

```bash
# Add tensor file
ipfrs tensor add model_weights.safetensors
# Output: added QmTensorHash... (1.2 GB, shape: [1024, 768])

# Get tensor metadata
ipfrs tensor info QmTensorHash...

# Download tensor
ipfrs tensor get QmTensorHash... --output weights.safetensors

# Export to different format
ipfrs tensor export QmTensorHash... --output model.pt --target-format pytorch
```

### Logic Programming

Distributed inference with Prolog-style logic:

```bash
# Run inference query
ipfrs logic infer --predicate "ancestor" --terms "Alice,Bob"

# Generate proof tree
ipfrs logic prove --goal "path(A,B)"

# Knowledge base statistics
ipfrs logic kb-stats

# Save knowledge base
ipfrs logic kb-save --output kb.db

# Load knowledge base
ipfrs logic kb-load --input kb.db
```

### Semantic Search

Vector-based similarity search:

```bash
# Search by text query
ipfrs semantic search "neural network optimization" -k 10

# Find similar content
ipfrs semantic similar QmHash... -k 5

# Manually index content
ipfrs semantic index QmHash...

# Index statistics
ipfrs semantic stats

# Save index
ipfrs semantic save --output index.bin

# Load index
ipfrs semantic load --input index.bin
```

### Model Versioning

Track and manage ML models:

```bash
# Add model directory
ipfrs model add ./model_checkpoint/ --name "gpt-2-small"

# Create checkpoint
ipfrs model checkpoint QmModelHash... -m "Improved accuracy to 95%"

# Compare model versions
ipfrs model diff QmOldHash... QmNewHash...

# Rollback to previous version
ipfrs model rollback QmOldHash... --output ./restored_model/
```

### Federated Learning

Gradient sharing for distributed training:

```bash
# Publish gradient update
ipfrs gradient push ./gradients/ --model-cid QmModelHash...

# Fetch gradient from peer
ipfrs gradient pull QmGradientHash... --output ./received_gradients/

# Aggregate multiple gradients
ipfrs gradient aggregate --method mean QmGrad1... QmGrad2... QmGrad3...

# View gradient history
ipfrs gradient history QmModelHash... --limit 20
```

## Daemon Management

### Lifecycle Control

```bash
# Start daemon in foreground (logs to stdout)
ipfrs daemon

# Start as background service
ipfrs daemon start

# Stop background daemon
ipfrs daemon stop

# Check daemon status
ipfrs daemon status

# Restart daemon (preserves config)
ipfrs daemon restart
```

### Custom Configuration

```bash
# Use custom data directory
ipfrs daemon -d /mnt/ipfrs-data/

# Custom PID and log files
ipfrs daemon start --pid-file /var/run/ipfrs.pid --log-file /var/log/ipfrs.log
```

## Interactive Shell

Launch an interactive REPL:

```bash
# Start interactive shell
ipfrs shell

# Inside the shell
ipfrs> add myfile.txt
added QmHash... myfile.txt

ipfrs> cat QmHash...
[file contents]

ipfrs> swarm peers
/ip4/192.168.1.100/tcp/4001/p2p/QmPeer1...

ipfrs> help
[list of commands]

ipfrs> exit
```

Features:
- Command history (Up/Down arrows)
- Tab completion
- Multi-line input (backslash continuation)
- Persistent history file (`~/.ipfrs_history`)

## Terminal UI Dashboard

Launch the interactive dashboard:

```bash
ipfrs tui
```

### Dashboard Tabs

1. **Overview Tab**
   - Peer count gauge
   - Storage usage gauge
   - Bandwidth gauge
   - Node information (ID, version, uptime)

2. **Network Tab**
   - Network activity sparkline graph
   - Connected peers list with addresses

3. **Storage Tab**
   - Block statistics
   - Recent blocks list
   - Cache metrics

4. **Help Tab**
   - Keyboard shortcuts
   - Navigation guide

### Navigation

- `Tab` or `←/→` - Switch between tabs
- `1-4` - Jump directly to a tab
- `q` or `Ctrl+C` - Exit the dashboard

## Plugin System

Extend IPFRS CLI with custom plugins.

### Installing Plugins

Place executable plugins in `~/.ipfrs/plugins/` with the naming pattern: `ipfrs-plugin-<name>`

### Using Plugins

```bash
# List available plugins
ipfrs plugin list

# Show plugin information
ipfrs plugin info backup

# Execute a plugin
ipfrs plugin run backup --destination /mnt/backup/

# Pass arguments to plugin
ipfrs plugin run stats --format json --verbose
```

### Environment Variables

Plugins receive these environment variables:
- `IPFRS_DATA_DIR` - Repository data directory
- `IPFRS_LOG_LEVEL` - Current log level
- `IPFRS_API_URL` - Remote API URL (if configured)
- `IPFRS_API_TOKEN` - API authentication token (if configured)

### Creating Plugins

Create an executable with the naming convention `ipfrs-plugin-<name>`:

```bash
#!/bin/bash
# ipfrs-plugin-hello

case "$1" in
  --plugin-info)
    echo "name: hello"
    echo "version: 1.0.0"
    echo "description: Simple hello world plugin"
    echo "author: Your Name"
    ;;
  *)
    echo "Hello from IPFRS plugin!"
    echo "Data dir: $IPFRS_DATA_DIR"
    echo "Arguments: $@"
    ;;
esac
```

Make it executable:
```bash
chmod +x ipfrs-plugin-hello
mv ipfrs-plugin-hello ~/.ipfrs/plugins/
```

## Configuration

### Configuration File

Location: `~/.ipfrs/config.toml`

```toml
[general]
data_dir = ".ipfrs"
log_level = "info"  # error, warn, info, debug, trace

[storage]
blocks_path = "blocks"
cache_size = 104857600  # 100MB in bytes

[network]
listen_addrs = ["/ip4/0.0.0.0/tcp/4001", "/ip4/0.0.0.0/udp/4001/quic-v1"]
max_connections = 256

[api]
listen_addr = "127.0.0.1:5001"
auth_enabled = false
timeout = 60

[gateway]
listen_addr = "127.0.0.1:8080"
```

### Environment Variables

Override configuration with environment variables:

```bash
# Data directory
export IPFRS_PATH=/mnt/ipfrs-data

# Log level
export IPFRS_LOG_LEVEL=debug

# Remote daemon
export IPFRS_API_URL=http://remote-host:5001
export IPFRS_API_TOKEN=your-secret-token
```

### Command-Line Flags

Override per-command:

```bash
# Custom config file
ipfrs -c /path/to/config.toml add file.txt

# Verbose logging
ipfrs -v daemon

# Disable colors
ipfrs --no-color swarm peers

# Quiet mode (scripts)
ipfrs -q add file.txt
```

## Shell Scripting

### Exit Codes

IPFRS uses standard exit codes for script error handling:

- `0` - Success
- `1` - General error
- `2` - Invalid arguments
- `3` - File or content not found
- `4` - Permission denied
- `5` - Network error
- `6` - I/O error
- `7` - Timeout error
- `8` - Configuration error

### Example Script

```bash
#!/bin/bash
set -e  # Exit on error

# Add file and capture CID
CID=$(ipfrs add --quiet myfile.txt)

if [ $? -eq 0 ]; then
    echo "File added successfully: $CID"

    # Pin the content
    if ipfrs pin add "$CID"; then
        echo "Content pinned"
    else
        echo "Failed to pin content" >&2
        exit 1
    fi
else
    echo "Failed to add file" >&2
    exit $?
fi
```

### Batch Operations

```bash
#!/bin/bash

# Add multiple files and collect CIDs
for file in *.txt; do
    CID=$(ipfrs add --quiet "$file")
    echo "$file -> $CID"
done

# Retrieve multiple files
while read CID; do
    ipfrs get --quiet "$CID" &
done < cid_list.txt
wait

echo "All downloads complete"
```

### JSON Parsing with jq

```bash
#!/bin/bash

# Get peer list as JSON and parse
ipfrs swarm peers --format json | jq -r '.[] | .id'

# Get directory listing and filter
ipfrs ls --format json "$DIR_CID" | jq -r '.[] | select(.type == "file") | .name'

# Get stats and extract values
PEER_COUNT=$(ipfrs stats bw --format json | jq -r '.peers')
echo "Connected peers: $PEER_COUNT"
```

### Quiet Mode

Suppress non-essential output for pipelines:

```bash
# Just get the CID
CID=$(ipfrs add -q file.txt)

# Pipe content directly
ipfrs cat -q "$CID" | grep "pattern"

# Chain operations
ipfrs ls -q "$DIR_CID" --format json | jq '.[] | .hash' | while read CID; do
    ipfrs pin add -q "$CID"
done
```

## Best Practices

### Content Management

1. **Pin important content** to prevent garbage collection
2. **Use meaningful names** when pinning: `ipfrs pin add --name "dataset-v1" QmHash...`
3. **Regular garbage collection** to reclaim space: `ipfrs repo gc`
4. **Verify repository integrity** periodically: `ipfrs repo fsck`

### Network Optimization

1. **Configure bootstrap peers** for better connectivity
2. **Monitor peer count**: Aim for 20-50 connected peers
3. **Use DHT provide** for content you want to share
4. **Adjust max_connections** based on your bandwidth

### Performance

1. **Increase cache_size** if you have RAM available
2. **Pin frequently accessed content**
3. **Use compression** for gateway serving
4. **Monitor bandwidth** with `ipfrs stats bw`

### Security

1. **Use TLS for gateway** in production: `--tls-cert` and `--tls-key`
2. **Enable API authentication** for remote access
3. **Use HTTPS for remote_url** configuration
4. **Restrict listen addresses** to trusted networks

### Backups

1. **Export important DAGs** to CAR files: `ipfrs dag export QmHash... --output backup.car`
2. **Save configuration**: `cp ~/.ipfrs/config.toml backup/`
3. **Backup pins list**: `ipfrs pin ls > pins.txt`
4. **Test restoration** regularly

## Troubleshooting

### Daemon Won't Start

```bash
# Check if already running
ipfrs daemon status

# Check logs
tail -f ~/.ipfrs/daemon.log

# Remove stale PID file
rm ~/.ipfrs/daemon.pid

# Start with verbose logging
ipfrs -v daemon
```

### Cannot Connect to Peers

```bash
# Check bootstrap peers
ipfrs bootstrap list

# Add default bootstrap peers
ipfrs bootstrap add /dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN

# Check listening addresses
ipfrs swarm addrs

# Verify firewall allows port 4001
```

### Content Not Found

```bash
# Check if content is pinned
ipfrs pin ls | grep QmHash...

# Find providers
ipfrs dht findprovs QmHash...

# Check repository integrity
ipfrs repo fsck

# Connect to a known provider
ipfrs swarm connect /ip4/.../tcp/4001/p2p/QmPeerId...
```

### High Memory Usage

```bash
# Reduce cache size in config.toml
[storage]
cache_size = 52428800  # 50MB

# Run garbage collection
ipfrs repo gc

# Restart daemon
ipfrs daemon restart
```

### Slow Operations

```bash
# Check peer count
ipfrs swarm peers | wc -l

# Verify network connectivity
ipfrs ping QmPeerId...

# Check repository size
ipfrs repo stat

# Enable verbose logging to identify bottleneck
ipfrs -v <command>
```

### Repository Corruption

```bash
# Verify repository
ipfrs repo fsck

# If corruption detected, restore from backup
ipfrs dag import backup.car

# Verify pins
ipfrs pin verify

# Rebuild if necessary
ipfrs init --force
```

## Additional Resources

- **README.md** - Quick start and feature overview
- **TODO.md** - Development roadmap and completed features
- **CHANGELOG.md** - Version history and release notes
- **Man Pages** - Detailed command reference: `man ipfrs`, `man ipfrs-add`, etc.

## Getting Help

```bash
# General help
ipfrs --help

# Command-specific help
ipfrs add --help
ipfrs daemon --help

# Interactive help
ipfrs shell
ipfrs> help

# Version information
ipfrs version
```

For more assistance, visit the [IPFRS GitHub repository](https://github.com/tensorlogic/ipfrs) or check the issue tracker.
