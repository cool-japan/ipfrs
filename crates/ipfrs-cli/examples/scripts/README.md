# IPFRS Example Scripts

This directory contains example shell scripts demonstrating common IPFRS workflows.

## Available Scripts

### backup.sh
Backs up IPFRS repository data, configuration, and pins.

**Usage:**
```bash
./backup.sh [backup_directory]
```

**What it does:**
- Backs up configuration file
- Exports list of pinned content
- Saves repository statistics
- Exports pinned DAGs to CAR files
- Creates a manifest for restoration

**Example:**
```bash
# Backup to default location
./backup.sh

# Backup to custom location
./backup.sh /mnt/backups/ipfrs
```

### restore.sh
Restores IPFRS repository from a backup.

**Usage:**
```bash
./restore.sh <backup_directory>
```

**What it does:**
- Restores configuration
- Initializes repository if needed
- Imports CAR files
- Restores pins

**Example:**
```bash
./restore.sh ./ipfrs_backup/20260109_123456
```

### batch_add.sh
Adds multiple files from a directory and tracks their CIDs.

**Usage:**
```bash
./batch_add.sh <source_directory> [output_file]
```

**What it does:**
- Recursively adds all files from a directory
- Records each file's CID and size
- Provides summary statistics
- Generates a manifest file

**Example:**
```bash
# Add all files from a directory
./batch_add.sh ./my_documents ./added_files.txt

# Pin all added files
cat ./added_files.txt | grep -v '^#' | awk '{print $2}' | xargs -n1 ipfrs pin add
```

### monitor.sh
Real-time monitoring dashboard for IPFRS node.

**Usage:**
```bash
./monitor.sh [refresh_interval_seconds]
```

**What it displays:**
- Node identity (Peer ID)
- Repository statistics (blocks, size)
- Network statistics (peers, bandwidth)
- Bitswap statistics (want list, pending)
- Pin statistics
- Recent connected peers

**Example:**
```bash
# Monitor with 5 second refresh
./monitor.sh 5

# Monitor with 10 second refresh
./monitor.sh 10
```

### sync.sh
Synchronizes pinned content between two IPFRS nodes.

**Usage:**
```bash
./sync.sh <source_api_url> <target_api_url>
```

**What it does:**
- Fetches pins from source node
- Exports content as CAR files
- Imports to target node
- Pins content on target
- Reports sync statistics

**Example:**
```bash
# Sync from local to remote
./sync.sh http://localhost:5001 http://remote.example.com:5001

# Sync between two remote nodes
./sync.sh http://node1:5001 http://node2:5001
```

## Prerequisites

All scripts require:
- IPFRS CLI installed and in PATH
- Bash shell (version 4.0+)
- Standard Unix utilities (awk, sed, grep, etc.)

Some scripts have additional requirements:
- `jq` for JSON parsing (monitor.sh)
- `numfmt` for human-readable sizes (batch_add.sh, monitor.sh)

## Making Scripts Executable

Before running, make scripts executable:

```bash
chmod +x *.sh
```

## Environment Variables

Scripts respect IPFRS environment variables:
- `IPFRS_PATH` - Data directory
- `IPFRS_LOG_LEVEL` - Log level
- `IPFRS_API_URL` - API endpoint
- `IPFRS_API_TOKEN` - Authentication token

## Tips and Best Practices

### 1. Regular Backups

Schedule regular backups with cron:

```bash
# Add to crontab: backup daily at 2 AM
0 2 * * * /path/to/backup.sh /mnt/backups/ipfrs >> /var/log/ipfrs_backup.log 2>&1
```

### 2. Monitoring

Run monitor script in a tmux/screen session for persistent monitoring:

```bash
tmux new-session -d -s ipfrs-monitor './monitor.sh 5'
```

### 3. Batch Operations

Combine batch_add.sh with other operations:

```bash
# Add files and automatically pin them
./batch_add.sh ./data ./cids.txt && \
  cat ./cids.txt | grep -v '^#' | awk '{print $2}' | xargs -n1 ipfrs pin add
```

### 4. Remote Sync

Use environment variables for authentication:

```bash
export IPFRS_API_TOKEN="your-secret-token"
./sync.sh http://localhost:5001 https://remote:5001
```

## Troubleshooting

### Script fails with "daemon not running"

Start the daemon first:
```bash
ipfrs daemon start
```

### Permission denied

Make script executable:
```bash
chmod +x script.sh
```

### jq command not found

Install jq:
```bash
# Ubuntu/Debian
sudo apt-get install jq

# macOS
brew install jq

# Fedora
sudo dnf install jq
```

### numfmt command not found

Install coreutils:
```bash
# macOS
brew install coreutils
```

## Customization

Feel free to modify these scripts for your specific needs. Common customizations:

1. **Backup retention**: Add logic to keep only N most recent backups
2. **Notification**: Add email/slack notifications on completion
3. **Error recovery**: Add retry logic for transient failures
4. **Parallel operations**: Use GNU parallel for faster batch operations
5. **Filtering**: Add file type filters in batch_add.sh

## Contributing

Found a bug or have an improvement? Submit an issue or pull request to the IPFRS repository.

## License

These scripts are provided as examples and are released under the same license as IPFRS.
