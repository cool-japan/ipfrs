#!/bin/bash
# IPFRS Backup Script
# Backs up repository data and exports important DAGs

set -e  # Exit on error

BACKUP_DIR="${1:-./ipfrs_backup}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "IPFRS Backup Script"
echo "==================="
echo "Backup directory: $BACKUP_DIR"
echo "Timestamp: $TIMESTAMP"
echo

# Create backup directory
mkdir -p "$BACKUP_DIR/$TIMESTAMP"

# 1. Backup configuration
echo "[1/5] Backing up configuration..."
if [ -f ~/.ipfrs/config.toml ]; then
    cp ~/.ipfrs/config.toml "$BACKUP_DIR/$TIMESTAMP/config.toml"
    echo "✓ Configuration backed up"
else
    echo "⚠ Configuration file not found"
fi

# 2. Export pins list
echo "[2/5] Exporting pins list..."
if ipfrs daemon status > /dev/null 2>&1; then
    ipfrs pin ls > "$BACKUP_DIR/$TIMESTAMP/pins.txt"
    PIN_COUNT=$(wc -l < "$BACKUP_DIR/$TIMESTAMP/pins.txt")
    echo "✓ Exported $PIN_COUNT pins"
else
    echo "⚠ Daemon not running, skipping pins export"
fi

# 3. Export repository statistics
echo "[3/5] Saving repository statistics..."
if ipfrs daemon status > /dev/null 2>&1; then
    ipfrs repo stat --format json > "$BACKUP_DIR/$TIMESTAMP/repo_stats.json"
    echo "✓ Repository statistics saved"
fi

# 4. Export important DAGs to CAR files
echo "[4/5] Exporting pinned content to CAR files..."
if [ -f "$BACKUP_DIR/$TIMESTAMP/pins.txt" ]; then
    mkdir -p "$BACKUP_DIR/$TIMESTAMP/cars"
    CAR_COUNT=0
    while IFS= read -r line; do
        CID=$(echo "$line" | awk '{print $1}')
        if [ ! -z "$CID" ]; then
            echo "  Exporting $CID..."
            ipfrs dag export "$CID" --output "$BACKUP_DIR/$TIMESTAMP/cars/${CID}.car" 2>/dev/null || true
            CAR_COUNT=$((CAR_COUNT + 1))
        fi
    done < "$BACKUP_DIR/$TIMESTAMP/pins.txt"
    echo "✓ Exported $CAR_COUNT CAR files"
fi

# 5. Create manifest
echo "[5/5] Creating backup manifest..."
cat > "$BACKUP_DIR/$TIMESTAMP/manifest.txt" <<EOF
IPFRS Backup Manifest
=====================
Timestamp: $TIMESTAMP
Date: $(date)
Hostname: $(hostname)

Contents:
- config.toml: IPFRS configuration
- pins.txt: List of pinned content
- repo_stats.json: Repository statistics
- cars/: CAR files for pinned content

To restore:
1. Copy config.toml to ~/.ipfrs/config.toml
2. Initialize IPFRS: ipfrs init
3. Import CAR files: for f in cars/*.car; do ipfrs dag import "\$f"; done
4. Re-pin content: cat pins.txt | while read cid type; do ipfrs pin add "\$cid"; done
EOF

echo "✓ Manifest created"
echo
echo "Backup completed successfully!"
echo "Backup location: $BACKUP_DIR/$TIMESTAMP"
echo
echo "To restore this backup, run:"
echo "  ./restore.sh $BACKUP_DIR/$TIMESTAMP"
