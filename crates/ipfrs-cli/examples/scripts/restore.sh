#!/bin/bash
# IPFRS Restore Script
# Restores repository data from backup

set -e  # Exit on error

BACKUP_DIR="$1"

if [ -z "$BACKUP_DIR" ] || [ ! -d "$BACKUP_DIR" ]; then
    echo "Usage: $0 <backup_directory>"
    echo "Example: $0 ./ipfrs_backup/20260109_123456"
    exit 1
fi

echo "IPFRS Restore Script"
echo "===================="
echo "Backup directory: $BACKUP_DIR"
echo

# Verify backup
if [ ! -f "$BACKUP_DIR/manifest.txt" ]; then
    echo "Error: Invalid backup directory (manifest.txt not found)"
    exit 1
fi

echo "Backup manifest:"
echo "----------------"
head -10 "$BACKUP_DIR/manifest.txt"
echo "----------------"
echo

read -p "Proceed with restore? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Restore cancelled"
    exit 0
fi

# 1. Restore configuration
echo "[1/4] Restoring configuration..."
if [ -f "$BACKUP_DIR/config.toml" ]; then
    mkdir -p ~/.ipfrs
    cp "$BACKUP_DIR/config.toml" ~/.ipfrs/config.toml
    echo "✓ Configuration restored"
else
    echo "⚠ Configuration file not found in backup"
fi

# 2. Initialize repository if needed
echo "[2/4] Initializing repository..."
if [ ! -d ~/.ipfrs ]; then
    ipfrs init
    echo "✓ Repository initialized"
else
    echo "✓ Repository already exists"
fi

# 3. Import CAR files
echo "[3/4] Importing CAR files..."
if [ -d "$BACKUP_DIR/cars" ]; then
    CAR_COUNT=0
    for car_file in "$BACKUP_DIR/cars"/*.car; do
        if [ -f "$car_file" ]; then
            echo "  Importing $(basename "$car_file")..."
            ipfrs dag import "$car_file" || true
            CAR_COUNT=$((CAR_COUNT + 1))
        fi
    done
    echo "✓ Imported $CAR_COUNT CAR files"
else
    echo "⚠ No CAR files found in backup"
fi

# 4. Restore pins
echo "[4/4] Restoring pins..."
if [ -f "$BACKUP_DIR/pins.txt" ]; then
    # Start daemon if not running
    if ! ipfrs daemon status > /dev/null 2>&1; then
        echo "  Starting daemon..."
        ipfrs daemon start
        sleep 2
    fi

    PIN_COUNT=0
    while IFS= read -r line; do
        CID=$(echo "$line" | awk '{print $1}')
        if [ ! -z "$CID" ] && [ "$CID" != "CID" ]; then
            echo "  Pinning $CID..."
            ipfrs pin add "$CID" 2>/dev/null || true
            PIN_COUNT=$((PIN_COUNT + 1))
        fi
    done < "$BACKUP_DIR/pins.txt"
    echo "✓ Restored $PIN_COUNT pins"
else
    echo "⚠ Pins list not found in backup"
fi

echo
echo "Restore completed successfully!"
echo
echo "Summary:"
echo "- Configuration: $([ -f "$BACKUP_DIR/config.toml" ] && echo "✓" || echo "⚠")"
echo "- CAR files: $CAR_COUNT imported"
echo "- Pins: $PIN_COUNT restored"
echo
echo "Verify restoration with:"
echo "  ipfrs repo stat"
echo "  ipfrs pin ls"
