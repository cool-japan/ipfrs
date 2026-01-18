#!/bin/bash
# IPFRS Sync Script
# Synchronize pinned content between two IPFRS nodes

set -e  # Exit on error

SOURCE_API="${1}"
TARGET_API="${2}"

if [ -z "$SOURCE_API" ] || [ -z "$TARGET_API" ]; then
    echo "Usage: $0 <source_api_url> <target_api_url>"
    echo "Example: $0 http://localhost:5001 http://remote:5001"
    echo
    echo "This script syncs pinned content from source to target node"
    exit 1
fi

echo "IPFRS Sync Script"
echo "================="
echo "Source: $SOURCE_API"
echo "Target: $TARGET_API"
echo

# Function to call ipfrs with specific API
ipfrs_source() {
    IPFRS_API_URL="$SOURCE_API" ipfrs "$@"
}

ipfrs_target() {
    IPFRS_API_URL="$TARGET_API" ipfrs "$@"
}

# Verify both nodes are accessible
echo "Verifying nodes..."
if ! ipfrs_source id > /dev/null 2>&1; then
    echo "Error: Cannot connect to source node at $SOURCE_API"
    exit 1
fi

if ! ipfrs_target id > /dev/null 2>&1; then
    echo "Error: Cannot connect to target node at $TARGET_API"
    exit 1
fi

echo "✓ Both nodes accessible"
echo

# Get source pins
echo "Fetching pins from source..."
TEMP_PINS=$(mktemp)
ipfrs_source pin ls > "$TEMP_PINS"
PIN_COUNT=$(wc -l < "$TEMP_PINS")
echo "Found $PIN_COUNT pinned items on source"
echo

# Sync each pin to target
echo "Syncing content to target..."
SYNCED=0
SKIPPED=0
FAILED=0

while IFS= read -r line; do
    CID=$(echo "$line" | awk '{print $1}')

    if [ -z "$CID" ] || [ "$CID" = "CID" ]; then
        continue
    fi

    echo "Processing $CID..."

    # Check if already exists on target
    if ipfrs_target block stat "$CID" > /dev/null 2>&1; then
        echo "  ⊙ Already exists on target, re-pinning"
        ipfrs_target pin add "$CID" > /dev/null 2>&1 || true
        SKIPPED=$((SKIPPED + 1))
    else
        # Export from source
        TEMP_CAR=$(mktemp -u).car
        echo "  ↓ Exporting from source..."
        if ipfrs_source dag export "$CID" --output "$TEMP_CAR" > /dev/null 2>&1; then
            # Import to target
            echo "  ↑ Importing to target..."
            if ipfrs_target dag import "$TEMP_CAR" > /dev/null 2>&1; then
                # Pin on target
                echo "  📌 Pinning on target..."
                ipfrs_target pin add "$CID" > /dev/null 2>&1 || true
                echo "  ✓ Synced successfully"
                SYNCED=$((SYNCED + 1))
            else
                echo "  ✗ Failed to import"
                FAILED=$((FAILED + 1))
            fi
            rm -f "$TEMP_CAR"
        else
            echo "  ✗ Failed to export"
            FAILED=$((FAILED + 1))
        fi
    fi
done < "$TEMP_PINS"

# Cleanup
rm -f "$TEMP_PINS"

echo
echo "Sync completed!"
echo "==============="
echo "Total pins: $PIN_COUNT"
echo "Synced: $SYNCED"
echo "Skipped (already exist): $SKIPPED"
echo "Failed: $FAILED"
echo

if [ $FAILED -gt 0 ]; then
    echo "⚠ Some items failed to sync. Check the logs above for details."
    exit 1
fi
