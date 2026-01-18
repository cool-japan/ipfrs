#!/bin/bash
# IPFRS Monitoring Script
# Display real-time statistics and health information

REFRESH_INTERVAL="${1:-5}"

echo "IPFRS Monitoring Dashboard"
echo "=========================="
echo "Refresh interval: ${REFRESH_INTERVAL}s (Ctrl+C to exit)"
echo

# Check if daemon is running
if ! ipfrs daemon status > /dev/null 2>&1; then
    echo "Error: IPFRS daemon is not running"
    echo "Start with: ipfrs daemon start"
    exit 1
fi

# Function to format bytes
format_bytes() {
    numfmt --to=iec-i --suffix=B "$1" 2>/dev/null || echo "$1 bytes"
}

# Function to display dashboard
display_dashboard() {
    clear
    echo "IPFRS Monitoring Dashboard"
    echo "=========================="
    echo "Last updated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo

    # Node identity
    echo "Node Information:"
    echo "-----------------"
    NODE_ID=$(ipfrs id --format json 2>/dev/null | jq -r '.id' 2>/dev/null || echo "N/A")
    echo "Peer ID: $NODE_ID"

    # Repository stats
    echo
    echo "Repository Statistics:"
    echo "----------------------"
    REPO_STATS=$(ipfrs repo stat --format json 2>/dev/null)
    if [ $? -eq 0 ]; then
        BLOCK_COUNT=$(echo "$REPO_STATS" | jq -r '.num_blocks' 2>/dev/null || echo "N/A")
        REPO_SIZE=$(echo "$REPO_STATS" | jq -r '.repo_size' 2>/dev/null || echo "0")
        echo "Blocks: $BLOCK_COUNT"
        echo "Size: $(format_bytes $REPO_SIZE)"
    else
        echo "Unable to fetch repository statistics"
    fi

    # Network stats
    echo
    echo "Network Statistics:"
    echo "-------------------"
    BW_STATS=$(ipfrs stats bw --format json 2>/dev/null)
    if [ $? -eq 0 ]; then
        PEER_COUNT=$(echo "$BW_STATS" | jq -r '.peers' 2>/dev/null || echo "0")
        echo "Connected peers: $PEER_COUNT"
    else
        echo "Unable to fetch network statistics"
    fi

    # Bitswap stats
    BITSWAP_STATS=$(ipfrs stats bitswap --format json 2>/dev/null)
    if [ $? -eq 0 ]; then
        WANT_LIST=$(echo "$BITSWAP_STATS" | jq -r '.wantlist_size' 2>/dev/null || echo "0")
        PENDING=$(echo "$BITSWAP_STATS" | jq -r '.pending_requests' 2>/dev/null || echo "0")
        echo "Want list: $WANT_LIST"
        echo "Pending requests: $PENDING"
    fi

    # Pin stats
    echo
    echo "Pin Statistics:"
    echo "---------------"
    PIN_COUNT=$(ipfrs pin ls 2>/dev/null | wc -l)
    echo "Pinned items: $PIN_COUNT"

    # Recent peers
    echo
    echo "Recent Peers:"
    echo "-------------"
    ipfrs swarm peers 2>/dev/null | head -5 | while read peer; do
        PEER_ID=$(echo "$peer" | rev | cut -d'/' -f1 | rev | cut -c1-20)
        echo "  ${PEER_ID}..."
    done

    echo
    echo "Press Ctrl+C to exit | Refresh: ${REFRESH_INTERVAL}s"
}

# Main monitoring loop
trap "echo; echo 'Monitoring stopped'; exit 0" INT TERM

while true; do
    display_dashboard
    sleep "$REFRESH_INTERVAL"
done
