#!/bin/bash
# IPFRS Batch Add Script
# Add multiple files and track their CIDs

set -e  # Exit on error

SOURCE_DIR="$1"
OUTPUT_FILE="${2:-./cids.txt}"

if [ -z "$SOURCE_DIR" ] || [ ! -d "$SOURCE_DIR" ]; then
    echo "Usage: $0 <source_directory> [output_file]"
    echo "Example: $0 ./my_files ./cids.txt"
    exit 1
fi

echo "IPFRS Batch Add Script"
echo "======================"
echo "Source directory: $SOURCE_DIR"
echo "Output file: $OUTPUT_FILE"
echo

# Start daemon if not running
if ! ipfrs daemon status > /dev/null 2>&1; then
    echo "Starting daemon..."
    ipfrs daemon start
    sleep 2
fi

# Create output file with header
cat > "$OUTPUT_FILE" <<EOF
# IPFRS Batch Add Results
# Generated: $(date)
# Source: $SOURCE_DIR
# Format: FILENAME CID SIZE
EOF

# Count files
FILE_COUNT=$(find "$SOURCE_DIR" -type f | wc -l)
echo "Found $FILE_COUNT files to add"
echo

# Add files and track progress
CURRENT=0
TOTAL_SIZE=0

find "$SOURCE_DIR" -type f | while IFS= read -r file; do
    CURRENT=$((CURRENT + 1))
    REL_PATH="${file#$SOURCE_DIR/}"
    FILE_SIZE=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)

    echo "[$CURRENT/$FILE_COUNT] Adding: $REL_PATH"

    # Add file and capture CID
    CID=$(ipfrs add --quiet "$file" 2>/dev/null)

    if [ $? -eq 0 ]; then
        echo "$REL_PATH $CID $FILE_SIZE" >> "$OUTPUT_FILE"
        TOTAL_SIZE=$((TOTAL_SIZE + FILE_SIZE))
        echo "  ✓ $CID"
    else
        echo "  ✗ Failed to add $file"
    fi
done

echo
echo "Batch add completed!"
echo "Results saved to: $OUTPUT_FILE"
echo "Files added: $CURRENT"
echo "Total size: $(numfmt --to=iec-i --suffix=B $TOTAL_SIZE 2>/dev/null || echo "$TOTAL_SIZE bytes")"
echo
echo "Pin all added content with:"
echo "  cat $OUTPUT_FILE | grep -v '^#' | awk '{print \$2}' | xargs -n1 ipfrs pin add"
