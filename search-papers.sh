#!/usr/bin/env bash
# search-papers.sh — Search the arXiv paper collection
# Usage: ./search-papers.sh <query>
# Examples:
#   ./search-papers.sh mamba          # search by keyword
#   ./search-papers.sh 2604.14191     # search by arxiv ID
#   ./search-papers.sh 开山           # find all landmark papers
#   ./search-papers.sh "2026.*quant"  # regex: 2026 quantization papers

set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <query>"
    echo "  Search arXiv paper collection by keyword, arxiv ID, or regex"
    exit 1
fi

QUERY="$1"

if echo "$QUERY" | grep -qP '^\d{4}\.\d{4,5}$'; then
    # Exact arxiv ID search
    echo "=== Searching for arxiv ID: $QUERY ==="
    results=$(grep -rn "abs/$QUERY" "$DIR"/*-arxiv-urls.md 2>/dev/null || true)
    # Also check if a note exists for this arxiv ID
    note_path=$(find "$DIR/notes" -name "$QUERY.md" 2>/dev/null | head -1)
    if [ -n "$note_path" ]; then
        note_topic=$(basename "$(dirname "$note_path")")
        echo "  Note found: notes/$note_topic/$QUERY.md"
    fi
    if [ -z "$results" ]; then
        echo "  No results found."
    else
        echo "$results" | while IFS= read -r line; do
            file=$(echo "$line" | cut -d: -f1)
            topic=$(basename "$file" | sed 's/-arxiv-urls.md//' | sed 's/-/ /g')
            # Extract the description column (2nd column in markdown table)
            # Handles both plain URL and markdown link [url](url) formats
            desc=$(echo "$line" | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/, "", $3); print $3}')
            url="https://arxiv.org/abs/$QUERY"
            echo "  [$topic] $desc"
            echo "    $url"
        done
    fi
else
    # Keyword/regex search (case-insensitive)
    echo "=== Searching for: $QUERY ==="
    results=$(grep -rni "$QUERY" "$DIR"/*-arxiv-urls.md 2>/dev/null | grep "arxiv.org" || true)
    if [ -z "$results" ]; then
        echo "  No results found."
    else
        echo "$results" | while IFS= read -r line; do
            file=$(echo "$line" | cut -d: -f1)
            topic=$(basename "$file" | sed 's/-arxiv-urls.md//' | sed 's/-/ /g')
            # Extract description (after the description | separator)
            content=$(echo "$line" | sed 's/^[^:]*:[0-9]*://' | sed 's/.*| //' | sed 's/ |.*//')
            # Extract URL - handle both plain and markdown link format, take only first
            url=$(echo "$line" | grep -oP 'https://arxiv.org/abs/\d+\.\d+' | head -1)
            echo "  [$topic] $content"
            echo "    $url"
        done
    fi
fi
