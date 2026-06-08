#!/usr/bin/env bash
# Dispatch the Phase 10 fuzz marathon (24 h × 7 libFuzzer targets).
# Requires: gh CLI authenticated (gh auth login).
set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
    echo "error: gh CLI not found. Install GitHub CLI or trigger fuzz-24h.yml from the Actions UI." >&2
    exit 1
fi

gh workflow run fuzz-24h.yml
echo "Dispatched fuzz-24h.yml"
echo "Monitor: gh run list --workflow=fuzz-24h.yml"
echo "Watch:   gh run watch"
