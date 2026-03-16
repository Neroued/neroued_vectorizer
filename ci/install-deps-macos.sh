#!/bin/bash
set -euo pipefail

# ── CI dependency installer for macOS ─────────────────────────────────────────

brew install opencv potrace

echo "=== macOS dependency installation complete ==="
