#!/bin/bash
# Install git hooks for quant-tools

HOOKS_DIR="$(cd "$(dirname "$0")/.." && pwd)/.git/hooks"

if [ ! -d "$HOOKS_DIR" ]; then
    echo "Error: .git/hooks directory not found. Is this a git repo?"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cp "$SCRIPT_DIR/pre-commit" "$HOOKS_DIR/pre-commit"
chmod +x "$HOOKS_DIR/pre-commit"

echo "Pre-commit hook installed successfully."
