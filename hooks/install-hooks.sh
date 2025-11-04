#!/bin/bash
# Installation script for Git hooks
# Run this script from the repository root: ./hooks/install-hooks.sh

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🔧 Installing Git hooks for torch-science..."
echo ""

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "Error: Not in a git repository root. Please run from repository root."
    exit 1
fi

# Method 1: Install pre-commit framework (recommended)
echo "Method 1: pre-commit framework (recommended)"
echo "-------------------------------------------"

if command -v pre-commit &> /dev/null; then
    echo "${GREEN}✓${NC} pre-commit is already installed"
else
    echo "${YELLOW}!${NC} pre-commit is not installed"
    echo "  Install with: pip install pre-commit"
    echo "  Or: uv pip install pre-commit"
fi

echo ""
read -p "Install hooks using pre-commit framework? [Y/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    if command -v pre-commit &> /dev/null; then
        pre-commit install
        echo "${GREEN}✓${NC} pre-commit hooks installed"
        echo ""
        echo "Testing hooks on all files..."
        pre-commit run --all-files || true
    else
        echo "Please install pre-commit first:"
        echo "  pip install pre-commit"
        echo "  # or"
        echo "  uv pip install pre-commit"
        echo ""
        echo "Then run this script again or run: pre-commit install"
    fi
fi

echo ""
echo "Method 2: Native Git hooks (alternative)"
echo "---------------------------------------"
read -p "Also install native Git hook? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Create symlink for native hook
    if [ -f ".git/hooks/pre-commit" ]; then
        echo "${YELLOW}!${NC} .git/hooks/pre-commit already exists"
        read -p "Overwrite? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm .git/hooks/pre-commit
            ln -s ../../hooks/pre-commit .git/hooks/pre-commit
            echo "${GREEN}✓${NC} Native Git hook installed (symlink)"
        fi
    else
        ln -s ../../hooks/pre-commit .git/hooks/pre-commit
        echo "${GREEN}✓${NC} Native Git hook installed (symlink)"
    fi
fi

echo ""
echo "✅ Hook installation complete!"
echo ""
echo "Usage:"
echo "  - Hooks run automatically on 'git commit'"
echo "  - Run manually: pre-commit run --all-files"
echo "  - Skip hooks: git commit --no-verify"
echo ""
echo "Required tools:"
echo "  - Ruff (Python): pip install ruff"
echo "  - clang-format (C++): brew install clang-format (macOS)"
