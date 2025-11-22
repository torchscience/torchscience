# Git Hooks for torch-science

This directory contains Git hooks for automatic code formatting and linting.

## Quick Start

```bash
# Install hooks (from repository root)
./hooks/install-hooks.sh

# Or manually install pre-commit framework
pip install pre-commit
pre-commit install
```

## What Gets Checked

### Python Files (`.py`)
- **Ruff**: Linting and formatting
  - Checks code style (PEP 8)
  - Auto-fixes common issues
  - Formats code consistently
  - Sorts imports

### C++/CUDA/Metal Files (`.cpp`, `.h`, `.cu`, `.mm`)
- **clang-format**: Code formatting
  - Enforces consistent style
  - Based on Google style with PyTorch adjustments
  - 100 character line limit
  - 4-space indentation

## Two Installation Methods

### Method 1: pre-commit Framework (Recommended)

The pre-commit framework provides:
- Automatic hook management
- Easy updates
- Consistent behavior across systems
- Additional checks (trailing whitespace, YAML/TOML syntax, etc.)

**Installation:**
```bash
pip install pre-commit
pre-commit install
```

**Usage:**
```bash
# Hooks run automatically on commit
git commit -m "Your message"

# Run manually on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files src/file.py

# Update hooks to latest versions
pre-commit autoupdate

# Skip hooks temporarily
git commit --no-verify
```

**Configuration:** `.pre-commit-config.yaml`

### Method 2: Native Git Hooks

Traditional Git hooks without external dependencies.

**Installation:**
```bash
./hooks/install-hooks.sh
# Select option 2 when prompted
```

**Usage:**
```bash
# Hooks run automatically on commit
git commit -m "Your message"

# Skip hooks
git commit --no-verify
```

**Location:** `.git/hooks/pre-commit` → `hooks/pre-commit` (symlink)

## Required Tools

### Ruff (Python)
```bash
# Using pip
pip install ruff

# Using uv
uv pip install ruff

# Check installation
ruff --version
```

### clang-format (C++/CUDA/Metal)
```bash
# macOS
brew install clang-format

# Ubuntu/Debian
sudo apt-get install clang-format

# Check installation
clang-format --version
```

## Configuration Files

- `.clang-format` - C++/CUDA/Metal formatting rules
- `.pre-commit-config.yaml` - pre-commit framework configuration
- `pyproject.toml` - Ruff configuration (under `[tool.ruff]`)

## Customization

### Adjusting Python Rules

Edit `pyproject.toml`:
```toml
[tool.ruff]
line-length = 100

lint.select = ["E", "F", "I", ...]
lint.ignore = ["E501", ...]
```

### Adjusting C++ Rules

Edit `.clang-format`:
```yaml
ColumnLimit: 100
IndentWidth: 4
...
```

### Adjusting pre-commit Behavior

Edit `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.4
    hooks:
      - id: ruff
        args: [--fix]  # Remove to disable auto-fix
```

## Running Formatters Manually

### Format Python files
```bash
# Check only
ruff check src/

# Check and auto-fix
ruff check --fix src/

# Format
ruff format src/
```

### Format C++ files
```bash
# Single file
clang-format -i src/torchscience/csrc/ops/cpu/example_kernel.cpp

# Multiple files
find src/torchscience/csrc -name "*.cpp" -o -name "*.cu" -o -name "*.mm" | \
  xargs clang-format -i
```

## Troubleshooting

### Hook installation fails
```bash
# Check Git repository
git status

# Verify you're in repository root
ls -la | grep .git

# Re-run installation
./hooks/install-hooks.sh
```

### Ruff not found
```bash
# Install in current environment
pip install ruff

# Or use uv
uv pip install ruff

# Verify
which ruff
```

### clang-format not found
```bash
# macOS
brew install clang-format

# Linux
sudo apt-get install clang-format

# Verify
which clang-format
```

### Skip hooks for emergency commits
```bash
git commit --no-verify -m "Emergency fix"
```

## CI Integration

The `.pre-commit-config.yaml` is configured for pre-commit.ci integration. When enabled:
- Automatically runs on all PRs
- Auto-fixes formatting issues
- Creates fixup commits if needed

To enable, visit: https://pre-commit.ci

## Best Practices

1. **Install hooks early**: Run `./hooks/install-hooks.sh` right after cloning
2. **Keep tools updated**: `pre-commit autoupdate` periodically
3. **Don't skip hooks**: Only use `--no-verify` for emergencies
4. **Fix issues promptly**: Don't let formatting errors accumulate
5. **Test before commit**: Run `pre-commit run --all-files` after major changes

## Additional Checks (pre-commit framework only)

When using pre-commit framework, you also get:
- ✅ Large file detection (>1MB)
- ✅ Merge conflict marker detection
- ✅ YAML/TOML syntax validation
- ✅ Trailing whitespace removal
- ✅ End-of-file newline enforcement
- ✅ Debug statement detection

## See Also

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [clang-format Documentation](https://clang.llvm.org/docs/ClangFormat.html)
- [pre-commit Framework](https://pre-commit.com/)
- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
