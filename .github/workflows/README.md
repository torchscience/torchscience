# GitHub Actions Workflows

This directory contains CI/CD workflows for the torch-science project.

## Workflows

### test.yml - Test Suite

Comprehensive testing across multiple Python versions, PyTorch versions, and backends.

**Jobs:**

1. **test-cpu**: Tests CPU backend across all platforms
   - OS: Ubuntu, macOS, Windows
   - Python: 3.10, 3.11, 3.12, 3.13, 3.14
   - PyTorch: 2.9.0
   - Runs full test suite with pytest

2. **test-cuda**: Tests CUDA backend on self-hosted GPU runner
   - Runner: torchscience-linux-cuda (NVIDIA Tesla T4)
   - Python: 3.10, 3.11, 3.12, 3.13, 3.14
   - PyTorch: 2.9.0
   - Real GPU hardware with 16 GB VRAM
   - Tests CUDA kernels and operators on actual GPU

3. **test-mps**: Tests MPS (Apple Silicon) compilation
   - Python: 3.11, 3.12, 3.13, 3.14
   - PyTorch: 2.9.0
   - Note: GitHub Actions macOS runners are Intel-based, so MPS runtime tests are skipped
   - Verifies MPS code compiles correctly

4. **test-sparse**: Tests sparse tensor backends
   - Python: 3.11, 3.12, 3.13, 3.14
   - PyTorch: 2.9.0
   - Tests SparseCPU implementation

5. **integration-test**: End-to-end integration testing
   - Tests PT2 compliance with opcheck
   - Tests autograd functionality
   - Tests torch.compile compatibility
   - Runs after test-cpu passes

**Commented Out Jobs:**

- **test-hip**: ROCm/HIP backend testing (requires self-hosted AMD GPU runner)
  - Uncomment and configure if you have access to AMD GPU runners
  - See comments in test.yml for setup instructions

### lint.yml - Code Quality

Code formatting and linting checks.

**Jobs:**

1. **pre-commit**: Runs all pre-commit hooks
2. **clang-format**: C++ code formatting verification
3. **python-lint**: Python linting with ruff and mypy
4. **cpp-lint**: C++ linting with clang-tidy (basic check)
5. **build-check**: Verifies project builds on all platforms

## Backend Testing Matrix

### CPU Backend
- ✅ Fully tested on all platforms (Linux, macOS, Windows)
- ✅ Python versions 3.10-3.14
- ✅ PyTorch 2.9.0

### CUDA Backend
- ✅ Tested on self-hosted NVIDIA Tesla T4 GPU runner
- ✅ Real GPU hardware testing (16 GB VRAM, Compute 7.5)
- ✅ PyTorch 2.9.0 with CUDA 12.4
- ✅ Ubuntu NVIDIA GPU-Optimized Image
- ✅ Full GPU runtime testing (not just compilation)

### MPS Backend (Apple Silicon)
- ⚠️ Compilation tested on Intel macOS
- ⚠️ Runtime tests skipped (no Apple Silicon runners available)
- 💡 Recommend testing manually on Apple Silicon hardware

### HIP/ROCm Backend (AMD GPUs)
- ❌ Not tested in CI (no AMD GPU runners)
- 💡 Self-hosted runner required
- 💡 Uncomment test-hip job if you have AMD GPU access

### Sparse Backends
- ✅ SparseCPU tested
- ⚠️ SparseCUDA covered by CUDA tests

## Running Workflows

### Automatic Triggers
Workflows run automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

### Manual Triggers
You can manually trigger workflows from the GitHub Actions tab using the "workflow_dispatch" event.

## Local Testing

To replicate CI behavior locally:

### CPU Testing
```bash
# Install dependencies
uv pip install torch pytest pytest-xdist numpy

# Build extension
USE_CUDA=0 uv pip install -e .

# Run tests
pytest tests/ -v -n auto
```

### CUDA Testing (Linux with CUDA)
```bash
# Install PyTorch with CUDA
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# Build with CUDA
USE_CUDA=1 uv pip install -e .

# Run tests
pytest tests/ -v
```

### macOS Testing
```bash
# Install Homebrew LLVM
brew install llvm

# Set compiler
export CC=$(brew --prefix llvm)/bin/clang
export CXX=$(brew --prefix llvm)/bin/clang++

# Build
USE_CUDA=0 uv pip install -e .

# Run tests
pytest tests/ -v
```

### Code Quality Checks
```bash
# Install tools
pip install pre-commit ruff mypy

# Run checks
pre-commit run --all-files
ruff check src/ tests/
mypy src/torchscience
```

## Optimization Tips

### Reducing CI Time

The current matrix excludes some combinations to reduce CI time:
- macOS: Reduced Python versions (3.10, 3.11, 3.12)
- Windows: Reduced PyTorch versions (2.4.0, latest)

To test all combinations:
1. Remove exclusions from the matrix
2. Note: This will significantly increase CI time

### Caching

Consider adding caching for:
- PyTorch installation (large download)
- pip/uv cache
- CUDA toolkit (very large download)

Example cache step:
```yaml
- name: Cache PyTorch
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/setup.py') }}
```

## Troubleshooting

### CUDA Tests Failing
- Check CUDA Toolkit installation logs
- Verify PyTorch CUDA version matches toolkit version
- Check `TORCH_CUDA_ARCH_LIST` environment variable

### macOS Build Failures
- Ensure Homebrew LLVM is installed
- Verify `CC` and `CXX` environment variables are set
- Check SDK path in setup.py is valid

### Windows Build Failures
- May need Visual Studio Build Tools
- Check MSVC compiler version compatibility
- Ensure PATH includes necessary tools

### MPS Tests Skipped
- Expected behavior on Intel Macs
- MPS is only available on Apple Silicon
- Compilation is still verified

## Adding New Tests

When adding new operators:

1. Add tests to `tests/torchscience/test_<operator>.py`
2. Include opcheck tests for PT2 compliance
3. Test multiple dtypes and devices
4. Include autograd tests if applicable
5. CI will automatically run new tests

## Self-Hosted Runners

For specialized hardware testing (AMD GPUs, Apple Silicon):

1. Set up self-hosted runner on your hardware
2. Tag runner appropriately (e.g., `[self-hosted, linux, amd-gpu]`)
3. Uncomment relevant job in test.yml
4. Configure job to use your runner tags

See: https://docs.github.com/en/actions/hosting-your-own-runners

## Performance Testing

For performance regression testing, consider adding:
- Benchmark suite with pytest-benchmark
- Separate workflow for performance tests
- Comparison against baseline metrics
- Alerts for significant regressions

## Security Scanning

Consider adding:
- CodeQL analysis for security vulnerabilities
- Dependency scanning with Dependabot
- SAST tools for C++ code

## Badge Status

Add workflow status badges to README.md:

```markdown
![Test Suite](https://github.com/0x00b1/torch-science/workflows/Test%20Suite/badge.svg)
![Code Quality](https://github.com/0x00b1/torch-science/workflows/Code%20Quality/badge.svg)
```
