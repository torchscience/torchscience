# GitHub Actions Workflows Quick Reference

## Overview

This project uses GitHub Actions for comprehensive CI/CD testing across multiple:
- **Python versions**: 3.9, 3.10, 3.11, 3.12
- **PyTorch versions**: 2.2.0, 2.3.0, 2.4.0, latest
- **Platforms**: Linux, macOS, Windows
- **Backends**: CPU, CUDA, MPS, sparse tensors, (ROCm/HIP with self-hosted runners)

## Workflows

| Workflow | Purpose | Trigger |
|----------|---------|---------|
| `test.yml` | Comprehensive test suite | Push/PR to main/develop |
| `lint.yml` | Code quality checks | Push/PR to main/develop |
| `build.yml` | Build and publish packages | Release or manual |

## Test Matrix Coverage

### ✅ Fully Tested in CI

| Backend | Platforms | Python Versions | PyTorch Version |
|---------|-----------|-----------------|-----------------|
| CPU | Linux, macOS, Windows | 3.10-3.14 | 2.9.0 |
| CUDA | Self-hosted Tesla T4 | 3.10-3.14 | 2.9.0 |
| SparseCPU | Linux | 3.11-3.14 | 2.9.0 |

### ⚠️ Partially Tested

| Backend | Status | Notes |
|---------|--------|-------|
| MPS | Compilation only | GitHub Actions has Intel Macs; runtime tests skipped |
| SparseCUDA | Via CUDA tests | Covered by main CUDA test suite |

### 🔧 Requires Self-Hosted Runner

| Backend | Requirements |
|---------|--------------|
| HIP/ROCm | AMD GPU with ROCm installed |
| MPS Runtime | Apple Silicon Mac |

## Quick Commands

### Run Tests Locally

```bash
# CPU tests
pytest tests/ -v -n auto

# CUDA tests (if you have CUDA)
pytest tests/ -v -k cuda

# Sparse tests
pytest tests/ -v -k sparse

# Skip slow tests
pytest tests/ -v -m "not slow"
```

### Trigger Workflows Manually

1. Go to **Actions** tab in GitHub
2. Select workflow (e.g., "Test Suite")
3. Click **Run workflow**
4. Select branch and configure options

### Check Workflow Status

```bash
# Install GitHub CLI
gh workflow list

# View runs
gh run list

# Watch a specific run
gh run watch
```

## Badge Examples

Add to your README.md:

```markdown
[![Test Suite](https://github.com/0x00b1/torch-science/actions/workflows/test.yml/badge.svg)](https://github.com/0x00b1/torch-science/actions/workflows/test.yml)
[![Code Quality](https://github.com/0x00b1/torch-science/actions/workflows/lint.yml/badge.svg)](https://github.com/0x00b1/torch-science/actions/workflows/lint.yml)
[![Build](https://github.com/0x00b1/torch-science/actions/workflows/build.yml/badge.svg)](https://github.com/0x00b1/torch-science/actions/workflows/build.yml)
```

## Test Coverage by Job

### test-cpu (6-8 min)
- ✅ All platforms (Linux, macOS, Windows)
- ✅ Python 3.10-3.14
- ✅ PyTorch 2.9.0
- ✅ 15 combinations (3 platforms × 5 Python versions)

### test-cuda (8-10 min)
- ✅ Self-hosted NVIDIA Tesla T4 GPU runner
- ✅ Python 3.10-3.14
- ✅ PyTorch 2.9.0
- ✅ Real GPU hardware testing
- ✅ 5 combinations (5 Python versions)

### test-mps (3-5 min)
- ✅ macOS compilation check
- ✅ Python 3.11-3.14
- ✅ PyTorch 2.9.0
- ⚠️ Runtime tests skipped
- ✅ 4 combinations

### test-sparse (2-3 min)
- ✅ SparseCPU implementation
- ✅ Python 3.11-3.14
- ✅ PyTorch 2.9.0
- ✅ 4 combinations

### integration-test (5-10 min)
- ✅ PT2 compliance (opcheck)
- ✅ Autograd functionality
- ✅ torch.compile compatibility
- ✅ Full test suite with durations

## Publishing Workflow

### Automatic (on Release)

1. Create a new release on GitHub
2. Workflows automatically:
   - Build source distribution
   - Build wheels for all platforms
   - Test all wheels
   - Publish to PyPI
   - Attach artifacts to release

### Manual Testing

```bash
# Trigger build workflow manually
gh workflow run build.yml

# Download artifacts
gh run download <run-id>
```

### Test PyPI

```bash
# Publish to Test PyPI for testing
gh workflow run build.yml -f publish=false

# Install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ torchscience
```

## Adding Self-Hosted Runners

### For ROCm/HIP Testing

1. **Setup AMD GPU machine**:
   ```bash
   # Install ROCm
   # See: https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html
   ```

2. **Add GitHub runner**:
   - Go to Settings → Actions → Runners
   - Click "New self-hosted runner"
   - Follow instructions
   - Add labels: `self-hosted, linux, amd-gpu`

3. **Uncomment test-hip job** in `.github/workflows/test.yml`

4. **Verify**:
   ```bash
   # Check runner is online
   gh api repos/0x00b1/torch-science/actions/runners
   ```

### For Apple Silicon MPS Testing

1. **Setup Apple Silicon Mac**
2. **Add runner with labels**: `self-hosted, macos, arm64`
3. **Update test-mps job** to use self-hosted runner

## Debugging Failed Workflows

### Common Issues

**macOS build failures**:
```bash
# Check Homebrew LLVM installation
brew install llvm
export CC=$(brew --prefix llvm)/bin/clang
export CXX=$(brew --prefix llvm)/bin/clang++
```

**CUDA out of memory**:
```bash
# Reduce test parallelization
pytest tests/ -n 2  # Instead of -n auto
```

**Windows path issues**:
- Ensure paths use forward slashes or proper escaping
- Check Visual Studio Build Tools are installed

### View Logs

```bash
# Get specific run logs
gh run view <run-id> --log

# Download logs
gh run download <run-id>
```

## Optimization Tips

### Reduce CI Time

1. **Use matrix exclusions** (already configured)
2. **Add caching**:
   ```yaml
   - uses: actions/cache@v3
     with:
       path: ~/.cache/pip
       key: ${{ runner.os }}-pip-${{ hashFiles('**/setup.py') }}
   ```

3. **Conditional jobs**:
   ```yaml
   if: contains(github.event.head_commit.message, '[full-ci]')
   ```

### Parallel Execution

Jobs run in parallel by default. Current setup:
- CPU tests: 48 parallel jobs
- CUDA tests: 12 parallel jobs
- Total: ~60 jobs can run concurrently (GitHub limits apply)

## Cost Considerations

### GitHub Actions Free Tier

- **Public repos**: Unlimited minutes
- **Private repos**: 2,000 minutes/month

### Current Usage (Estimated)

Per workflow run:
- test-cpu: ~400 minutes (48 jobs × ~8 min)
- test-cuda: ~150 minutes (12 jobs × ~12 min)
- lint: ~20 minutes
- Total: ~570 minutes per full CI run

For private repos, consider:
- Reducing matrix size
- Using caching aggressively
- Running full tests only on main branch

## Security

### Secrets Required

For publishing:
- `PYPI_API_TOKEN` (optional, uses trusted publishing)
- `GITHUB_TOKEN` (automatically provided)

### Trusted Publishing (Recommended)

1. Go to https://pypi.org/manage/project/torchscience/settings/publishing/
2. Add GitHub Actions as trusted publisher
3. No API token needed!

## Best Practices

✅ **DO**:
- Test on all target platforms before merging
- Use matrix exclusions to reduce redundant testing
- Add PT2 compliance tests (opcheck) for new operators
- Update test matrix when adding new backends

❌ **DON'T**:
- Skip tests for "quick fixes"
- Commit without running pre-commit hooks
- Ignore warnings in logs (they often indicate real issues)
- Mix CPU and CUDA tests in same job

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [PyTorch Custom Operators Manual](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
- [Self-Hosted Runners](https://docs.github.com/en/actions/hosting-your-own-runners)
- [Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)

## Support

For issues with workflows:
1. Check logs in GitHub Actions tab
2. Review `.github/workflows/README.md`
3. Open an issue with `[CI]` prefix
4. Include workflow run URL and error logs
