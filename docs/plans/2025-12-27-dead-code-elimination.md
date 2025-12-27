# Dead Code Elimination Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove all dead code from torchscience - unreferenced files, unused templates, orphaned headers, and stale includes.

**Architecture:** Systematic elimination of dead code in three phases: (1) commit already-staged deletions, (2) delete newly-identified dead files, (3) verify build succeeds.

**Tech Stack:** C++, CMake, Python

---

## Summary of Dead Code Identified

### Already Staged for Deletion (11 files)
These files have been deleted but not yet committed:
- `benchmark/torchscience/csrc/impl/special_functions/polygamma_benchmark.cpp`
- `src/torchscience/csrc/cpu/all_operators.h`
- `src/torchscience/csrc/impl/reduction_traits.h`
- `src/torchscience/csrc/impl/signal_processing/waveform/sine_wave.h`
- `src/torchscience/csrc/impl/signal_processing/window_function/rectangular_window.h`
- `src/torchscience/csrc/impl/special_functions/factorial_backward_backward.h`
- `src/torchscience/csrc/impl/special_functions/polygamma.h`
- `src/torchscience/csrc/impl/special_functions/tensor_iterator_config.h`
- `src/torchscience/csrc/impl/statistics/descriptive/kurtosis_traits.h`
- `src/torchscience/csrc/impl/waveform/sine_wave_traits.h`
- `tests/torchscience/csrc/impl/special_functions/polygamma_test.cpp`

### Newly Identified Dead Files (12 files)
These headers are never `#include`d anywhere:
- `src/torchscience/csrc/impl/special_functions/factorial_backward.h`
- `src/torchscience/csrc/autocast/reduction_operators.h`
- `src/torchscience/csrc/autograd/reduction_operators.h`
- `src/torchscience/csrc/cpu/identity_operators.h`
- `src/torchscience/csrc/cpu/flatten_operators.h`
- `src/torchscience/csrc/cpu/batched_operators.h`
- `src/torchscience/csrc/cpu/reduction_operators.h`
- `src/torchscience/csrc/cpu/pairwise_operators.h`
- `src/torchscience/csrc/cpu/fixed_operators.h`
- `src/torchscience/csrc/cpu/dynamic_operators.h`
- `src/torchscience/csrc/meta/reduction_operators.h`
- `src/torchscience/csrc/meta/fixed_operators.h`

---

## Task 1: Commit Already-Staged Deletions

**Files:**
- Stage: All already-deleted files (11 files)

**Step 1: Verify staged deletions**

Run: `git -C /Users/goodmaa3/com/github/0x00b1/torchscience status --short`
Expected: Shows `D` prefix for deleted files

**Step 2: Stage the deletions and modified CMakeLists**

```bash
git -C /Users/goodmaa3/com/github/0x00b1/torchscience add -A
```

**Step 3: Commit**

```bash
git -C /Users/goodmaa3/com/github/0x00b1/torchscience commit -m "chore: remove dead code - polygamma, sine_wave, and unused traits"
```

---

## Task 2: Delete Unused Template Headers (CPU)

**Files:**
- Delete: `src/torchscience/csrc/cpu/identity_operators.h`
- Delete: `src/torchscience/csrc/cpu/flatten_operators.h`
- Delete: `src/torchscience/csrc/cpu/batched_operators.h`
- Delete: `src/torchscience/csrc/cpu/reduction_operators.h`
- Delete: `src/torchscience/csrc/cpu/pairwise_operators.h`
- Delete: `src/torchscience/csrc/cpu/fixed_operators.h`
- Delete: `src/torchscience/csrc/cpu/dynamic_operators.h`

**Step 1: Delete the files**

```bash
rm /Users/goodmaa3/com/github/0x00b1/torchscience/src/torchscience/csrc/cpu/identity_operators.h
rm /Users/goodmaa3/com/github/0x00b1/torchscience/src/torchscience/csrc/cpu/flatten_operators.h
rm /Users/goodmaa3/com/github/0x00b1/torchscience/src/torchscience/csrc/cpu/batched_operators.h
rm /Users/goodmaa3/com/github/0x00b1/torchscience/src/torchscience/csrc/cpu/reduction_operators.h
rm /Users/goodmaa3/com/github/0x00b1/torchscience/src/torchscience/csrc/cpu/pairwise_operators.h
rm /Users/goodmaa3/com/github/0x00b1/torchscience/src/torchscience/csrc/cpu/fixed_operators.h
rm /Users/goodmaa3/com/github/0x00b1/torchscience/src/torchscience/csrc/cpu/dynamic_operators.h
```

**Step 2: Stage deletions**

```bash
git -C /Users/goodmaa3/com/github/0x00b1/torchscience add -A
```

**Step 3: Commit**

```bash
git -C /Users/goodmaa3/com/github/0x00b1/torchscience commit -m "chore: remove unused CPU operator templates"
```

---

## Task 3: Delete Unused Template Headers (Meta, Autocast, Autograd)

**Files:**
- Delete: `src/torchscience/csrc/meta/reduction_operators.h`
- Delete: `src/torchscience/csrc/meta/fixed_operators.h`
- Delete: `src/torchscience/csrc/autocast/reduction_operators.h`
- Delete: `src/torchscience/csrc/autograd/reduction_operators.h`

**Step 1: Delete the files**

```bash
rm /Users/goodmaa3/com/github/0x00b1/torchscience/src/torchscience/csrc/meta/reduction_operators.h
rm /Users/goodmaa3/com/github/0x00b1/torchscience/src/torchscience/csrc/meta/fixed_operators.h
rm /Users/goodmaa3/com/github/0x00b1/torchscience/src/torchscience/csrc/autocast/reduction_operators.h
rm /Users/goodmaa3/com/github/0x00b1/torchscience/src/torchscience/csrc/autograd/reduction_operators.h
```

**Step 2: Stage deletions**

```bash
git -C /Users/goodmaa3/com/github/0x00b1/torchscience add -A
```

**Step 3: Commit**

```bash
git -C /Users/goodmaa3/com/github/0x00b1/torchscience commit -m "chore: remove unused meta/autocast/autograd operator templates"
```

---

## Task 4: Delete Unused Impl Headers

**Files:**
- Delete: `src/torchscience/csrc/impl/special_functions/factorial_backward.h`

**Step 1: Delete the file**

```bash
rm /Users/goodmaa3/com/github/0x00b1/torchscience/src/torchscience/csrc/impl/special_functions/factorial_backward.h
```

**Step 2: Stage deletion**

```bash
git -C /Users/goodmaa3/com/github/0x00b1/torchscience add -A
```

**Step 3: Commit**

```bash
git -C /Users/goodmaa3/com/github/0x00b1/torchscience commit -m "chore: remove unused factorial_backward.h"
```

---

## Task 5: Clean Up Empty Directories

**Step 1: Check for empty directories**

```bash
find /Users/goodmaa3/com/github/0x00b1/torchscience/src -type d -empty 2>/dev/null
```

**Step 2: Remove any empty directories found**

If any empty directories are found (e.g., `impl/signal_processing/waveform/`), remove them:

```bash
rmdir /Users/goodmaa3/com/github/0x00b1/torchscience/src/torchscience/csrc/impl/signal_processing/waveform 2>/dev/null || true
rmdir /Users/goodmaa3/com/github/0x00b1/torchscience/src/torchscience/csrc/impl/signal_processing/window_function 2>/dev/null || true
rmdir /Users/goodmaa3/com/github/0x00b1/torchscience/src/torchscience/csrc/impl/waveform 2>/dev/null || true
```

Note: Git doesn't track empty directories, so no commit needed for this step.

---

## Task 6: Build Verification

**Step 1: Run Python tests to verify nothing is broken**

```bash
cd /Users/goodmaa3/com/github/0x00b1/torchscience && uv run pytest tests/torchscience -x -q --tb=short 2>&1 | head -50
```

Expected: Tests pass or show only pre-existing failures.

**Step 2: Verify the package can be imported**

```bash
cd /Users/goodmaa3/com/github/0x00b1/torchscience && uv run python -c "import torchscience; print('OK')"
```

Expected: `OK`

---

## Task 7: Final Summary Commit (Optional)

If you want a single squashed commit instead of multiple:

**Step 1: Interactive rebase to squash commits**

```bash
git -C /Users/goodmaa3/com/github/0x00b1/torchscience rebase -i HEAD~4
```

Mark all but the first commit as `squash`, then edit the message to:

```
chore: eliminate dead code

Remove 23 unused files:
- 11 already-deleted files (polygamma, sine_wave, unused traits)
- 7 CPU operator templates (identity, flatten, batched, reduction, etc.)
- 4 meta/autocast/autograd operator templates
- 1 impl header (factorial_backward.h)

These files were never #include'd or used anywhere in the codebase.
```

---

## Summary

| Phase | Files Removed | Description |
|-------|---------------|-------------|
| Task 1 | 11 | Already-staged deletions |
| Task 2 | 7 | Unused CPU operator templates |
| Task 3 | 4 | Unused meta/autocast/autograd templates |
| Task 4 | 1 | Unused impl header |
| **Total** | **23** | **Dead files eliminated** |
