// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>
#include <torch/library.h>

#include "cpu/noise.h"
#include "meta/noise.h"

TORCH_LIBRARY_FRAGMENT(torchscience, m) {
  // Anchor tensor (typically 0-dim) supplies dispatch device/dtype like torch.randn.
  m.def(
      "pink_noise(Tensor anchor, int size, Generator? generator=None, "
      "ScalarType? dtype=None, Layout? layout=None, Device? device=None, "
      "bool requires_grad=False, bool? pin_memory=None) -> Tensor"
  );
}
