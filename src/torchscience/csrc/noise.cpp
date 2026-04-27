// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>
#include <torch/library.h>

#include "cpu/noise.h"
#include "meta/noise.h"

TORCH_LIBRARY_FRAGMENT(torchscience, m) {
  // Anchor tensor (typically 0-dim) supplies dispatch device/dtype like torch.randn.
  // All five colored-noise ops share the same signature.
  m.def(
      "blue_noise(Tensor anchor, int size, Generator? generator=None, "
      "ScalarType? dtype=None, Layout? layout=None, Device? device=None, "
      "bool requires_grad=False, bool? pin_memory=None) -> Tensor"
  );
  m.def(
      "brown_noise(Tensor anchor, int size, Generator? generator=None, "
      "ScalarType? dtype=None, Layout? layout=None, Device? device=None, "
      "bool requires_grad=False, bool? pin_memory=None) -> Tensor"
  );
  // Grey noise has an extra ``sample_rate`` because A-weighting is defined
  // in absolute Hz; the rest of the signature matches the other colors.
  m.def(
      "grey_noise(Tensor anchor, int size, float sample_rate=44100.0, "
      "Generator? generator=None, "
      "ScalarType? dtype=None, Layout? layout=None, Device? device=None, "
      "bool requires_grad=False, bool? pin_memory=None) -> Tensor"
  );
  m.def(
      "pink_noise(Tensor anchor, int size, Generator? generator=None, "
      "ScalarType? dtype=None, Layout? layout=None, Device? device=None, "
      "bool requires_grad=False, bool? pin_memory=None) -> Tensor"
  );
  m.def(
      "violet_noise(Tensor anchor, int size, Generator? generator=None, "
      "ScalarType? dtype=None, Layout? layout=None, Device? device=None, "
      "bool requires_grad=False, bool? pin_memory=None) -> Tensor"
  );
  m.def(
      "white_noise(Tensor anchor, int size, Generator? generator=None, "
      "ScalarType? dtype=None, Layout? layout=None, Device? device=None, "
      "bool requires_grad=False, bool? pin_memory=None) -> Tensor"
  );
}
