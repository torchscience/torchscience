#include "vision.h"

#include <torch/library.h>

#ifdef WITH_CUDA
#include <cuda.h>
#endif

#ifdef WITH_HIP
#include <hip/hip_runtime.h>
#endif

#if !defined(MOBILE) && defined(_WIN32)
void* PyInit__C(void) {
  return nullptr;
}
#endif // !defined(MOBILE) && defined(_WIN32)

namespace science {
int64_t cuda_version() {
#ifdef WITH_CUDA
  return CUDA_VERSION;
#else
  return -1;
#endif
}

TORCH_LIBRARY_FRAGMENT(torchscience, module) {
  module.def("_cuda_version", &cuda_version);
}
} // namespace science
