#include <torch/extension.h>

#include "cpu/special_functions.h"
#include "autograd/special_functions.h"
#include "autocast/special_functions.h"
#include "meta/special_functions.h"
#include "sparse/coo/cpu/special_functions.h"
#include "sparse/csr/cpu/special_functions.h"
#include "quantized/cpu/special_functions.h"

#ifdef TORCHSCIENCE_CUDA
#include "cuda/special_functions/chebyshev_polynomial_t.h"
#include "cuda/special_functions/gamma.h"
#include "sparse/coo/cuda/special_functions.h"
#include "sparse/csr/cuda/special_functions.h"
#include "quantized/cuda/special_functions.h"
#endif

extern "C" {
  PyObject* PyInit__csrc(void) {
    static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      "_csrc",
      NULL,
      -1,
      NULL,
    };

    return PyModule_Create(&module_def);
  }
}

TORCH_LIBRARY(torchscience, m) {
  m.def("chebyshev_polynomial_t(Tensor v, Tensor z) -> Tensor");
  m.def("chebyshev_polynomial_t_backward(Tensor grad_output, Tensor v, Tensor z) -> (Tensor, Tensor)");
  m.def("chebyshev_polynomial_t_backward_backward(Tensor ggv, Tensor ggz, Tensor grad_output, Tensor v, Tensor z) -> (Tensor, Tensor, Tensor)");

  m.def("gamma(Tensor x) -> Tensor");
  m.def("gamma_backward(Tensor grad_output, Tensor x) -> Tensor");
  m.def("gamma_backward_backward(Tensor gg_x, Tensor grad_output, Tensor x) -> (Tensor, Tensor)");
}
