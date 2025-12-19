#include <torch/extension.h>

#include "cpu/special_functions.h"
#include "cpu/window_function.h"
#include "autograd/special_functions.h"
#include "autocast/special_functions.h"
#include "meta/special_functions.h"
#include "meta/window_function.h"
#include "sparse/coo/cpu/special_functions.h"
#include "sparse/csr/cpu/special_functions.h"
#include "quantized/cpu/special_functions.h"

#ifdef TORCHSCIENCE_CUDA
#include "sparse/coo/cuda/special_functions.h"
#include "sparse/csr/cuda/special_functions.h"
#include "quantized/cuda/special_functions.h"
#endif

extern "C" {
  PyObject* PyInit__csrc(void) {
    static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      "_csrc",
      nullptr,
      -1,
      nullptr,
    };

    return PyModule_Create(&module_def);
  }
}

TORCH_LIBRARY(torchscience, m) {
  // Special functions
  m.def("chebyshev_polynomial_t(Tensor v, Tensor z) -> Tensor");
  m.def("chebyshev_polynomial_t_backward(Tensor gradient_output, Tensor v, Tensor z) -> (Tensor, Tensor)");
  m.def("chebyshev_polynomial_t_backward_backward(Tensor gradient_gradient_v, Tensor gradient_gradient_z, Tensor gradient_output, Tensor v, Tensor z) -> (Tensor, Tensor, Tensor)");

  m.def("gamma(Tensor z) -> Tensor");
  m.def("gamma_backward(Tensor grad_output, Tensor z) -> Tensor");
  m.def("gamma_backward_backward(Tensor gradient_gradient_z, Tensor gradient_output, Tensor z) -> (Tensor, Tensor)");

  m.def("hypergeometric_2_f_1(Tensor a, Tensor b, Tensor c, Tensor z) -> Tensor");
  m.def("hypergeometric_2_f_1_backward(Tensor gradient_output, Tensor a, Tensor b, Tensor c, Tensor z) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("hypergeometric_2_f_1_backward_backward(Tensor gradient_gradient_a, Tensor gradient_gradient_b, Tensor gradient_gradient_c, Tensor gradient_gradient_z, Tensor gradient_output, Tensor a, Tensor b, Tensor c, Tensor z) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

  m.def("incomplete_beta(Tensor z, Tensor a, Tensor b) -> Tensor");
  m.def("incomplete_beta_backward(Tensor gradient_output, Tensor z, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");
  m.def("incomplete_beta_backward_backward(Tensor gradient_gradient_z, Tensor gradient_gradient_a, Tensor gradient_gradient_b, Tensor gradient_output, Tensor z, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor, Tensor)");

  // Window functions (creation operators)
  m.def("rectangular_window(int m, *, ScalarType? dtype=None, Device? device=None, bool requires_grad=False) -> Tensor");
}
