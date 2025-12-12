#include <torch/library.h>

#include <torchscience/csrc/autocast/special_functions.h>
#include <torchscience/csrc/autograd/special_functions.h>
#include <torchscience/csrc/cpu/special_functions.h>
#include <torchscience/csrc/meta/special_functions.h>
#include <torchscience/csrc/quantized/cpu/special_functions.h>
#include <torchscience/csrc/sparse/coo/cpu/special_functions.h>
#include <torchscience/csrc/sparse/csr/cpu/special_functions.h>

TORCH_LIBRARY_FRAGMENT(torchscience, module) {
  module.def("_cos_pi(Tensor input) -> Tensor");
  module.def("_cos_pi_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_sin_pi(Tensor input) -> Tensor");
  module.def("_sin_pi_backward(Tensor grad_output, Tensor input) -> Tensor");
}
