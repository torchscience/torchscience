#include "science.h"

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <Python.h>

#include <vector>

#ifdef WITH_CUDA
#include <cuda.h>
#endif

#ifdef WITH_HIP
#include <hip/hip_runtime.h>
#endif

extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
   The import from Python will load the .so consisting of this file
   in this extension, so that the TORCH_LIBRARY static initializers
   below are run. */
PyObject* PyInit__C(void) {
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_C", /* name of module */
        NULL, /* module documentation, may be NULL */
        -1,   /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
        NULL, /* methods */
    };
    return PyModule_Create(&module_def);
}
}

namespace science {
int64_t cuda_version() {
#ifdef WITH_CUDA
    return CUDA_VERSION;
#else
    return -1;
#endif
}

TORCH_LIBRARY(torchscience, module) {
    module.def("_cuda_version() -> int");

    // Register the hypergeometric_2_f_1 operator (Gaussian hypergeometric function)
    module.def("hypergeometric_2_f_1(Tensor a, Tensor b, Tensor c, Tensor z) -> Tensor");

    // Register the backward operator for hypergeometric_2_f_1
    module.def("_hypergeometric_2_f_1_backward(Tensor grad_out, Tensor a, Tensor b, Tensor c, Tensor z, Tensor result) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("_cuda_version", &cuda_version);
}

// Autograd implementation is in ops/autograd/example_kernel.cpp

}  // namespace science
