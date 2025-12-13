// CUDA kernels for high-use special functions
// This file is only compiled when CUDA is available (controlled by CMakeLists.txt)

// Gamma-related functions
#include <torchscience/csrc/cuda/special_functions/gamma.h>
#include <torchscience/csrc/cuda/special_functions/log_gamma.h>
#include <torchscience/csrc/cuda/special_functions/digamma.h>
#include <torchscience/csrc/cuda/special_functions/trigamma.h>
#include <torchscience/csrc/cuda/special_functions/beta.h>
#include <torchscience/csrc/cuda/special_functions/log_beta.h>

// Error functions
#include <torchscience/csrc/cuda/special_functions/error_erf.h>
#include <torchscience/csrc/cuda/special_functions/error_erfc.h>
#include <torchscience/csrc/cuda/special_functions/error_inverse_erf.h>
#include <torchscience/csrc/cuda/special_functions/error_inverse_erfc.h>

// Bessel functions
#include <torchscience/csrc/cuda/special_functions/bessel_j.h>
#include <torchscience/csrc/cuda/special_functions/bessel_y.h>
#include <torchscience/csrc/cuda/special_functions/modified_bessel_i.h>
#include <torchscience/csrc/cuda/special_functions/modified_bessel_k.h>

// Elliptic integrals
#include <torchscience/csrc/cuda/special_functions/complete_elliptic_integral_k.h>
#include <torchscience/csrc/cuda/special_functions/complete_elliptic_integral_e.h>
#include <torchscience/csrc/cuda/special_functions/jacobi_elliptic_sn.h>
#include <torchscience/csrc/cuda/special_functions/jacobi_elliptic_cn.h>
#include <torchscience/csrc/cuda/special_functions/jacobi_elliptic_dn.h>

// Trigonometric functions
#include <torchscience/csrc/cuda/special_functions/sin_pi.h>
#include <torchscience/csrc/cuda/special_functions/cos_pi.h>
