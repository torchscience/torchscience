#pragma once

#include "operators.cuh"

// Template-based registration (Quantized CUDA operators dequant -> compute -> requant)
TORCH_LIBRARY_IMPL(torchscience, QuantizedCUDA, m_quantized_cuda_special_functions) {
    REGISTER_QUANTIZED_CUDA_UNARY(m_quantized_cuda_special_functions, gamma);
    REGISTER_QUANTIZED_CUDA_BINARY(m_quantized_cuda_special_functions, chebyshev_polynomial_t);
    REGISTER_QUANTIZED_CUDA_TERNARY(m_quantized_cuda_special_functions, incomplete_beta);
    REGISTER_QUANTIZED_CUDA_QUATERNARY(m_quantized_cuda_special_functions, hypergeometric_2_f_1);
}
