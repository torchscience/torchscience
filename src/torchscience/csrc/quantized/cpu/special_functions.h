#pragma once

#include "operators.h"

// Template-based registration (Quantized CPU operators dequant -> compute -> requant)
TORCH_LIBRARY_IMPL(torchscience, QuantizedCPU, m_quantized_cpu_special_functions) {
    REGISTER_QUANTIZED_CPU_UNARY(m_quantized_cpu_special_functions, gamma);
    REGISTER_QUANTIZED_CPU_BINARY(m_quantized_cpu_special_functions, chebyshev_polynomial_t);
    REGISTER_QUANTIZED_CPU_TERNARY(m_quantized_cpu_special_functions, incomplete_beta);
    REGISTER_QUANTIZED_CPU_QUATERNARY(m_quantized_cpu_special_functions, hypergeometric_2_f_1);
}
