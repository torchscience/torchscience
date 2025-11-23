#include "../../../special_functions.h"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace science::ops::quantized::cpu {

at::Tensor
hypergeometric_2_f_1_forward_kernel(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const at::Tensor& z
) {
    TORCH_CHECK(
        z.is_quantized(),
        "z must be a quantized tensor for QuantizedCPU dispatch"
    );

    const auto a_dequant = a.is_quantized() ? a.dequantize() : a;
    const auto b_dequant = b.is_quantized() ? b.dequantize() : b;
    const auto c_dequant = c.is_quantized() ? c.dequantize() : c;
    const auto z_dequant = z.dequantize();

    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
        "torchscience::hypergeometric_2_f_1",
        ""
    ).typed<at::Tensor(
        const at::Tensor &,
        const at::Tensor &,
        const at::Tensor &,
        const at::Tensor &
    )>();

    const at::Tensor result_float = op.call(
        a_dequant,
        b_dequant,
        c_dequant,
        z_dequant
    );

    const auto qscheme = z.qscheme();

    if (qscheme == at::kPerTensorAffine) {
        return quantize_per_tensor(
            result_float,
            z.q_scale(),
            z.q_zero_point(),
            z.scalar_type()
        );
    }

    if (qscheme == at::kPerChannelAffine) {
        return quantize_per_channel(
            result_float, z.q_per_channel_scales(),
            z.q_per_channel_zero_points(),
            z.q_per_channel_axis(),
            z.scalar_type()
        );
    }

    TORCH_CHECK(
        false,
        "Unsupported quantization scheme for hypergeometric_2_f_1"
    );
}

} // namespace science::ops::quantized::cpu

TORCH_LIBRARY_IMPL(torchscience, QuantizedCPU, module) {
    module.impl(
        TORCH_SELECTIVE_NAME(
            "torchscience::hypergeometric_2_f_1"
        ),
        TORCH_FN(
            science::ops::quantized::cpu::hypergeometric_2_f_1_forward_kernel
        )
    );
}
