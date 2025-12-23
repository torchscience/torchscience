#pragma once

#include <vector>
#include <ATen/ATen.h>
#include <torch/library.h>
#include "../core/creation_common.h"

namespace torchscience::meta {

// MetaCreationTraits only needs output_shape - no kernel dispatch
template<typename CreationTraits>
struct MetaCreationOperator {
    template<typename... Args>
    static at::Tensor forward(
        Args... args,
        const c10::optional<at::ScalarType>& dtype,
        const c10::optional<at::Layout>& layout,
        const c10::optional<at::Device>& device,
        bool requires_grad
    ) {
        std::vector<int64_t> shape_vec = CreationTraits::output_shape(args...);
        ::torchscience::core::check_size_nonnegative(shape_vec, "meta_creation_op");

        auto options = at::TensorOptions()
            .dtype(dtype.value_or(
                c10::typeMetaToScalarType(at::get_default_dtype())
            ))
            .layout(layout.value_or(at::kStrided))
            .device(at::kMeta)
            .requires_grad(requires_grad);

        return at::empty(shape_vec, options);
    }
};

#define REGISTER_META_CREATION(module, name, Traits, ...) \
    module.impl(#name, &::torchscience::meta::MetaCreationOperator<Traits>::forward<__VA_ARGS__>)

}  // namespace torchscience::meta
