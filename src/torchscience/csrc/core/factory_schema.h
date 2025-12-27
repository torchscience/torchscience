#pragma once

#include <string>
#include <sstream>

#include <torch/library.h>

namespace torchscience::core {

// ============================================================================
// Schema string generators for factory (tensor creation) operators
// These generate the TORCH_LIBRARY schema definitions for non-differentiable
// tensor creation operations that use TensorOptions.
// ============================================================================

struct FactorySchema {
    // Factory: op(int n, <extra_args>, *, TensorOptions) -> Tensor
    // The "*" separates positional from keyword-only arguments
    static std::string forward(const char* name, const char* extra_args) {
        std::ostringstream ss;
        ss << name << "(int n";
        if (extra_args && extra_args[0]) {
            ss << ", " << extra_args;
        }
        ss << ", *, ScalarType? dtype=None, Layout? layout=None, "
           << "Device? device=None, bool requires_grad=False) -> Tensor";
        return ss.str();
    }
};

// Helper to register schema for a factory operator
// Factory operators are non-differentiable, so no backward schemas needed
inline void register_factory_schema(
    torch::Library& m,
    const char* name,
    const char* extra_args
) {
    m.def(FactorySchema::forward(name, extra_args).c_str());
}

}  // namespace torchscience::core

// Note: impl is unused by schema registration but is part of the X-macro signature
#define DEFINE_FACTORY_SCHEMA(m, name, extra_args, impl) \
    ::torchscience::core::register_factory_schema(m, #name, extra_args)
