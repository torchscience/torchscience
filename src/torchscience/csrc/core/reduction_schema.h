#pragma once

#include <string>
#include <sstream>

#include <torch/library.h>

namespace torchscience::core {

// ============================================================================
// Schema string generators for reduction operators
// These generate the TORCH_LIBRARY schema definitions
// ============================================================================

// Schema generator for reduction operators with extra args
struct ReductionSchema {
    // Forward: op(Tensor input, int[]? dim, bool keepdim, <extra_args>) -> Tensor
    static std::string forward(const char* name, const char* extra_args) {
        std::ostringstream ss;
        ss << name << "(Tensor input, int[]? dim, bool keepdim";
        if (extra_args && extra_args[0]) {
            ss << ", " << extra_args;
        }
        ss << ") -> Tensor";
        return ss.str();
    }

    // Backward: op_backward(Tensor grad_output, Tensor input, int[]? dim, bool keepdim, <extra_args>) -> Tensor
    static std::string backward(const char* name, const char* extra_args) {
        std::ostringstream ss;
        ss << name << "_backward(Tensor grad_output, Tensor input, int[]? dim, bool keepdim";
        if (extra_args && extra_args[0]) {
            ss << ", " << extra_args;
        }
        ss << ") -> Tensor";
        return ss.str();
    }

    // Backward-backward: op_backward_backward(...) -> (Tensor, Tensor)
    static std::string backward_backward(const char* name, const char* extra_args) {
        std::ostringstream ss;
        ss << name << "_backward_backward(Tensor grad_grad_input, Tensor grad_output, "
           << "Tensor input, int[]? dim, bool keepdim";
        if (extra_args && extra_args[0]) {
            ss << ", " << extra_args;
        }
        ss << ") -> (Tensor, Tensor)";
        return ss.str();
    }
};

// Helper to register all schemas for a reduction operator
inline void register_reduction_schema(
    torch::Library& m,
    const char* name,
    const char* extra_args
) {
    m.def(ReductionSchema::forward(name, extra_args).c_str());
    m.def(ReductionSchema::backward(name, extra_args).c_str());
    m.def(ReductionSchema::backward_backward(name, extra_args).c_str());
}

}  // namespace torchscience::core

// Note: extra_count and impl are unused by schema registration but are part
// of the X-macro signature for use by other macros (e.g., kernel registration)
#define DEFINE_REDUCTION_SCHEMA(m, name, extra_args, extra_count, impl) \
    ::torchscience::core::register_reduction_schema(m, #name, extra_args)

