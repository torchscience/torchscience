#pragma once

#include <string>
#include <sstream>

#include <torch/library.h>

namespace torchscience::core {

namespace transform_detail {

// Strip default values from a parameter string
// e.g., "int padding_mode=0, float padding_value=0.0" -> "int padding_mode, float padding_value"
inline std::string strip_defaults(const std::string& args) {
    std::string result;
    size_t i = 0;
    while (i < args.size()) {
        // Skip whitespace
        while (i < args.size() && (args[i] == ' ' || args[i] == '\t')) ++i;

        // Find the parameter (up to '=' or ',' or end)
        size_t param_start = i;
        while (i < args.size() && args[i] != '=' && args[i] != ',') ++i;

        // Trim trailing whitespace from parameter
        size_t param_end = i;
        while (param_end > param_start && (args[param_end-1] == ' ' || args[param_end-1] == '\t')) --param_end;

        // Add parameter to result
        if (!result.empty() && param_end > param_start) result += ", ";
        result += args.substr(param_start, param_end - param_start);

        // Skip default value if present
        if (i < args.size() && args[i] == '=') {
            // Skip until comma or end
            while (i < args.size() && args[i] != ',') ++i;
        }

        // Skip comma
        if (i < args.size() && args[i] == ',') ++i;
    }
    return result;
}

}  // namespace transform_detail

// ============================================================================
// Schema string generators for fixed-dimension transform operators
// These generate the TORCH_LIBRARY schema definitions
// ============================================================================

// Schema generator for fixed-dimension transform operators
struct TransformSchema {
    // Forward: op(Tensor input, int n=-1, int dim=-1, <extra_args>) -> Tensor
    static std::string forward(const char* name, const char* extra_args) {
        std::ostringstream ss;
        ss << name << "(Tensor input, int n=-1, int dim=-1";
        if (extra_args && extra_args[0]) {
            ss << ", " << extra_args;
        }
        ss << ") -> Tensor";
        return ss.str();
    }

    // Backward: op_backward(Tensor grad_output, Tensor input, int n, int dim, <extra_args_no_defaults>) -> Tensor
    static std::string backward(const char* name, const char* extra_args) {
        std::ostringstream ss;
        ss << name << "_backward(Tensor grad_output, Tensor input, int n, int dim";
        if (extra_args && extra_args[0]) {
            ss << ", " << transform_detail::strip_defaults(extra_args);
        }
        ss << ") -> Tensor";
        return ss.str();
    }

    // Backward-backward: op_backward_backward(...) -> (Tensor, Tensor)
    static std::string backward_backward(const char* name, const char* extra_args) {
        std::ostringstream ss;
        ss << name << "_backward_backward(Tensor grad_grad_input, Tensor grad_output, "
           << "Tensor input, int n, int dim";
        if (extra_args && extra_args[0]) {
            ss << ", " << transform_detail::strip_defaults(extra_args);
        }
        ss << ") -> (Tensor, Tensor)";
        return ss.str();
    }
};

// Helper to register all schemas for a transform operator
inline void register_transform_schema(
    torch::Library& m,
    const char* name,
    const char* extra_args
) {
    m.def(TransformSchema::forward(name, extra_args).c_str());
    m.def(TransformSchema::backward(name, extra_args).c_str());
    m.def(TransformSchema::backward_backward(name, extra_args).c_str());
}

}  // namespace torchscience::core

// Note: impl is unused by schema registration but is part of the X-macro signature
#define DEFINE_TRANSFORM_SCHEMA(m, name, extra_args, impl) \
    ::torchscience::core::register_transform_schema(m, #name, extra_args)
