#pragma once

#include <string>
#include <sstream>

#include <torch/library.h>

namespace torchscience::core {

namespace pairwise_detail {

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

}  // namespace pairwise_detail

// ============================================================================
// Schema string generators for pairwise distance operators
// These generate the TORCH_LIBRARY schema definitions
// ============================================================================

// Schema generator for pairwise distance operators
struct PairwiseSchema {
    // Forward: op(Tensor x, Tensor y, <extra_args>) -> Tensor
    static std::string forward(const char* name, const char* extra_args) {
        std::ostringstream ss;
        ss << name << "(Tensor x, Tensor y";
        if (extra_args && extra_args[0]) {
            ss << ", " << extra_args;
        }
        ss << ") -> Tensor";
        return ss.str();
    }

    // Backward: op_backward(Tensor grad_output, Tensor x, Tensor y, <extra_args>, Tensor dist_output) -> (Tensor, Tensor, Tensor)
    static std::string backward(const char* name, const char* extra_args) {
        std::ostringstream ss;
        ss << name << "_backward(Tensor grad_output, Tensor x, Tensor y";
        if (extra_args && extra_args[0]) {
            ss << ", " << pairwise_detail::strip_defaults(extra_args);
        }
        ss << ", Tensor dist_output) -> (Tensor, Tensor, Tensor)";
        return ss.str();
    }
};

// Helper to register all schemas for a pairwise distance operator
inline void register_pairwise_schema(
    torch::Library& m,
    const char* name,
    const char* extra_args
) {
    m.def(PairwiseSchema::forward(name, extra_args).c_str());
    m.def(PairwiseSchema::backward(name, extra_args).c_str());
}

}  // namespace torchscience::core

// Note: impl is unused by schema registration but is part of the X-macro signature
#define DEFINE_PAIRWISE_SCHEMA(m, name, extra_args, impl) \
    ::torchscience::core::register_pairwise_schema(m, #name, extra_args)
