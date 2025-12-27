#pragma once

#include <string>
#include <sstream>

#include <torch/library.h>

namespace torchscience::core {

// ============================================================================
// Schema string generators for graphics operators
// These generate the TORCH_LIBRARY schema definitions for N-input graphics ops
// ============================================================================

// Helper to generate a comma-separated list of "Tensor tN" for N items
inline std::string make_tensor_list(const char* prefix, int count) {
    std::ostringstream ss;
    for (int i = 0; i < count; ++i) {
        if (i > 0) ss << ", ";
        ss << "Tensor " << prefix << (i + 1);
    }
    return ss.str();
}

// Schema generator for multi-input graphics operators
struct GraphicsSchema {
    // Forward: op(Tensor t1, ..., Tensor tN) -> Tensor
    static std::string forward(const char* name, int input_count) {
        std::ostringstream ss;
        ss << name << "(" << make_tensor_list("t", input_count) << ") -> Tensor";
        return ss.str();
    }

    // Backward: op_backward(Tensor grad_output, Tensor t1, ..., Tensor tN) -> (Tensor, ..., Tensor)
    static std::string backward(const char* name, int input_count) {
        std::ostringstream ss;
        ss << name << "_backward(Tensor grad_output, " << make_tensor_list("t", input_count) << ") -> (";
        for (int i = 0; i < input_count; ++i) {
            if (i > 0) ss << ", ";
            ss << "Tensor";
        }
        ss << ")";
        return ss.str();
    }

    // Backward-backward: op_backward_backward(Tensor gg_t1, ..., Tensor gg_tN, Tensor grad_output, Tensor t1, ..., Tensor tN) -> (Tensor, ..., Tensor)
    static std::string backward_backward(const char* name, int input_count) {
        std::ostringstream ss;
        ss << name << "_backward_backward(" << make_tensor_list("gg_t", input_count)
           << ", Tensor grad_output, " << make_tensor_list("t", input_count) << ") -> (";
        // N+1 outputs (grad_grad for each input + grad_grad_output)
        for (int i = 0; i < input_count + 1; ++i) {
            if (i > 0) ss << ", ";
            ss << "Tensor";
        }
        ss << ")";
        return ss.str();
    }
};

// Helper to register all schemas for a graphics operator
inline void register_graphics_schema(
    torch::Library& m,
    const char* name,
    int input_count
) {
    m.def(GraphicsSchema::forward(name, input_count).c_str());
    m.def(GraphicsSchema::backward(name, input_count).c_str());
    m.def(GraphicsSchema::backward_backward(name, input_count).c_str());
}

}  // namespace torchscience::core

// Note: impl is unused by schema registration but is part of the X-macro signature
#define DEFINE_GRAPHICS_SCHEMA(m, name, input_count, impl) \
    ::torchscience::core::register_graphics_schema(m, #name, input_count)
