#pragma once

#include <string>
#include <sstream>

#include <torch/library.h>

namespace torchscience::core {

// ============================================================================
// Schema string generators for pointwise operators
// These generate the TORCH_LIBRARY schema definitions
// ============================================================================

template<std::size_t Arity>
struct PointwiseSchema;

template<>
struct PointwiseSchema<1> {
    static std::string forward(const char* name) {
        std::ostringstream ss;
        ss << name << "(Tensor z) -> Tensor";
        return ss.str();
    }

    static std::string backward(const char* name) {
        std::ostringstream ss;
        ss << name << "_backward(Tensor grad_output, Tensor z) -> Tensor";
        return ss.str();
    }

    static std::string backward_backward(const char* name) {
        std::ostringstream ss;
        ss << name << "_backward_backward(Tensor gradient_gradient_z, "
           << "Tensor gradient_output, Tensor z) -> (Tensor, Tensor)";
        return ss.str();
    }
};

template<>
struct PointwiseSchema<2> {
    static std::string forward(const char* name) {
        std::ostringstream ss;
        ss << name << "(Tensor a, Tensor b) -> Tensor";
        return ss.str();
    }

    static std::string backward(const char* name) {
        std::ostringstream ss;
        ss << name << "_backward(Tensor gradient_output, Tensor a, Tensor b) "
           << "-> (Tensor, Tensor)";
        return ss.str();
    }

    static std::string backward_backward(const char* name) {
        std::ostringstream ss;
        ss << name << "_backward_backward(Tensor gradient_gradient_a, "
           << "Tensor gradient_gradient_b, Tensor gradient_output, "
           << "Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)";
        return ss.str();
    }
};

template<>
struct PointwiseSchema<3> {
    static std::string forward(const char* name) {
        std::ostringstream ss;
        ss << name << "(Tensor a, Tensor b, Tensor c) -> Tensor";
        return ss.str();
    }

    static std::string backward(const char* name) {
        std::ostringstream ss;
        ss << name << "_backward(Tensor gradient_output, Tensor a, Tensor b, "
           << "Tensor c) -> (Tensor, Tensor, Tensor)";
        return ss.str();
    }

    static std::string backward_backward(const char* name) {
        std::ostringstream ss;
        ss << name << "_backward_backward(Tensor gradient_gradient_a, "
           << "Tensor gradient_gradient_b, Tensor gradient_gradient_c, "
           << "Tensor gradient_output, Tensor a, Tensor b, Tensor c) "
           << "-> (Tensor, Tensor, Tensor, Tensor)";
        return ss.str();
    }
};

template<>
struct PointwiseSchema<4> {
    static std::string forward(const char* name) {
        std::ostringstream ss;
        ss << name << "(Tensor a, Tensor b, Tensor c, Tensor d) -> Tensor";
        return ss.str();
    }

    static std::string backward(const char* name) {
        std::ostringstream ss;
        ss << name << "_backward(Tensor gradient_output, Tensor a, Tensor b, "
           << "Tensor c, Tensor d) -> (Tensor, Tensor, Tensor, Tensor)";
        return ss.str();
    }

    static std::string backward_backward(const char* name) {
        std::ostringstream ss;
        ss << name << "_backward_backward(Tensor gradient_gradient_a, "
           << "Tensor gradient_gradient_b, Tensor gradient_gradient_c, "
           << "Tensor gradient_gradient_d, Tensor gradient_output, "
           << "Tensor a, Tensor b, Tensor c, Tensor d) "
           << "-> (Tensor, Tensor, Tensor, Tensor, Tensor)";
        return ss.str();
    }
};

// Helper to register all schemas for a pointwise operator
template<std::size_t Arity>
inline void register_pointwise_schema(torch::Library& m, const char* name) {
    m.def(PointwiseSchema<Arity>::forward(name).c_str());
    m.def(PointwiseSchema<Arity>::backward(name).c_str());
    m.def(PointwiseSchema<Arity>::backward_backward(name).c_str());
}

}  // namespace torchscience::core

#define DEFINE_POINTWISE_SCHEMA(m, name, arity) \
    ::torchscience::core::register_pointwise_schema<arity>(m, #name)
