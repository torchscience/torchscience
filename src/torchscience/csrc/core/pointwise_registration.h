#pragma once

#include <cstddef>
#include <string>
#include <type_traits>

#include <torch/library.h>
#include <c10/macros/Macros.h>

#include "../cpu/operators.h"
#include "../meta/operators.h"
#include "../autograd/operators.h"
#include "../autocast/operators.h"

namespace torchscience::core {

// ============================================================================
// Arity-based CPU registration
// ============================================================================

template<std::size_t Arity, typename Impl>
struct CPUPointwiseRegistrar;

template<typename Impl>
struct CPUPointwiseRegistrar<1, Impl> {
    static void register_op(torch::Library& m, const char* name) {
        ::torchscience::cpu::CPUUnaryOperator<Impl>::register_all(
            m, name,
            (std::string(name) + "_backward").c_str(),
            (std::string(name) + "_backward_backward").c_str()
        );
    }
};

template<typename Impl>
struct CPUPointwiseRegistrar<2, Impl> {
    static void register_op(torch::Library& m, const char* name) {
        ::torchscience::cpu::CPUBinaryOperator<Impl>::register_all(
            m, name,
            (std::string(name) + "_backward").c_str(),
            (std::string(name) + "_backward_backward").c_str()
        );
    }
};

template<typename Impl>
struct CPUPointwiseRegistrar<3, Impl> {
    static void register_op(torch::Library& m, const char* name) {
        ::torchscience::cpu::CPUTernaryOperator<Impl>::register_all(
            m, name,
            (std::string(name) + "_backward").c_str(),
            (std::string(name) + "_backward_backward").c_str()
        );
    }
};

template<typename Impl>
struct CPUPointwiseRegistrar<4, Impl> {
    static void register_op(torch::Library& m, const char* name) {
        ::torchscience::cpu::CPUQuaternaryOperator<Impl>::register_all(
            m, name,
            (std::string(name) + "_backward").c_str(),
            (std::string(name) + "_backward_backward").c_str()
        );
    }
};

// ============================================================================
// Arity-based Meta registration
// ============================================================================

template<std::size_t Arity>
struct MetaPointwiseRegistrar;

template<>
struct MetaPointwiseRegistrar<1> {
    static void register_op(torch::Library& m, const char* name) {
        ::torchscience::meta::MetaUnaryOperator::register_all(
            m, name,
            (std::string(name) + "_backward").c_str(),
            (std::string(name) + "_backward_backward").c_str()
        );
    }
};

template<>
struct MetaPointwiseRegistrar<2> {
    static void register_op(torch::Library& m, const char* name) {
        ::torchscience::meta::MetaBinaryOperator::register_all(
            m, name,
            (std::string(name) + "_backward").c_str(),
            (std::string(name) + "_backward_backward").c_str()
        );
    }
};

template<>
struct MetaPointwiseRegistrar<3> {
    static void register_op(torch::Library& m, const char* name) {
        ::torchscience::meta::MetaTernaryOperator::register_all(
            m, name,
            (std::string(name) + "_backward").c_str(),
            (std::string(name) + "_backward_backward").c_str()
        );
    }
};

template<>
struct MetaPointwiseRegistrar<4> {
    static void register_op(torch::Library& m, const char* name) {
        ::torchscience::meta::MetaQuaternaryOperator::register_all(
            m, name,
            (std::string(name) + "_backward").c_str(),
            (std::string(name) + "_backward_backward").c_str()
        );
    }
};

// ============================================================================
// Arity-based Autograd registration
// ============================================================================

template<std::size_t Arity, typename Impl>
struct AutogradPointwiseRegistrar;

template<typename Impl>
struct AutogradPointwiseRegistrar<1, Impl> {
    static void register_op(torch::Library& m, const char* name) {
        ::torchscience::autograd::AutogradUnaryOperator<Impl>::register_all(m, name);
    }
};

template<typename Impl>
struct AutogradPointwiseRegistrar<2, Impl> {
    static void register_op(torch::Library& m, const char* name) {
        ::torchscience::autograd::AutogradBinaryOperator<Impl>::register_all(m, name);
    }
};

template<typename Impl>
struct AutogradPointwiseRegistrar<3, Impl> {
    static void register_op(torch::Library& m, const char* name) {
        ::torchscience::autograd::AutogradTernaryOperator<Impl>::register_all(m, name);
    }
};

template<typename Impl>
struct AutogradPointwiseRegistrar<4, Impl> {
    static void register_op(torch::Library& m, const char* name) {
        ::torchscience::autograd::AutogradQuaternaryOperator<Impl>::register_all(m, name);
    }
};

// ============================================================================
// Arity-based Autocast registration
// ============================================================================

template<std::size_t Arity>
struct AutocastPointwiseRegistrar;

template<>
struct AutocastPointwiseRegistrar<1> {
    static void register_op(torch::Library& m, const char* name) {
        std::string schema = std::string("torchscience::") + name;
        ::torchscience::autocast::AutocastUnaryOperator::register_all(
            m, name, schema.c_str()
        );
    }
};

template<>
struct AutocastPointwiseRegistrar<2> {
    static void register_op(torch::Library& m, const char* name) {
        std::string schema = std::string("torchscience::") + name;
        ::torchscience::autocast::AutocastBinaryOperator::register_all(
            m, name, schema.c_str()
        );
    }
};

template<>
struct AutocastPointwiseRegistrar<3> {
    static void register_op(torch::Library& m, const char* name) {
        std::string schema = std::string("torchscience::") + name;
        ::torchscience::autocast::AutocastTernaryOperator::register_all(
            m, name, schema.c_str()
        );
    }
};

template<>
struct AutocastPointwiseRegistrar<4> {
    static void register_op(torch::Library& m, const char* name) {
        std::string schema = std::string("torchscience::") + name;
        ::torchscience::autocast::AutocastQuaternaryOperator::register_all(
            m, name, schema.c_str()
        );
    }
};

}  // namespace torchscience::core

// ============================================================================
// X-Macro expansion macros
// ============================================================================

#define REGISTER_POINTWISE_CPU(m, name, arity, impl) \
    ::torchscience::core::CPUPointwiseRegistrar<arity, impl>::register_op(m, #name)

#define REGISTER_POINTWISE_META(m, name, arity) \
    ::torchscience::core::MetaPointwiseRegistrar<arity>::register_op(m, #name)

#define REGISTER_POINTWISE_AUTOGRAD(m, name, arity, impl) \
    ::torchscience::core::AutogradPointwiseRegistrar<arity, impl>::register_op(m, #name)

#define REGISTER_POINTWISE_AUTOCAST(m, name, arity) \
    ::torchscience::core::AutocastPointwiseRegistrar<arity>::register_op(m, #name)
