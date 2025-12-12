#pragma once

#include <boost/math/special_functions/factorials.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

// Unsigned Stirling numbers of the first kind |s(n, k)|
// These count the number of permutations of n elements with k disjoint cycles
template <typename T>
T stirling_number_s_1(T n, T k) {
    unsigned int n_int = static_cast<unsigned int>(n);
    unsigned int k_int = static_cast<unsigned int>(k);
    return static_cast<T>(boost::math::stirling_first(n_int, k_int));
}

template <typename T>
std::tuple<T, T> stirling_number_s_1_backward(T n, T k) {
    // Stirling numbers are defined for discrete n and k
    // Gradients are not well-defined
    return std::make_tuple(T(0), T(0));
}

} // namespace torchscience::impl::special_functions
