#pragma once

#include <boost/math/special_functions/factorials.hpp>
#include <cmath>
#include <tuple>

namespace torchscience::impl::special_functions {

// Stirling numbers of the second kind S(n, k)
// These count the number of ways to partition n elements into k non-empty subsets
template <typename T>
T stirling_number_s_2(T n, T k) {
    unsigned int n_int = static_cast<unsigned int>(n);
    unsigned int k_int = static_cast<unsigned int>(k);
    return static_cast<T>(boost::math::stirling_second(n_int, k_int));
}

template <typename T>
std::tuple<T, T> stirling_number_s_2_backward(T n, T k) {
    // Stirling numbers are defined for discrete n and k
    // Gradients are not well-defined
    return std::make_tuple(T(0), T(0));
}

} // namespace torchscience::impl::special_functions
