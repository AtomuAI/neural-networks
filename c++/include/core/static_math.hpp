// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_STATIC_MATH_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_STATIC_MATH_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Types Header
#include "bewusstsein_neural_networks/c++/include/types.hpp"

namespace nn
{
    template <typename T, unsigned N>
    struct Factorial
    {
        static constexpr T value = N * Factorial<T, N - 1>::value;
    };

    template <typename T>
    struct Factorial<T, 0>
    {
        static constexpr T value = 1;
    };

    template <typename T, unsigned N>
    struct EulersNumber
    {
        static constexpr T value = 1.0 / Factorial<T, N>::value + EulersNumber<T, N - 1>::value;
    };

    template <typename T>
    struct EulersNumber<T, 0>
    {
        static constexpr T value = 1;
    };
} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_STATIC_MATH_HPP_
