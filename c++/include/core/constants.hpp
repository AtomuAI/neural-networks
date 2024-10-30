// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_CONSTANTS_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_CONSTANTS_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Types Header
#include "bewusstsein_neural_networks/c++/include/types.hpp"

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/core/static_math.hpp"

namespace nn
{
    template <typename T>
    struct e;

    template <>
    struct e<float>
    {
        static constexpr f32 value = EulersNumber<f32, 10>::value;
    };

    template <>
    struct e<double>
    {
        static constexpr f64 value = EulersNumber<f64, 20>::value;
    };
} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_CONSTANTS_HPP_
