// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_UINT128_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_UINT128_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/neural/int.hpp"

#ifdef UINT128
#ifdef NEURAL_UINT128

namespace nn
{
    typedef Int<u128> nu128;

    template<typename T>
    struct is_nu128
    {
        static const bool value = false;
    };
    template<>
    struct is_nu128<nu128>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // NEURAL_UINT128
#endif // UINT128
#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_UINT128_HPP_
