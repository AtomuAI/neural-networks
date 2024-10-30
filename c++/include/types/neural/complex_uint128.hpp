// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_UINT128_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_UINT128_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/neural/uint128.hpp"
#include "bewusstsein_neural_networks/c++/include/types/complex.hpp"

#ifdef UINT128
#ifdef COMPLEX_NEURAL_UINT128

namespace nn
{
    typedef Complex<nu128> cnu128;

    template<typename T>
    struct is_cnu128
    {
        static const bool value = false;
    };
    template<>
    struct is_cnu128<cnu128>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // COMPLEX_NEURAL_UINT128
#endif // UINT128
#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_UINT128_HPP_
