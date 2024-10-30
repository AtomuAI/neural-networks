// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_UINT32_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_UINT32_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/neural/uint32.hpp"
#include "bewusstsein_neural_networks/c++/include/types/complex.hpp"

#ifdef UINT32
#ifdef COMPLEX_NEURAL_UINT32

namespace nn
{
    typedef Complex<nu32> cnu32;

    template<typename T>
    struct is_cnu32
    {
        static const bool value = false;
    };
    template<>
    struct is_cnu32<cnu32>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // COMPLEX_NEURAL_UINT32
#endif // UINT32
#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_UINT32_HPP_
