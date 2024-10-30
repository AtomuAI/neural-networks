// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_UINT8_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_UINT8_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/neural/uint8.hpp"
#include "bewusstsein_neural_networks/c++/include/types/complex.hpp"

#ifdef UINT8
#ifdef COMPLEX_NEURAL_UINT8

namespace nn
{
    typedef Complex<nu8> cnu8;

    template<typename T>
    struct is_cnu8
    {
        static const bool value = false;
    };
    template<>
    struct is_cnu8<cnu8>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // COMPLEX_NEURAL_UINT8
#endif // UINT8
#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_UINT8_HPP_
