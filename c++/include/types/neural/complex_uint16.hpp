// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_UINT16_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_UINT16_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/neural/uint16.hpp"
#include "bewusstsein_neural_networks/c++/include/types/complex.hpp"

#ifdef UINT16
#ifdef COMPLEX_NEURAL_UINT16

namespace nn
{
    typedef Complex<nu16> cnu16;

    template<typename T>
    struct is_cnu16
    {
        static const bool value = false;
    };
    template<>
    struct is_cnu16<cnu16>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // COMPLEX_NEURAL_UINT16
#endif // UINT16
#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_UINT16_HPP_
