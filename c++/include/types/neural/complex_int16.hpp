// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_INT16_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_INT16_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/neural/int16.hpp"
#include "bewusstsein_neural_networks/c++/include/types/complex.hpp"

#ifdef INT16
#ifdef COMPLEX_NEURAL_INT16

namespace nn
{
    typedef Complex<ni16> cni16;

    template<typename T>
    struct is_cni16
    {
        static const bool value = false;
    };
    template<>
    struct is_cni16<cni16>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // COMPLEX_NEURAL_INT16
#endif // INT16
#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_INT16_HPP_
