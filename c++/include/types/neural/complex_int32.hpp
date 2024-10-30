// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_INT32_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_INT32_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/neural/int32.hpp"
#include "bewusstsein_neural_networks/c++/include/types/complex.hpp"

#ifdef INT32
#ifdef COMPLEX_NEURAL_INT32

namespace nn
{
    typedef Complex<ni32> cni32;

    template<typename T>
    struct is_cni32
    {
        static const bool value = false;
    };
    template<>
    struct is_cni32<cni32>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // COMPLEX_NEURAL_INT32
#endif // INT32
#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_INT32_HPP_
