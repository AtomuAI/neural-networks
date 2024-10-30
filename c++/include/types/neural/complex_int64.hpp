// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_INT64_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_INT64_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/neural/int64.hpp"
#include "bewusstsein_neural_networks/c++/include/types/complex.hpp"

#ifdef INT64
#ifdef COMPLEX_NEURAL_INT64

namespace nn
{
    typedef Complex<ni64> cni64;

    template<typename T>
    struct is_cni64
    {
        static const bool value = false;
    };
    template<>
    struct is_cni64<cni64>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // COMPLEX_NEURAL_INT64
#endif // INT64
#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_INT64_HPP_
