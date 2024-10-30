// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_INT128_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_INT128_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/neural/int128.hpp"
#include "bewusstsein_neural_networks/c++/include/types/complex.hpp"

#ifdef INT128
#ifdef COMPLEX_NEURAL_INT128

namespace nn
{
    typedef Complex<ni128> cni128;

    template<typename T>
    struct is_cni128
    {
        static const bool value = false;
    };
    template<>
    struct is_cni128<cni128>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // COMPLEX_NEURAL_INT128
#endif // INT128
#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_INT128_HPP_
