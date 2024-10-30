// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_FLOAT128_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_FLOAT128_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/neural/float.hpp"

#ifdef FLOAT128
#ifdef NEURAL_FLOAT128

namespace nn
{
    typedef Float<f128> nf128;

    template<typename T>
    struct is_nf128
    {
        static const bool value = false;
    };
    template<>
    struct is_nf128<nf128>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // NEURAL_FLOAT128
#endif // FLOAT128
#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_FLOAT128_HPP_
