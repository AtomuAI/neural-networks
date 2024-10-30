// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_UINT64_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_UINT64_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/neural/uint64.hpp"
#include "bewusstsein_neural_networks/c++/include/types/complex.hpp"

#ifdef UINT64
#ifdef COMPLEX_NEURAL_UINT64

namespace nn
{
    typedef Complex<nu64> cnu64;

    template<typename T>
    struct is_cnu64
    {
        static const bool value = false;
    };
    template<>
    struct is_cnu64<cnu64>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // COMPLEX_NEURAL_UINT64
#endif // UINT64
#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_UINT64_HPP_
