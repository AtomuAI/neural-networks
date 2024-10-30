// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_FLOAT32_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_FLOAT32_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/neural/float32.hpp"
#include "bewusstsein_neural_networks/c++/include/types/complex.hpp"

#ifdef FLOAT32
#ifdef COMPLEX_NEURAL_FLOAT32

namespace nn
{
    typedef Complex<nf32> cnf32;

    template<typename T>
    struct is_cnf32
    {
        static const bool value = false;
    };
    template<>
    struct is_cnf32<cnf32>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // COMPLEX_NEURAL_FLOAT32
#endif // FLOAT32
#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_FLOAT32_HPP_
