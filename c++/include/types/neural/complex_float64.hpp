// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_FLOAT64_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_FLOAT64_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/neural/float64.hpp"
#include "bewusstsein_neural_networks/c++/include/types/complex.hpp"

#ifdef FLOAT64
#ifdef COMPLEX_NEURAL_FLOAT64

namespace nn
{
    typedef Complex<nf64> cnf64;

    template<typename T>
    struct is_cnf64
    {
        static const bool value = false;
    };
    template<>
    struct is_cnf64<cnf64>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // COMPLEX_NEURAL_FLOAT64
#endif // FLOAT64
#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_FLOAT64_HPP_
