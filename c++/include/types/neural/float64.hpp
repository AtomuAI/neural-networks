// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_FLOAT64_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_FLOAT64_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/neural/float.hpp"

#ifdef FLOAT64
#ifdef NEURAL_FLOAT64

namespace nn
{
    typedef Float<f64> nf64;

    template<typename T>
    struct is_nf64
    {
        static const bool value = false;
    };
    template<>
    struct is_nf64<nf64>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // NEURAL_FLOAT64
#endif // FLOAT64
#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_FLOAT64_HPP_
