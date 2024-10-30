// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_FLOAT16_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_FLOAT16_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/neural/float16.hpp"
#include "bewusstsein_neural_networks/c++/include/types/complex.hpp"

#ifdef FLOAT16
#ifdef COMPLEX_NEURAL_FLOAT16

namespace nn
{
    typedef Complex<nf16> cnf16;

    template<typename T>
    struct is_cnf16
    {
        static const bool value = false;
    };
    template<>
    struct is_cnf16<cnf16>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // COMPLEX_NEURAL_FLOAT16
#endif // FLOAT16
#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_FLOAT16_HPP_
