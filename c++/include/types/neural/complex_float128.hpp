// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_FLOAT128_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_FLOAT128_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/neural/float128.hpp"
#include "bewusstsein_neural_networks/c++/include/types/complex.hpp"

#ifdef FLOAT128
#ifdef COMPLEX_NEURAL_FLOAT128

namespace nn
{
    typedef Complex<nf128> cnf128;

    template<typename T>
    struct is_cnf128
    {
        static const bool value = false;
    };
    template<>
    struct is_cnf128<cnf128>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // COMPLEX_NEURAL_FLOAT128
#endif // FLOAT128
#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_COMPLEX_FLOAT128_HPP_
