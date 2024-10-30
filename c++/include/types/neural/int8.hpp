// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_INT8_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_INT8_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/neural/int.hpp"

#ifdef INT8
#ifdef NEURAL_INT8

namespace nn
{
    typedef Int<i8> ni8;

    template<typename T>
    struct is_ni8
    {
        static const bool value = false;
    };
    template<>
    struct is_ni8<ni8>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // NEURAL_INT8
#endif // INT8
#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_INT8_HPP_
