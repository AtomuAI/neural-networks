// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_INT32_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_INT32_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/neural/int.hpp"

#ifdef INT32
#ifdef NEURAL_INT32

namespace nn
{
    typedef Int<i32> ni32;

    template<typename T>
    struct is_ni32
    {
        static const bool value = false;
    };
    template<>
    struct is_ni32<ni32>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // NEURAL_INT32
#endif // INT32
#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_INT32_HPP_
