// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_UINT32_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_UINT32_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/neural/int.hpp"

#ifdef UINT32
#ifdef NEURAL_UINT32

namespace nn
{
    typedef Int<u32> nu32;

    template<typename T>
    struct is_nu32
    {
        static const bool value = false;
    };
    template<>
    struct is_nu32<nu32>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // NEURAL_UINT32
#endif // UINT32
#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_UINT32_HPP_
