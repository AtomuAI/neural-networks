// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_UINT8_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_UINT8_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/neural/int.hpp"

#ifdef UINT8
#ifdef NEURAL_UINT8

namespace nn
{
    typedef Int<u8> nu8;

    template<typename T>
    struct is_nu8
    {
        static const bool value = false;
    };
    template<>
    struct is_nu8<nu8>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // NEURAL_UINT8
#endif // UINT8
#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_UINT8_HPP_
