// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_UINT16_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_UINT16_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/neural/int.hpp"

#ifdef UINT16
#ifdef NEURAL_UINT16

namespace nn
{
    typedef Int<u16> nu16;

    template<typename T>
    struct is_nu16
    {
        static const bool value = false;
    };
    template<>
    struct is_nu16<nu16>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // NEURAL_UINT16
#endif // UINT16
#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_UINT16_HPP_
