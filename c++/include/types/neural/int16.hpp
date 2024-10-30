// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_INT16_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_INT16_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/neural/int.hpp"

#ifdef INT16
#ifdef NEURAL_INT16

namespace nn
{
    typedef Int<i16> ni16;

    template<typename T>
    struct is_ni16
    {
        static const bool value = false;
    };
    template<>
    struct is_ni16<ni16>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // NEURAL_INT16
#endif // INT16
#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_INT16_HPP_
