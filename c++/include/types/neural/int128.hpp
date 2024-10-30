// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_INT128_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_INT128_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/neural/int.hpp"

#ifdef INT128
#ifdef NEURAL_INT128

namespace nn
{
    typedef Int<i128> ni128;

    template<typename T>
    struct is_ni128
    {
        static const bool value = false;
    };
    template<>
    struct is_ni128<ni128>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // NEURAL_INT128
#endif // INT128
#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_INT128_HPP_
