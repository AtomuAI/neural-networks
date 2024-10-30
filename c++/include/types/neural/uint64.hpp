// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_UINT64_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_UINT64_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/types/fundamental.hpp"
#include "bewusstsein_neural_networks/c++/include/types/neural/int.hpp"

#ifdef UINT64
#ifdef NEURAL_UINT64

namespace nn
{
    typedef Int<u64> nu64;

    template<typename T>
    struct is_nu64
    {
        static const bool value = false;
    };
    template<>
    struct is_nu64<nu64>
    {
        static const bool value = true;
    };
} // namespace nn

#endif // NEURAL_UINT64
#endif // UINT64
#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_TYPES_NEURAL_UINT64_HPP_
