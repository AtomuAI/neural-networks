// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_DIM_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_DIM_HPP_

//: C Headers
#include <cstdint>
#include <cmath>

//: C++ Headers
#include <vector>

//: Types Header
#include "bewusstsein_neural_networks/c++/include/types.hpp"

namespace nn
{
    using Dim = i64;
    using Idx = u64;
    using Size = u64;
    template <u8 N>
    using DimND = std::array<Dim, N>;
    using Dim2D = std::array<Dim, 2>;
    using Dim3D = std::array<Dim, 3>;
    using Dim4D = std::array<Dim, 4>;
    using Dim5D = std::array<Dim, 5>;
} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_DIM_HPP_
