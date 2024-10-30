// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_ERROR_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_ERROR_HPP_

//: Types Header
#include "bewusstsein_neural_networks/c++/include/types.hpp"

namespace nn
{
    enum class Error : u8
    {
        NONE,
        MISMATCHED_WIDTH,
        MISMATCHED_HEIGHT,
        MISMATCHED_DEPTH,
        MISMATCHED_CHANNELS,
        MISMATCHED_BATCHES,
        MISMATCHED_SHAPES,
        MISMATCHED_VOLUMES,
        OVERSIZED_SHAPE,
        OVERSIZED_DIMENSION,
        INCORRECT_TRAINING_MODE,
        OUT_OF_BOUNDS,
    };
} // namespace nn

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_INCLUDE_CORE_ERROR_HPP_
