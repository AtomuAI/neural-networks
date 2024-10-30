// Copyright 2024 Shane W. Mulcahy

//: C Headers
#include <cstdint>
#include <cmath>

//: C++ Headers
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <complex>
#include <utility>

//: Type Header
#include "bewusstsein_neural_networks/c++/include/types.hpp"

//: Project Headers
#include "bewusstsein_neural_networks/c++/include/core/dim.hpp"
#include "bewusstsein_neural_networks/c++/include/core/shape.hpp"

//: This Header
#include "bewusstsein_neural_networks/c++/include/core/random.hpp"

namespace nn
{
    u32* random_seed()
    {
        static u32 seed = time( 0 );
        return &seed;
    }
} // namespace nn
